"""TTL response caching with namespace-filtered cosine search.

SQLite adapters run locally. Redis and Qdrant adapters use real service protocols.
Embeddings must come from an explicitly configured local embedding service.
"""
from contextlib import closing
import hashlib
import json
import math
import re
import socket
import sqlite3
import threading
import time
from urllib.parse import urlsplit
import urllib.error
import urllib.request
import uuid


def local_url(url):
    value = urlsplit(url)
    if value.scheme != "http" or value.hostname not in ("127.0.0.1", "localhost") or value.username or value.password or value.query or value.fragment:
        raise ValueError("Only loopback HTTP service URLs are supported")
    return url.rstrip("/")


def vector(values):
    if not isinstance(values, list) or not values or len(values) > 65536:
        raise ValueError("Nonempty bounded vector required")
    if any(isinstance(n, bool) or not isinstance(n, (int, float)) or not math.isfinite(n) for n in values):
        raise ValueError("Embedding contains invalid values")
    scale = max(abs(n) for n in values)
    if scale == 0:
        raise ValueError("Zero vector has no cosine similarity")
    normalized = [n / scale for n in values]
    norm = math.sqrt(sum(n * n for n in normalized))
    return [n / norm for n in normalized]


def cosine(a, b):
    if len(a) != len(b):
        raise ValueError("Embedding dimension differs")
    return max(-1, min(1, sum(x * y for x, y in zip(a, b))))


def guard(text):
    # Similar vectors can still describe opposite instructions or different quantities.
    return {"numbers": re.findall(r"[-+]?\d+(?:\.\d+)?", text),
            "negations": re.findall(r"\b(?:no|not|never|without|don't|cannot|can't)\b", text.lower())}


def identity(tenant, revision, embedding_revision, body):
    if not tenant or not revision or not embedding_revision:
        raise ValueError("Tenant and both revisions are required")
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        return None
    last = messages[-1]
    if not isinstance(last, dict) or last.get("role") != "user" or not isinstance(last.get("content"), str):
        return None
    if body.get("stream") or body.get("tools") or body.get("functions"):
        return None
    if body.get("temperature", 1) != 0 or body.get("n", 1) != 1:
        return None
    if any(not isinstance(message, dict) or not isinstance(message.get("content"), str) for message in messages):
        return None
    # Include every setting and all earlier turns; only the final query can vary.
    template = json.loads(json.dumps(body))
    template["messages"][-1]["content"] = ""
    namespace = hashlib.sha256(json.dumps([tenant, revision, embedding_revision, template, guard(last["content"])],
        sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    return namespace, last["content"]


class SQLiteVectors:
    def __init__(self, path, maximum=10000):
        self.path, self.maximum = str(path), maximum
        with closing(sqlite3.connect(self.path)) as db:
            db.execute("CREATE TABLE IF NOT EXISTS vectors(id TEXT PRIMARY KEY, namespace TEXT, embedding TEXT, expires REAL)")
            db.execute("CREATE INDEX IF NOT EXISTS vectors_namespace ON vectors(namespace)")
            db.commit()

    def put(self, key, namespace, embedding, expires):
        embedding = vector(embedding)
        with closing(sqlite3.connect(self.path, timeout=10)) as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute("INSERT OR REPLACE INTO vectors VALUES(?,?,?,?)", (key, namespace, json.dumps(embedding), expires))
            count = db.execute("SELECT count(*) FROM vectors").fetchone()[0]
            if count > self.maximum:
                db.execute("DELETE FROM vectors WHERE id IN (SELECT id FROM vectors ORDER BY expires,id LIMIT ?)", (count - self.maximum,))
            db.commit()

    def search(self, namespace, embedding, now, threshold):
        embedding = vector(embedding)
        with closing(sqlite3.connect(self.path, timeout=10)) as db:
            db.execute("DELETE FROM vectors WHERE expires<=?", (now,))
            rows = db.execute("SELECT id,embedding FROM vectors WHERE namespace=? AND expires>?", (namespace, now)).fetchall()
            db.commit()
        candidates = []
        for key, encoded in rows:
            stored = json.loads(encoded)
            score = cosine(embedding, stored)
            if score > threshold:
                candidates.append((key, score))
        return sorted(candidates, key=lambda pair: (-pair[1], pair[0]))[:8]


class SQLiteResponses:
    def __init__(self, path, maximum=10000):
        self.path, self.maximum = str(path), maximum
        with closing(sqlite3.connect(self.path)) as db:
            db.execute("CREATE TABLE IF NOT EXISTS responses(id TEXT PRIMARY KEY, response TEXT, expires REAL)")
            db.commit()

    def put(self, key, response, ttl, now):
        with closing(sqlite3.connect(self.path, timeout=10)) as db:
            db.execute("DELETE FROM responses WHERE expires<=?", (now,))
            db.execute("INSERT OR REPLACE INTO responses VALUES(?,?,?)", (key, json.dumps(response, allow_nan=False), now + ttl))
            count = db.execute("SELECT count(*) FROM responses").fetchone()[0]
            if count > self.maximum:
                db.execute("DELETE FROM responses WHERE id IN (SELECT id FROM responses ORDER BY expires,id LIMIT ?)", (count - self.maximum,))
            db.commit()

    def get(self, key, now):
        with closing(sqlite3.connect(self.path)) as db:
            row = db.execute("SELECT response FROM responses WHERE id=? AND expires>?", (key, now)).fetchone()
        return json.loads(row[0]) if row else None


def resp_reply(stream, depth=0):
    if depth > 8:
        raise ValueError("RESP nesting exceeds bound")
    line = stream.readline(4096)
    if not line.endswith(b"\r\n"):
        raise ValueError("Invalid RESP response")
    kind, value = line[:1], line[1:-2]
    if kind == b"+":
        return value.decode()
    if kind == b"-":
        raise OSError("Redis command failed")
    if kind == b":":
        return int(value)
    if kind == b"$":
        size = int(value)
        if size == -1:
            return None
        if not 0 <= size <= 2 * 1024 * 1024:
            raise ValueError("RESP bulk exceeds bound")
        result = stream.read(size)
        if len(result) != size or stream.read(2) != b"\r\n":
            raise ValueError("Truncated RESP bulk")
        return result
    if kind == b"*":
        count = int(value)
        if not 0 <= count <= 1024:
            raise ValueError("Invalid RESP array")
        return [resp_reply(stream, depth + 1) for _ in range(count)]
    raise ValueError("Unsupported RESP type")


class RedisResponses:
    def __init__(self, host="127.0.0.1", port=6379, prefix="inferencelab:", password=None):
        if host not in ("127.0.0.1", "localhost"):
            raise ValueError("This Redis adapter is loopback-only")
        self.host, self.port, self.prefix, self.password = host, port, prefix, password

    def command(self, *arguments):
        def send(sock, values):
            encoded = [value if isinstance(value, bytes) else str(value).encode() for value in values]
            frame = b"*" + str(len(encoded)).encode() + b"\r\n"
            frame += b"".join(b"$" + str(len(value)).encode() + b"\r\n" + value + b"\r\n" for value in encoded)
            sock.sendall(frame)
        with socket.create_connection((self.host, self.port), timeout=5) as sock:
            with sock.makefile("rb") as stream:
                if self.password:
                    send(sock, ("AUTH", self.password))
                    resp_reply(stream)
                send(sock, arguments)
                return resp_reply(stream)

    def put(self, key, response, ttl, now):
        envelope = json.dumps({"expires": now + ttl, "response": response}, allow_nan=False).encode()
        if len(envelope) > 2 * 1024 * 1024:
            raise ValueError("Response too large")
        self.command("SET", self.prefix + key, envelope, "EX", max(1, math.ceil(ttl)))

    def get(self, key, now):
        data = self.command("GET", self.prefix + key)
        if data is None:
            return None
        envelope = json.loads(data)
        return envelope["response"] if envelope["expires"] > now else None


class QdrantVectors:
    def __init__(self, url="http://127.0.0.1:6333", collection="inferencelab"):
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", collection):
            raise ValueError("Invalid collection name")
        self.base = local_url(url) + "/collections/" + collection
        self.dimension, self.lock = None, threading.Lock()

    def request(self, method, path="", payload=None):
        data = json.dumps(payload, allow_nan=False).encode() if payload is not None else None
        request = urllib.request.Request(self.base + path, data=data, method=method,
                                         headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=10) as response:
            return json.load(response)

    def ensure(self, dimensions):
        with self.lock:
            if self.dimension is None:
                try:
                    config = self.request("GET")["result"]["config"]["params"]["vectors"]
                    if config["size"] != dimensions or config["distance"].lower() != "cosine":
                        raise ValueError("Qdrant collection vector contract differs")
                except urllib.error.HTTPError as error:
                    if error.code != 404:
                        raise
                    error.close()
                    self.request("PUT", payload={"vectors": {"size": dimensions, "distance": "Cosine"}})
                self.request("PUT", "/index?wait=true", {"field_name": "namespace", "field_schema": "keyword"})
                self.request("PUT", "/index?wait=true", {"field_name": "expires", "field_schema": "float"})
                self.dimension = dimensions
            if self.dimension != dimensions:
                raise ValueError("Qdrant embedding dimension differs")

    def put(self, key, namespace, embedding, expires):
        embedding = vector(embedding)
        self.ensure(len(embedding))
        self.request("POST", "/points/delete?wait=true", {"filter": {"must": [
            {"key": "expires", "range": {"lte": time.time()}}]}})
        self.request("PUT", "/points?wait=true", {"points": [{"id": key, "vector": embedding,
            "payload": {"namespace": namespace, "expires": expires}}]})

    def search(self, namespace, embedding, now, threshold):
        embedding = vector(embedding)
        self.ensure(len(embedding))
        result = self.request("POST", "/points/query", {"query": embedding, "filter": {"must": [
            {"key": "namespace", "match": {"value": namespace}},
            {"key": "expires", "range": {"gt": now}}]}, "limit": 8, "score_threshold": threshold,
            "with_payload": False})["result"]["points"]
        candidates = [(point["id"], max(-1, min(1, point["score"]))) for point in result]
        return [(key, score) for key, score in candidates if score > threshold]


class HTTPEmbedder:
    def __init__(self, url, model, revision):
        if not model or not revision:
            raise ValueError("Embedding model and revision required")
        self.url, self.model, self.revision = local_url(url), model, revision

    def __call__(self, text):
        request = urllib.request.Request(self.url + "/v1/embeddings",
            data=json.dumps({"model": self.model, "input": text}).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=10) as response:
            body = json.load(response)
        return vector(body["data"][0]["embedding"])


class SemanticCache:
    def __init__(self, vectors, responses, embedder, ttl=300, threshold=.98, clock=time.time):
        if not 0 <= threshold <= 1 or not math.isfinite(ttl) or ttl <= 0:
            raise ValueError("Invalid cache policy")
        self.vectors, self.responses, self.embedder = vectors, responses, embedder
        self.ttl, self.threshold, self.clock = ttl, threshold, clock

    def lookup(self, namespace, query):
        embedding = vector(self.embedder(query))
        now = self.clock()
        for key, score in self.vectors.search(namespace, embedding, now, self.threshold):
            response = self.responses.get(key, now)
            if response is not None:
                return response, embedding, score
        return None, embedding, None

    def put(self, namespace, embedding, response):
        key, now = str(uuid.uuid4()), self.clock()
        # Publish vector only after response persistence. Inconsistent/expired state
        # causes a miss rather than returning an unbacked response.
        self.responses.put(key, response, self.ttl, now)
        self.vectors.put(key, namespace, embedding, now + self.ttl)
        return key
