import io
from contextlib import closing
import json
import math
import os
from pathlib import Path
import tempfile
import unittest
import uuid

from semantic_cache import (SemanticCache, SQLiteVectors, SQLiteResponses, RedisResponses,
                            QdrantVectors, identity, vector, resp_reply)


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        path = Path(self.temp.name) / "cache.db"
        self.now = 100
        self.index, self.responses = SQLiteVectors(path, maximum=4), SQLiteResponses(path, maximum=4)
        self.cache = SemanticCache(self.index, self.responses, lambda _: [1, 0], ttl=10, clock=lambda: self.now)
        self.body = {"model": "m", "temperature": 0,
                     "messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "Define a cat"}]}

    def test_similarity_hit_expiry_and_tenant_isolation(self):
        namespace, query = identity("a", "r", "e", self.body)
        self.assertIsNone(self.cache.lookup(namespace, query)[0])
        self.cache.put(namespace, [1, 0], {"choices": ["fixture"]})
        self.assertEqual(self.cache.lookup(namespace, "What is a cat")[0], {"choices": ["fixture"]})
        other = identity("b", "r", "e", self.body)[0]
        self.assertIsNone(self.cache.lookup(other, query)[0])
        self.now = 110
        self.assertIsNone(self.cache.lookup(namespace, query)[0])

    def test_context_revisions_settings_and_guard_fences(self):
        base = identity("a", "r", "e", self.body)[0]
        for tenant, revision, embedding_revision in (("b", "r", "e"), ("a", "r2", "e"), ("a", "r", "e2")):
            self.assertNotEqual(base, identity(tenant, revision, embedding_revision, self.body)[0])
        for change in ({"model": "m2"}, {"max_tokens": 123}):
            self.assertNotEqual(base, identity("a", "r", "e", {**self.body, **change})[0])
        for content in ("Do not define a cat", "Define 2 cats"):
            body = json.loads(json.dumps(self.body))
            body["messages"][-1]["content"] = content
            self.assertNotEqual(base, identity("a", "r", "e", body)[0])
        different_system = json.loads(json.dumps(self.body))
        different_system["messages"][0]["content"] = "Different instructions"
        self.assertNotEqual(base, identity("a", "r", "e", different_system)[0])
        for change in ({"stream": True}, {"temperature": .1}, {"tools": ["tool"]}, {"n": 2}):
            self.assertIsNone(identity("a", "r", "e", {**self.body, **change}))

    def test_threshold_is_strict_and_invalid_vectors_rejected(self):
        self.cache.put("ns", [1, 0], {"value": 1})
        strict = SemanticCache(self.index, self.responses, lambda _: [1, 0], threshold=1, clock=lambda: self.now)
        self.assertIsNone(strict.lookup("ns", "same")[0])
        far = SemanticCache(self.index, self.responses, lambda _: [0, 1], clock=lambda: self.now)
        self.assertIsNone(far.lookup("ns", "different")[0])
        for values in ([], [0, 0], [float("nan"), 1], [True, 1]):
            with self.assertRaises(ValueError):
                vector(values)
        with self.assertRaises(ValueError):
            self.index.search("ns", [1, 0, 0], self.now, .98)

    def test_orphaned_vector_is_miss_and_storage_bounded(self):
        self.index.put(str(uuid.uuid4()), "ns", [1, 0], self.now + 5)
        self.assertIsNone(self.cache.lookup("ns", "query")[0])
        for n in range(10):
            self.cache.put("ns" + str(n), [1, 0], {"n": n})
        import sqlite3
        with closing(sqlite3.connect(self.index.path)) as db:
            self.assertEqual(db.execute("SELECT count(*) FROM vectors").fetchone()[0], 4)
            self.assertEqual(db.execute("SELECT count(*) FROM responses").fetchone()[0], 4)

    def test_resp_binary_null_errors_and_bounds(self):
        self.assertEqual(resp_reply(io.BytesIO(b"$5\r\na\x00b\r\n\r\n")), b"a\x00b\r\n")
        self.assertIsNone(resp_reply(io.BytesIO(b"$-1\r\n")))
        self.assertEqual(resp_reply(io.BytesIO(b"*2\r\n+OK\r\n:3\r\n")), ["OK", 3])
        for data in (b"$2000000000\r\n", b"$5\r\nshort", b"invalid"):
            with self.assertRaises(ValueError):
                resp_reply(io.BytesIO(data))
        with self.assertRaises(OSError):
            resp_reply(io.BytesIO(b"-ERR fixture\r\n"))

    def test_labeled_reuse_evaluation_exposes_false_hit_tradeoff(self):
        from cache_evaluation import evaluate
        corpus = [{"namespace": "n", "embedding": [1, 0], "answer_id": "a"}]
        queries = [{"id": "correct", "namespace": "n", "embedding": [1, 0], "acceptable_answer_ids": ["a"]},
                   {"id": "wrong", "namespace": "n", "embedding": [1, .1], "acceptable_answer_ids": ["b"]}]
        results = evaluate(corpus, queries, [.98, .9999])["results"]
        self.assertEqual(results[0]["incorrect_reuses"], 1)
        self.assertEqual(results[1]["incorrect_reuses"], 0)
        self.assertEqual(results[1]["hits"], 1)


@unittest.skipUnless(os.environ.get("INFERENCELAB_REDIS_PORT") and os.environ.get("INFERENCELAB_QDRANT_URL"),
                     "Real Redis/Qdrant services are not configured; offline tests are separate")
class RealServiceTests(unittest.TestCase):
    def test_real_redis_qdrant_similarity_namespace_and_expiry(self):
        import time
        collection = "il_test_" + uuid.uuid4().hex
        index = QdrantVectors(os.environ["INFERENCELAB_QDRANT_URL"], collection)
        responses = RedisResponses(port=int(os.environ["INFERENCELAB_REDIS_PORT"]), prefix=collection + ":")
        cache = SemanticCache(index, responses, lambda _: [1, .001], ttl=10)
        key = None
        try:
            self.assertIsNone(cache.lookup("tenant_a", "q")[0])
            key = cache.put("tenant_a", [1, 0], {"choices": [{"message": {"content": "fixture"}}]})
            self.assertGreater(responses.command("TTL", responses.prefix + key), 0)
            self.assertEqual(cache.lookup("tenant_a", "q")[0]["choices"][0]["message"]["content"], "fixture")
            self.assertIsNone(cache.lookup("tenant_b", "q")[0])
            index.put(str(uuid.uuid4()), "expired", [1, 0], time.time() - 1)
            self.assertEqual(index.search("expired", [1, 0], time.time(), .98), [])
        finally:
            if key:
                responses.command("DEL", responses.prefix + key)
            index.request("DELETE")


if __name__ == "__main__":
    unittest.main()
