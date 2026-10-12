"""Durable checksummed CPU fixture state and resumable event-ID streams."""
from contextlib import closing
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import socket
import sqlite3
import threading


class CorruptState(ValueError):
    pass


def encoded(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)


def integrity(kind,key,data):
    return hashlib.sha256(encoded([kind,key,data]).encode()).hexdigest()


class RecoveryStore:
    def __init__(self,path,maximum=10000):
        self.path,self.maximum=str(path),maximum
        if maximum<1:
            raise ValueError('Positive store capacity required')
        with closing(sqlite3.connect(self.path)) as db:
            db.execute('PRAGMA journal_mode=WAL')
            db.executescript('''
                CREATE TABLE IF NOT EXISTS checkpoints(id TEXT PRIMARY KEY,data TEXT,checksum TEXT);
                CREATE TABLE IF NOT EXISTS completions(id TEXT PRIMARY KEY,data TEXT,checksum TEXT);
                CREATE TABLE IF NOT EXISTS streams(id TEXT PRIMARY KEY,context_digest TEXT,data TEXT,checksum TEXT,started INTEGER);
            ''')
            db.commit()

    def read(self,kind,key):
        if kind not in ('checkpoints','completions'):
            raise ValueError('Unknown record kind')
        with closing(sqlite3.connect(self.path)) as db:
            row=db.execute(f'SELECT data,checksum FROM {kind} WHERE id=?',(key,)).fetchone()
        if not row:
            return None
        if integrity(kind,key,row[0])!=row[1]:
            raise CorruptState('Stored state checksum differs')
        return json.loads(row[0])

    def record_success(self,request_id,response,checkpoint_key,checkpoint):
        with closing(sqlite3.connect(self.path,timeout=10)) as db:
            db.execute('BEGIN IMMEDIATE')
            existing=db.execute('SELECT data,checksum FROM completions WHERE id=?',(request_id,)).fetchone()
            if existing:
                if integrity('completions',request_id,existing[0])!=existing[1]:
                    raise CorruptState('Completion integrity differs; refuse recomputation')
                old=json.loads(existing[0])
                if old['answer_checksum']!=response['answer_checksum']:
                    raise ValueError('Request ID reused with another context')
                return old
            if db.execute('SELECT count(*) FROM completions').fetchone()[0]>=self.maximum:
                raise RuntimeError('Durable completion capacity reached')
            for kind,key,value in (('checkpoints',checkpoint_key,checkpoint),('completions',request_id,response)):
                data=encoded(value)
                db.execute(f'INSERT OR REPLACE INTO {kind} VALUES(?,?,?)',(key,data,integrity(kind,key,data)))
            db.commit()
        return response

    def prepare_stream(self,request_id,context,resume_after=None):
        if not isinstance(request_id,str) or not 1<=len(request_id)<=256 or not isinstance(context,str) or len(context.encode())>16384:
            raise ValueError('Bounded request identity/context required')
        context_digest=hashlib.sha256(context.encode()).hexdigest()
        with closing(sqlite3.connect(self.path,timeout=10)) as db:
            db.execute('BEGIN IMMEDIATE')
            row=db.execute('SELECT context_digest,data,checksum,started FROM streams WHERE id=?',(request_id,)).fetchone()
            if row:
                if row[0]!=context_digest:
                    raise ValueError('Stream request identity changed')
                if integrity('streams',request_id,row[1])!=row[2]:
                    raise CorruptState('Stream integrity differs')
                tokens=json.loads(row[1])
                if row[3] and resume_after is None:
                    raise RuntimeError('Explicit last-received event ID required for stream resume')
            else:
                if resume_after not in (None,-1):
                    raise ValueError('Cannot resume an unknown stream')
                if db.execute('SELECT count(*) FROM streams').fetchone()[0]>=self.maximum:
                    raise RuntimeError('Stream ledger capacity reached')
                tokens=['CPU-fixture:']+context.split()
                data=encoded(tokens)
                db.execute('INSERT INTO streams VALUES(?,?,?,?,0)',(request_id,context_digest,data,integrity('streams',request_id,data)))
            last=-1 if resume_after is None else resume_after
            if isinstance(last,bool) or not isinstance(last,int) or not -1<=last<len(tokens):
                raise ValueError('Invalid last-received event ID')
            db.execute('UPDATE streams SET started=1 WHERE id=?',(request_id,))
            db.commit()
            return [(index,tokens[index]) for index in range(last+1,len(tokens))]


def stream_handler(store,drop_after=None):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):
            pass
        def respond(self,status,body):
            data=encoded(body).encode()
            self.send_response(status)
            self.send_header('Content-Type','application/json')
            self.send_header('Content-Length',str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        def do_POST(self):
            if self.path!='/stream':
                return self.respond(404,{'error':'Unknown path'})
            try:
                size=int(self.headers.get('Content-Length','0'))
                if not 0<size<=65536:
                    raise ValueError('Bounded body required')
                self.connection.settimeout(10)
                body=json.loads(self.rfile.read(size))
                events=store.prepare_stream(body['request_id'],body['context'],body.get('resume_after'))
            except (ValueError,KeyError,TypeError):
                return self.respond(400,{'error':'Invalid or corrupted fixture stream'})
            except RuntimeError:
                return self.respond(409,{'error':'Explicit resume position required or ledger is full'})
            self.send_response(200)
            self.send_header('Content-Type','text/event-stream')
            self.end_headers()
            try:
                for count,(index,token) in enumerate(events,1):
                    self.wfile.write(f'id: {index}\ndata: {encoded({"token":token,"fixture":True})}\n\n'.encode())
                    self.wfile.flush()
                    if drop_after is not None and count>=drop_after:
                        self.connection.shutdown(socket.SHUT_RDWR)
                        self.close_connection=True
                        return
                self.wfile.write(b'data: [DONE]\n\n')
                self.wfile.flush()
            except OSError:
                self.close_connection=True
    return Handler


def network_recovery_demo(path):
    import urllib.request
    servers,threads=[],[]
    received=[]
    try:
        for drop_after in (2,None):
            server=ThreadingHTTPServer(('127.0.0.1',0),stream_handler(RecoveryStore(path),drop_after))
            servers.append(server)
            thread=threading.Thread(target=server.serve_forever)
            threads.append(thread)
            thread.start()
        complete=False
        for attempt,server in enumerate(servers):
            body={'request_id':'network-demo','context':'one two three four'}
            if attempt:
                body['resume_after']=received[-1][0] if received else -1
            request=urllib.request.Request(f'http://127.0.0.1:{server.server_port}/stream',data=encoded(body).encode(),headers={'Content-Type':'application/json'})
            with urllib.request.urlopen(request,timeout=5) as response:
                event_id=None
                for raw in response:
                    line=raw.decode().strip()
                    if line.startswith('id: '):
                        event_id=int(line[4:])
                    elif line.startswith('data: ') and line!='data: [DONE]':
                        received.append((event_id,json.loads(line[6:])['token']))
                    elif line=='data: [DONE]':
                        complete=True
        assert complete and [index for index,_ in received]==list(range(5))
        return {'received_events':received,'completed':complete,'replicas_used':2,
                'evidence':'Real localhost socket interruption and explicit event-ID resume via a shared durable CPU fixture ledger; no real network partition, model token recovery or GPU KV migration'}
    finally:
        for server in reversed(servers):
            server.shutdown()
            server.server_close()
        for thread in threads:
            thread.join()


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--db',required=True,type=Path)
    parser.add_argument('--out',required=True,type=Path)
    args=parser.parse_args()
    args.db.parent.mkdir(parents=True,exist_ok=True)
    report=network_recovery_demo(args.db)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('Verified interrupted fixture stream recovery across two localhost servers')
