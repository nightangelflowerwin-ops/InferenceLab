"""ETag-pinned resumable HTTP reads and checked one-shard-at-a-time CPU loading."""
import argparse
import hashlib
import http.client
import json
from pathlib import Path
import re
from urllib.parse import urlsplit
import urllib.error
import urllib.request

from weight_stream import load_stream, unique_object


def validate_url(url):
    parsed=urlsplit(url)
    if parsed.username or parsed.password or parsed.fragment or not (
        parsed.scheme=='https' or parsed.scheme=='http' and parsed.hostname in ('127.0.0.1','localhost')):
        raise ValueError('HTTPS object URL required, except for loopback fixtures')


class RangeReader:
    def __init__(self,url,max_object_bytes=257*1024*1024,max_reconnects=3,chunk_size=1024*1024):
        validate_url(url)
        if max_object_bytes<8 or not 0<=max_reconnects<=10 or chunk_size<1:
            raise ValueError('Invalid object/retry/chunk bounds')
        self.url,self.maximum,self.max_reconnects,self.chunk_size=url,max_object_bytes,max_reconnects,chunk_size
        self.position,self.length,self.etag,self.response,self.connections=0,None,None,None,0
        self.closed=False
        self._connect()

    def _close_response(self):
        if self.response is not None:
            self.response.close()
            self.response=None

    def _connect(self):
        while True:
            if self.connections>self.max_reconnects:
                raise OSError('Resumable object retry budget exhausted')
            self.connections+=1
            headers={}
            if self.etag is not None:
                headers={'Range':f'bytes={self.position}-','If-Match':self.etag}
            try:
                response=urllib.request.urlopen(urllib.request.Request(self.url,headers=headers),timeout=10)
            except urllib.error.HTTPError as error:
                status=error.code
                error.close()
                if status==412:
                    raise ValueError('Object changed during resume') from None
                if status not in (429,500,502,503,504):
                    raise
                continue
            except OSError:
                continue
            try:
                etag=response.headers.get('ETag','')
                if not re.fullmatch(r'"[^"\x00-\x1f\x7f]*"',etag):
                    raise ValueError('A strong opaque ETag is required for resume')
                count=int(response.headers['Content-Length'])
                if self.etag is None:
                    if response.status!=200 or not 8<=count<=self.maximum:
                        raise ValueError('Invalid initial object size/status')
                    self.etag,self.length=etag,count
                else:
                    match=re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)',response.headers.get('Content-Range',''))
                    if response.status!=206 or etag!=self.etag or not match:
                        raise ValueError('Server ignored range or changed object identity')
                    start,end,total=map(int,match.groups())
                    if start!=self.position or end!=self.length-1 or total!=self.length or count!=end-start+1:
                        raise ValueError('Content-Range/length differs from expected object')
                self.response=response
                return
            except (ValueError,KeyError,TypeError):
                response.close()
                raise

    def read(self,count):
        if self.closed:
            raise ValueError('Reader is closed')
        if isinstance(count,bool) or not isinstance(count,int) or count<0:
            raise ValueError('Explicit nonnegative bounded read size required')
        if count==0 or self.position==self.length:
            return b''
        requested=min(count,self.chunk_size,self.length-self.position)
        while self.position<self.length:
            if self.response is None:
                self._connect()
            try:
                block=self.response.read(requested)
            except http.client.IncompleteRead as error:
                block=error.partial
                self._close_response()
            except OSError:
                self._close_response()
                continue
            if not block:
                self._close_response()
                continue
            if len(block)>requested:
                self._close_response()
                raise ValueError('Object read exceeded requested range')
            self.position+=len(block)
            return block
        return b''

    def close(self):
        self._close_response()
        self.closed=True

    def __enter__(self):
        return self

    def __exit__(self,*_):
        self.close()


def shards(index,opener,max_payload_bytes=256*1024*1024,max_total_bytes=1024*1024*1024):
    mapping=index.get('weight_map')
    if not isinstance(mapping,dict) or not mapping or any(not isinstance(name,str) or not name for name in mapping):
        raise ValueError('Nonempty tensor-to-shard map required')
    if any(not isinstance(name,str) for name in mapping.values()):
        raise ValueError('Shard names must be strings')
    names=set(mapping.values())
    if any(not isinstance(name,str) or not re.fullmatch(r'[A-Za-z0-9_.-]+\.safetensors',name) or name.startswith('.') for name in names):
        raise ValueError('Only safe flat safetensors shard names are supported')
    declared=index.get('metadata',{}).get('total_size')
    if isinstance(declared,bool) or not isinstance(declared,int) or not 0<=declared<=max_total_bytes:
        raise ValueError('Bounded total payload size metadata required')
    checksums=index.get('shard_sha256',{})
    if not isinstance(checksums,dict) or any(name not in names or not isinstance(value,str) or not re.fullmatch(r'[0-9a-fA-F]{64}',value) for name,value in checksums.items()):
        raise ValueError('Invalid optional shard checksum map')
    total=0
    for name in sorted(names):
        with opener(name) as source:
            buffer=load_stream(source,max_payload_bytes,checksums.get(name))
        expected={tensor for tensor,shard in mapping.items() if shard==name}
        actual=set(buffer.header)-{'__metadata__'}
        if actual!=expected:
            raise ValueError('Shard tensor names differ from index')
        total+=len(buffer.payload)
        if total>declared or total>max_total_bytes:
            raise ValueError('Loaded payload exceeds index/aggregate bound')
        yield name,buffer
        del buffer
    if total!=declared:
        raise ValueError('Aggregate tensor bytes differ from index metadata')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    source=parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--url')
    source.add_argument('--index',type=Path)
    parser.add_argument('--directory',type=Path)
    parser.add_argument('--object-url-map',type=Path,help='Local JSON mapping shard names to individually signed HTTPS URLs')
    parser.add_argument('--sha256')
    parser.add_argument('--max-mib',type=int,default=256)
    parser.add_argument('--max-total-mib',type=int,default=1024)
    parser.add_argument('--out',required=True,type=Path)
    args=parser.parse_args()
    if args.max_mib<1 or args.max_total_mib<1:
        parser.error('Positive memory/object limits required')
    records=[]
    if args.url:
        with RangeReader(args.url,(args.max_mib+1)*1024*1024) as stream:
            buffer=load_stream(stream,args.max_mib*1024*1024,args.sha256)
            records.append({'payload_bytes':len(buffer.payload),'sha256':buffer.sha256,'reconnects':stream.connections-1})
    else:
        index=json.loads(args.index.read_text(encoding='utf-8'),object_pairs_hook=unique_object)
        if bool(args.directory)==bool(args.object_url_map):
            parser.error('Exactly one local directory or object URL map required for shards')
        if args.directory:
            directory=args.directory.resolve()
            def opener(name):
                path=(directory/name).resolve()
                if path.parent!=directory:
                    raise ValueError('Shard path escapes the supplied directory')
                return path.open('rb')
        else:
            urls=json.loads(args.object_url_map.read_text(encoding='utf-8'),object_pairs_hook=unique_object)
            def opener(name):
                return RangeReader(urls[name],(args.max_mib+1)*1024*1024)
        for name,buffer in shards(index,opener,args.max_mib*1024*1024,args.max_total_mib*1024*1024):
            records.append({'shard':name,'payload_bytes':len(buffer.payload),'sha256':buffer.sha256})
            del buffer
    report={'objects':records,'evidence':'CPU-memory object loading only. URLs/credentials are omitted. No S3 bucket was provisioned, GPU upload performed or model cold-start measured.'}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('Validated bounded resumable/sharded CPU loading')


if __name__=='__main__':
    main()
