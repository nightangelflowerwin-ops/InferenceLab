"""Prepared per-rank CUDA/NCCL collector. GPU execution is not locally verified."""
import argparse
from datetime import timedelta
import json
import os
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--out',required=True,type=Path)
    parser.add_argument('--bytes',type=int,default=1048576)
    parser.add_argument('--samples',type=int,default=10)
    parser.add_argument('--warmup',type=int,default=5)
    args=parser.parse_args()
    if args.bytes<=0 or args.bytes%4 or not 1<=args.samples<=10000 or args.warmup<0:
        parser.error('Float32-aligned positive size, bounded samples and nonnegative warmup required')
    import torch
    import torch.distributed as dist
    if not torch.cuda.is_available():
        parser.error('CUDA/NCCL required; --help and source parsing work without GPU dependencies')
    local_rank=int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    dist.init_process_group('nccl',timeout=timedelta(seconds=120))
    try:
        rank,world=dist.get_rank(),dist.get_world_size()
        if world<2:
            raise ValueError('At least two ranks required')
        data=torch.ones(args.bytes//4,device='cuda',dtype=torch.float32)
        for _ in range(args.warmup):
            data.fill_(1)
            dist.all_reduce(data)
        torch.cuda.synchronize()
        args.out.mkdir(parents=True,exist_ok=True)
        trace=args.out/f'rank-{rank}.jsonl'
        with trace.open('x',encoding='utf-8') as file:
            for sample_id in range(args.samples):
                data.fill_(1)
                dist.barrier()
                start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                start.record()
                dist.all_reduce(data)
                end.record()
                end.synchronize()
                if not torch.all(data==world).item():
                    raise RuntimeError('All-reduce numerical check failed')
                event={'run_id':args.run_id,'sample_id':sample_id,'rank':rank,'world_size':world,
                    'operation':'all_reduce','bytes':args.bytes,'duration_ms':start.elapsed_time(end),
                    'gpu':torch.cuda.get_device_name(local_rank),'torch_version':torch.__version__,
                    'evidence':'CUDA event timing on this rank; barrier is outside measured collective'}
                file.write(json.dumps(event)+'\n')
                file.flush()
    finally:
        dist.destroy_process_group()


if __name__=='__main__':
    main()
