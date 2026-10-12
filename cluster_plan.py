"""Generate unexecuted four-node Ray/vLLM and torchrun profiling plans."""
import argparse
import ipaddress
import json
from pathlib import Path
import re


def validate(config):
    nodes = config['nodes']
    if not isinstance(nodes, list) or len(nodes) != 4 or len(set(nodes)) != 4:
        raise ValueError('Four unique node addresses required')
    for node in nodes:
        address = ipaddress.ip_address(node)
        if address.version != 4 or not address.is_private or address.is_loopback or address.is_unspecified:
            raise ValueError('Explicit private IPv4 node addresses required')
    count = config['gpus_per_node']
    if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= 16:
        raise ValueError('GPU count must be 1..16 per node')
    world = len(nodes) * count
    heads = config['attention_heads']
    if isinstance(heads, bool) or not isinstance(heads, int) or heads <= 0 or heads % world:
        raise ValueError('Attention head count must divide evenly across tensor-parallel ranks')
    for field in ('model_snapshot', 'run_id', 'socket_interface', 'ib_hca'):
        if not isinstance(config[field], str) or not config[field]:
            raise ValueError('Explicit model/network/run metadata required')
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,64}', config['run_id']):
        raise ValueError('Bounded run identifier required')
    return world


def plan(config):
    world = validate(config)
    master = config['nodes'][0]
    nodes = []
    for rank, node in enumerate(config['nodes']):
        ray = ['ray', 'start', '--node-ip-address', node, '--block']
        ray += ['--head', '--port', '6379'] if rank == 0 else ['--address', f'{master}:6379']
        collector = ['torchrun', '--nnodes', '4', '--nproc-per-node', str(config['gpus_per_node']),
            '--node-rank', str(rank), '--master-addr', master, '--master-port', '29500',
            'projects/09-tensor-parallelism/collect_nccl.py', '--run-id', config['run_id'],
            '--out', 'results/nccl-traces', '--bytes', '1048576', '--samples', '10']
        nodes.append({'node':node,'node_rank':rank,'ray_argv':ray,'collector_argv':collector,
            'ray_env':{'VLLM_HOST_IP':node,'NCCL_IB_DISABLE':'0','NCCL_SOCKET_IFNAME':config['socket_interface'],
                       'NCCL_IB_HCA':config['ib_hca'],'NCCL_DEBUG':'INFO'},
            'collector_env':{'NCCL_IB_DISABLE':'0','NCCL_SOCKET_IFNAME':config['socket_interface'],
                'NCCL_IB_HCA':config['ib_hca'],'NCCL_DEBUG':'INFO'}})
    return {'world_size':world,'nodes':nodes,'serving_head_env':{'RAY_ADDRESS':f'{master}:6379','VLLM_HOST_IP':master},
        'serving_head_argv':['vllm','serve',config['model_snapshot'],
        '--distributed-executor-backend','ray','--tensor-parallel-size',str(world),
        '--host','127.0.0.1','--port','8000'],
        'evidence':'Unexecuted deployment/profiling plan. No model fit, RDMA configuration, transport choice or GPU/NCCL execution verified.',
        'assumptions':['Same immutable model snapshot and pinned software on every node',
            'Private trusted cluster networking and operator-configured RDMA/RoCE',
            'Network interface/HCA names verified on each actual node',
            'Memory fit and model-specific parallelism support checked on GPU hardware']}


def combine_trace_files(files):
    events, seen = [], set()
    for file in files:
        with Path(file).open(encoding='utf-8') as source:
            for line in source:
                if not line.strip():
                    continue
                event = json.loads(line)
                key = event['run_id'], event['sample_id'], event['rank']
                if key in seen:
                    raise ValueError('Duplicate rank sample across trace files')
                seen.add(key)
                events.append(event)
    from trace_profiler import collective_profile
    collective_profile(events)  # Reject missing ranks and invalid timing/byte units.
    return sorted(events, key=lambda event:(event['run_id'],event['sample_id'],event['rank']))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['plan','combine'])
    parser.add_argument('inputs',nargs='+',type=Path)
    parser.add_argument('--out',required=True,type=Path)
    args=parser.parse_args()
    args.out.parent.mkdir(parents=True,exist_ok=True)
    if args.action=='plan':
        if len(args.inputs)!=1:
            parser.error('One plan configuration required')
        result=plan(json.loads(args.inputs[0].read_text(encoding='utf-8')))
        args.out.write_text(json.dumps(result,indent=2),encoding='utf-8')
    else:
        events=combine_trace_files(args.inputs)
        args.out.write_text(''.join(json.dumps(event)+'\n' for event in events),encoding='utf-8')
    print('Prepared plan/validated combined trace; no remote command executed')


if __name__=='__main__':
    main()
