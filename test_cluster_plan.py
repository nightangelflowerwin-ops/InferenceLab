import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from cluster_plan import plan, combine_trace_files


class ClusterPlanTests(unittest.TestCase):
    def setUp(self):
        self.root=Path(__file__).parent
        self.config=json.loads((self.root/'projects/09-tensor-parallelism/plan-fixture.json').read_text())

    def test_four_node_launch_rank_interface_and_parallelism_contract(self):
        result=plan(self.config)
        self.assertEqual(result['world_size'],4)
        self.assertEqual([node['node_rank'] for node in result['nodes']],[0,1,2,3])
        self.assertIn('--head',result['nodes'][0]['ray_argv'])
        self.assertNotIn('--head',result['nodes'][1]['ray_argv'])
        self.assertEqual(result['nodes'][0]['collector_env']['NCCL_IB_DISABLE'],'0')
        self.assertEqual(result['serving_head_argv'][-3:],['127.0.0.1','--port','8000'])
        changed=copy.deepcopy(self.config)
        changed['attention_heads']=63
        with self.assertRaises(ValueError):
            plan(changed)
        changed=copy.deepcopy(self.config)
        changed['nodes'][0]='8.8.8.8'
        with self.assertRaises(ValueError):
            plan(changed)

    def test_collector_help_and_syntax_do_not_require_torch(self):
        source=self.root/'projects/09-tensor-parallelism/collect_nccl.py'
        compile(source.read_text(),str(source),'exec')
        result=subprocess.run([sys.executable,str(source),'--help'],capture_output=True,text=True,check=True,timeout=10)
        self.assertIn('--warmup',result.stdout)

    def test_combiner_validates_complete_rank_rounds(self):
        events=[json.loads(line) for line in (self.root/'projects/09-tensor-parallelism/fixture-collectives.jsonl').read_text().splitlines()]
        with tempfile.TemporaryDirectory() as directory:
            files=[]
            for rank in range(4):
                path=Path(directory)/f'rank-{rank}.jsonl'
                path.write_text(''.join(json.dumps(event)+'\n' for event in events if event['rank']==rank))
                files.append(path)
            self.assertEqual(len(combine_trace_files(files)),8)
            with self.assertRaises(ValueError):
                combine_trace_files(files[:-1])
            with self.assertRaises(ValueError):
                combine_trace_files(files+[files[0]])


if __name__=='__main__':
    unittest.main()
