"""Exercise distributed argument routing without spawning training processes."""
import os
from pathlib import Path
import shlex
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / 'scripts/mutildataset/train_robot6_24a_eef_delta_4node_hypertrain.sh'


class HyperTrainLauncherTests(unittest.TestCase):
    def run_script(self, **overrides):
        env = dict(os.environ)
        for key in ('NNODES', 'NPROC_PER_NODE', 'NODE_RANK', 'MASTER_ADDR', 'MASTER_PORT',
                    'SMOKE', 'CONFIG', 'CONFIG_FILE', 'RUN_NAME', 'OUTPUT_DIR', 'PROJECT_ROOT'):
            env.pop(key, None)
        env.update(DRY_RUN='1', MASTER_ADDR='master.example.internal', MASTER_PORT='29537')
        env.update(overrides)
        return subprocess.run(['bash', str(LAUNCHER)], env=env, cwd='/tmp',
                              capture_output=True, text=True)

    def test_all_node_ranks_and_distinct_logs(self):
        for rank in range(4):
            result = self.run_script(NODE_RANK=str(rank))
            self.assertEqual(result.returncode, 0, result.stderr)
            args = shlex.split(result.stdout.splitlines()[-1])
            self.assertIn('--nnodes=4', args)
            self.assertIn('--nproc_per_node=8', args)
            self.assertIn(f'--node-rank={rank}', args)
            self.assertIn('--master-addr=master.example.internal', args)
            self.assertIn('configs/multidataset_lap_robot6_24a_norm.yaml', args)
            self.assertIn(f'train_lap_hypertrain_node{rank}.log', result.stdout)
            self.assertIn('multidataset_lap_robot6_24a_norm_lap_hypertrain', args)

    def test_missing_rank_loopback_and_invalid_rank_fail(self):
        for env in ({}, {'NODE_RANK': '0', 'MASTER_ADDR': 'localhost'},
                    {'NODE_RANK': '4'}, {'NODE_RANK': '0', 'NNODES': '0'}):
            result = self.run_script(**env)
            self.assertNotEqual(result.returncode, 0)

    def test_torchrun_symbolic_process_counts(self):
        for count in ('auto', 'gpu', 'cpu'):
            result = self.run_script(NODE_RANK='0', NPROC_PER_NODE=count)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn(f'--nproc_per_node={count}', result.stdout)
        for name in ('NNODES', 'NODE_RANK', 'MASTER_PORT'):
            env = {'NODE_RANK': '0', name: 'auto'}
            result = self.run_script(**env)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(f'{name}=auto', result.stderr)
        for count in ('0', '-1', 'invalid'):
            self.assertNotEqual(self.run_script(NODE_RANK='0', NPROC_PER_NODE=count).returncode, 0)

    def test_legacy_smoke_variable_does_not_change_training(self):
        result = self.run_script(NODE_RANK='2', SMOKE='1')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn('smoke100', result.stdout)
        self.assertNotIn('resolved_config.yaml', result.stdout)
        self.assertIn('--config configs/multidataset_lap_robot6_24a_norm.yaml', result.stdout)


if __name__ == '__main__':
    unittest.main()
