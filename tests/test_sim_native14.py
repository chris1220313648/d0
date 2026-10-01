import unittest
import numpy as np
import torch
from data.lerobot.sim_native14 import select_native, normalize_native

class Native14Tests(unittest.TestCase):
    def test_gr1_selects_both_complete_arms(self):
        x=torch.arange(44.)
        self.assertEqual(select_native(x,'gr1','action').tolist(),list(range(7))+list(range(22,29)))
    def test_single_arm_action_order(self):
        x = torch.arange(12.)
        self.assertEqual(select_native(x, 'cosmos', 'action').tolist(), list(range(7)))
        self.assertEqual(select_native(x, 'xemb', 'action').tolist(), list(range(5, 12)))

    def test_old_stats_rejected(self):
        from data.lerobot.lerobot_sim_dataset import LeRobotSimDataset
        for kind in ('cosmos', 'xemb'):
            with self.subTest(kind=kind), self.assertRaisesRegex(ValueError, 'Statistics mapping differs'):
                LeRobotSimDataset(sim_kind=kind,
                    stats_path='data/utils/robot6_sim_native14_stats.json', visual_keys=[])

    def test_xemb_state_order(self):
        self.assertEqual(select_native(torch.arange(53.),'xemb','state').tolist(),list(range(25,32))+[21,22])
    def test_shape_rejected(self):
        with self.assertRaises(ValueError): select_native(torch.zeros(43),'gr1','action')
    def test_independent_extrema_and_constant(self):
        x=torch.tensor([[2.,7.],[4.,7.]])
        y=normalize_native(x,dict(min=[2,7],max=[4,7]))
        torch.testing.assert_close(y,torch.tensor([[0.,0.],[1.,0.]]))
    def test_nonfinite_rejected(self):
        with self.assertRaises(ValueError): normalize_native(torch.tensor([float('nan')]),dict(min=[0],max=[1]))
    def test_missing_stats_rejected(self):
        with self.assertRaises(KeyError): normalize_native(torch.zeros(2),{})
    def test_padding_mask_excludes_invalid_coordinates(self):
        from data.dataset import MultiDataset
        class Child:
            def __len__(self): return 1
            def __getitem__(self, idx):
                return dict(action_sequence=torch.ones(16,7),initial_state=torch.ones(9))
        sample=MultiDataset([Child()],['cosmos'],[1],14,14)[0]
        self.assertEqual(int(sample['action_mask'].sum()),16*7)
        self.assertEqual(int(sample['state_mask'].sum()),9)
        self.assertEqual(float(sample['action_sequence'][:,7:].sum()),0)
        self.assertEqual(float(sample['initial_state'][9:].sum()),0)

if __name__=='__main__': unittest.main()

