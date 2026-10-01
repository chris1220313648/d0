import unittest

import numpy as np
import torch

from scripts.data.convert_gr1_eef24 import aligned_actions, action_stats, aggregate_action
from data.lerobot.sim_native14 import select_native, normalize_native, eef24_target_delta


class EEF24Tests(unittest.TestCase):
    def test_native_order_and_alignment(self):
        raw = np.arange(72, dtype=np.float64).reshape(3, 24)
        joints = np.zeros((3, 44))
        joints[:, 29:35] = raw[:, 12:18]
        joints[:, 7:13] = raw[:, 18:24]
        np.testing.assert_array_equal(aligned_actions(raw, joints), raw.astype(np.float32))
        with self.assertRaisesRegex(ValueError, 'align'):
            aligned_actions(raw[::-1], joints)
        with self.assertRaisesRegex(ValueError, 'shape'):
            aligned_actions(raw[:-1], joints)
        with self.assertRaisesRegex(ValueError, 'Nonfinite'):
            aligned_actions(raw * np.nan, joints)

    def test_stats_and_roundtrip(self):
        x = np.random.default_rng(0).normal(size=(17, 24)).astype(np.float32)
        x[:, -1] = 3.0
        stats = aggregate_action([action_stats(x[:3]), action_stats(x[3:])])
        direct = action_stats(x)
        for key in direct:
            np.testing.assert_allclose(stats[key], direct[key], atol=1e-12)
        values = torch.from_numpy(x)
        selected = select_native(values, 'gr1_eef24', 'action')
        torch.testing.assert_close(selected, values)
        norm = normalize_native(selected, stats)
        lo, hi = torch.tensor(stats['min']), torch.tensor(stats['max'])
        torch.testing.assert_close(norm * (hi - lo) + lo, values)
        self.assertEqual(select_native(torch.arange(44), 'gr1_eef24', 'state').tolist(),
                         list(range(7)) + list(range(22, 29)))

    def test_delta_left_arm_hand_right_arm_hand_order(self):
        current = np.zeros((1, 24), dtype=np.float32)
        current[0, :3] = [1, 2, 3]
        current[0, 6:9] = [4, 5, 6]
        current[0, 12:] = np.arange(12) + 10
        delta = eef24_target_delta(current, np.zeros_like(current))
        selected = select_native(torch.from_numpy(delta), 'gr1_eef24_delta', 'action')
        np.testing.assert_array_equal(selected[0],
            [4, 5, 6, 0, 0, 0, 16, 17, 18, 19, 20, 21,
             1, 2, 3, 0, 0, 0, 10, 11, 12, 13, 14, 15])

    def test_mixed_24d_padding(self):
        from data.dataset import MultiDataset, collate_fn
        class Child:
            def __init__(self, dim): self.dim = dim
            def __len__(self): return 1
            def __getitem__(self, idx):
                return dict(action_sequence=torch.ones(16, self.dim), initial_state=torch.ones(14))
        for dim in (7, 24):
            sample = MultiDataset([Child(dim)], ['test'], [1], 24, 14)[0]
            self.assertEqual(tuple(sample['action_sequence'].shape), (16, 24))
            self.assertEqual(int(sample['action_mask'].sum()), 16 * dim)
            self.assertFalse(sample['action_mask'][:, dim:].any())
            self.assertEqual(float(sample['action_sequence'][:, dim:].sum()), 0)

    def test_delta_rotation_wrap_and_unchanged_hands(self):
        from scipy.spatial.transform import Rotation
        previous = np.zeros((2, 24), dtype=np.float32)
        current = previous.copy()
        previous[0, 5] = np.deg2rad(179)
        current[0, 5] = np.deg2rad(-179)
        previous[1, 9:12] = [0.4, -0.7, 0.2]
        current[1, 9:12] = [-0.1, 0.5, 0.8]
        current[:, :3] = [0.01, -0.02, 0.03]
        current[:, 12:] = np.arange(12)
        delta = eef24_target_delta(current, previous)
        self.assertAlmostEqual(float(delta[0, 5]), np.deg2rad(2), places=5)
        np.testing.assert_array_equal(delta[:, 12:], current[:, 12:])
        np.testing.assert_allclose(delta[:, :3], current[:, :3]-previous[:, :3])
        for start in (3, 9):
            reconstructed = Rotation.from_rotvec(delta[:, start:start+3]) * Rotation.from_rotvec(previous[:, start:start+3])
            error = (reconstructed.inv() * Rotation.from_rotvec(current[:, start:start+3])).magnitude()
            self.assertLess(float(error.max()), 1e-6)

    def test_delta_padding_and_conditioning_boundary(self):
        a = np.zeros((3, 24), dtype=np.float32)
        a[:, 0] = [0, 0.1, 0.3]
        a[:, 6] = [1, 1.5, 1.8]
        a[:, 12:] = np.arange(3)[:, None]
        indices = [2, 2, 2]
        delta = eef24_target_delta(a[indices], a[[1, 2, 2]])
        np.testing.assert_allclose(delta[0, [0,6]], [0.2,0.3], atol=1e-6)
        np.testing.assert_array_equal(delta[1:, :12], 0)
        np.testing.assert_array_equal(delta[:, 12:], 2)
        np.testing.assert_array_equal(eef24_target_delta(a[:1], a[:1])[:, :12], 0)
        with self.assertRaises(ValueError): eef24_target_delta(a, a[:1])
        with self.assertRaises(ValueError): eef24_target_delta(a*np.nan, a)


if __name__ == '__main__':
    unittest.main()
