import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from data.robotwin2.robodojo_dataset import RoboDojoDataset, ACTION_SLOTS
from scripts.data.prepare_robodojo_delta24 import target_deltas, WORLD_TO_ROBOT


class RoboDojoTests(unittest.TestCase):
    def test_rotations_and_hands(self):
        a=np.zeros((3,16),dtype=np.float32)
        for start in (0,8):
            q=Rotation.from_euler('z',np.array([179,-179,-177])[:,None],degrees=True).as_quat()
            a[:,start+3:start+7]=q[:,[3,0,1,2]]
            a[:,start:start+3]=np.array([[0,0,0],[1,2,3],[2,4,6]])
            a[:,start+7]=[0.2,0.5,0.8]
        d=target_deltas(a,a)
        np.testing.assert_allclose(d[1:,0:3],[[2,-1,3]]*2)
        np.testing.assert_allclose(d[1:,5],np.deg2rad(2),atol=1e-6)
        np.testing.assert_allclose(d[:,6],a[:,7])
        np.testing.assert_allclose(d[:,13],a[:,15])
        np.testing.assert_allclose(d[0,:6],0,atol=1e-7)

    def test_target_frame_shift_padding_and_slots(self):
        with tempfile.TemporaryDirectory() as root:
            root=Path(root);(root/'qpos').mkdir();(root/'action_delta').mkdir()
            q=torch.full((3,14),0.25)
            d=torch.stack([torch.full((14,),v) for v in (0.1,0.2,0.3)])
            torch.save(q,root/'qpos/0.pt');torch.save(d,root/'action_delta/0.pt')
            ds=RoboDojoDataset.__new__(RoboDojoDataset)
            ds.stats={k:dict(min=[0]*14,max=[1]*14) for k in ('state','action')}
            state,a,_=ds._load_robot_data(root/'qpos/0.pt',[1,2,2],0)
            torch.testing.assert_close(state,q[0])
            torch.testing.assert_close(a[0,ACTION_SLOTS],d[0])
            torch.testing.assert_close(a[1,ACTION_SLOTS],d[1])
            self.assertEqual(a[2,6].item(),d[1,6].item())
            self.assertEqual(a[2,18].item(),d[1,13].item())
            self.assertFalse(a[2,:6].any());self.assertFalse(a[2,12:18].any())
            self.assertFalse(a[:,7:12].any());self.assertFalse(a[:,19:].any())
            (root/'action_joint').mkdir()
            torch.save(d,root/'action_joint/0.pt')
            ds.action_signal='action_joint'
            _,j,_=ds._load_robot_data(root/'qpos/0.pt',[1,2,2],0)
            torch.testing.assert_close(j[0,ACTION_SLOTS],d[0])
            torch.testing.assert_close(j[1,ACTION_SLOTS],d[1])
            torch.testing.assert_close(j[2],j[1])  # hold absolute joint target


if __name__=='__main__': unittest.main()
