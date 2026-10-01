"""Analysis correctness checks independent of GPU checkpoints and NAS datasets."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts/analysis'
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, '/root/nasbak/cjy/robot_raw/motus_in_depth_analysis/python_deps')
import in_depth_analysis as pipeline
import analysis_report as stats


class AnalysisTests(unittest.TestCase):
    def test_normalization_preserves_physical_qualifiers(self):
        value = pipeline.normalize_lap('Left wrist: move forward 7 cm. Right arm: rotate 10 degrees, open gripper')
        self.assertEqual(value, 'left effector: move forward 7 cm. right effector: rotate 10 degrees, open gripper')

    def test_prompt_does_not_include_ground_truth(self):
        row = dict(instruction='pick cup <setup_start>robot<setup_end><control_start>qpos<control_end>', lap_text='SECRET_GROUND_TRUTH')
        prompt = pipeline.prompt_for(row)
        self.assertNotIn('SECRET_GROUND_TRUTH', prompt)
        self.assertNotIn('robot', prompt)
        self.assertNotIn('qpos', prompt)
        self.assertIn('pick cup', prompt)

    def test_neighbors_exclude_self_and_detect_separation(self):
        x = np.array([[1, .01], [1, -.01], [-1, .01], [-1, -.01]])
        d = np.array([0, 0, 1, 1])
        score = stats.neighbors(x, d, x, k=1)
        np.testing.assert_array_equal(score['mixing'], 0)
        self.assertTrue((d[score['nearest']] != d).all())
        score = stats.neighbors(x, np.array([0, 1, 0, 1]), x, k=1)
        np.testing.assert_array_equal(score['mixing'], 1)

    def test_zero_features_rejected_and_collapse_detected(self):
        with self.assertRaises(ValueError):
            stats.unit(np.zeros((4, 3)))
        self.assertEqual(stats.feature_health(np.ones((4, 3)))['effective_rank'], 0)

    def test_paired_bootstrap_uses_differences(self):
        result = stats.interval(np.full(20, .25), np.array(['a']*5+['b']*15), 100)
        np.testing.assert_allclose(result['ci95'], [.25, .25])
        self.assertEqual(result['mean'], .25)

    def test_blank_lap_lines_are_not_removed(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp)
            lap, video = p/'lap.txt', p/'video.mp4'
            lap.write_text('move up 1 cm\n\nrotate 2 degrees\n')
            video.write_bytes(b'test')
            row = dict(source='bridge', lap_path=str(lap), video_paths=[str(video)],
                       frame_index=2, lap_index=2, episode_key='one', instruction='task')
            with patch.object(pipeline, 'observation', return_value=np.zeros((384, 320, 3), dtype=np.uint8)):
                got = pipeline.materialize(row, p)
            self.assertEqual(got['lap_text'], 'rotate 2 degrees')
            row['lap_index'] = 1
            with self.assertRaises(ValueError):
                pipeline.materialize(row, p)

    def test_sparse_ego_frames_map_to_sidecar_rows(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp)
            (p/'lap.txt').write_text('first\nsecond\nthird\n')
            (p/'video.mp4').write_bytes(b'test')
            pq.write_table(pa.table({'frame_index': [100, 103, 109], 'episode_index': [7, 7, 7]}), p/'data.parquet')
            row = dict(source='egoverse', lap_path=str(p/'lap.txt'), video_paths=[str(p/'video.mp4')],
                       data_parquet=str(p/'data.parquet'), episode_index=7, segment_start=100, segment_end=110,
                       frame_index=103, lap_index=3, episode_key='ego1', instruction='task')
            with patch.object(pipeline, 'observation', return_value=np.zeros((384, 320, 3), dtype=np.uint8)):
                got = pipeline.materialize(row, p)
            self.assertEqual(got['lap_index'], 1)
            self.assertEqual(got['lap_text'], 'second')

    def test_three_views_keep_top_left_right_layout(self):
        red = np.full((4, 8, 3), [255, 0, 0], dtype=np.uint8)
        green = np.full((4, 4, 3), [0, 255, 0], dtype=np.uint8)
        blue = np.full((4, 4, 3), [0, 0, 255], dtype=np.uint8)
        with patch.object(pipeline, 'decode_frame', side_effect=[red, green, blue]):
            x = pipeline.observation(dict(video_paths=['a', 'b', 'c'], frame_index=0))
        np.testing.assert_array_equal(x[100, 160], [255, 0, 0])
        np.testing.assert_array_equal(x[250, 80], [0, 255, 0])
        np.testing.assert_array_equal(x[250, 240], [0, 0, 255])

    def test_short_robot_episodes_do_not_always_select_initial_frame(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            task = root/'data/robot_data/example/train/task'
            for folder in ['language_action', 'instructions', 'videos']:
                (task/folder).mkdir(parents=True)
            (task/'language_action/one.txt').write_text('move up 1 cm\n'*40)
            (task/'instructions/one.txt').write_text('pick cup')
            (task/'videos/one.mp4').write_bytes(b'test')
            with patch.object(pipeline, 'ROOT', root):
                rows = list(pipeline.bridge_candidates('bridge', 'example', 42))
            self.assertGreater(rows[0]['frame_index'], 0)
            self.assertLess(rows[0]['frame_index'], 24)


if __name__ == '__main__':
    unittest.main()
