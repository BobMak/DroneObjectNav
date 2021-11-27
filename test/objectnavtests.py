import unittest
import os


class TestObjectNav(unittest.TestCase):

    def test_config_exists(self):
        # check if config file exists
        self.assert_(os.path.exists('habitat-lab/configs/tasks/objectnav_mp3d.yaml'),
                     'config file not found. Expected full path to config file: config/objectnav/mp3d/v1/train/config.json')

    def test_scene_data_exists(self):
        # check if scene directory exists
        self.assert_(os.path.exists('data/scene_datasets/mp3d/1LXtFkjw3qL'),
                     'scene data not found. Expected a scene in: data/scene_datasets/mp3d/1LXtFkjw3qL')

    def test_dataset_exists(self):
        # check if dataset directory exists
        self.assert_(os.path.exists('data/datasets/objectnav/mp3d/v1/train/content/1LXtFkjw3qL.json.gz'),
                     'dataset not found. Expected full path to dataset contetns: '
                     'data/datasets/objectnav/mp3d/v1/train/content/1LXtFkjw3qL.json.gz')