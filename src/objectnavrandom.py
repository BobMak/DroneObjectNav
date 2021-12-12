import sys
import habitat

import numpy as np
import open3d as o3d

import cv2


class Buffer:
    def __init__(self, shape, steps=5000):
        buffer_shape = (steps,) + shape
        self.buffer = np.zeros(buffer_shape, dtype=np.float32)
        self.idx = 0

    def add(self, data):
        self.buffer[self.idx] = data
        self.idx += 1


# examples/python/reconstruction_system/make_fragments.py
def register_one_rgbd_pair(source_rgbd_image, target_rgbd_image, config):

    option = o3d.pipelines.odometry.OdometryOption()
    option.max_depth_diff = config["max_depth_diff"]

    odo_init = np.identity(4)
    [success, trans, info] = o3d.pipelines.odometry.compute_rgbd_odometry(
        source_rgbd_image, target_rgbd_image, None, odo_init,
        o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm(), option)
    return [success, trans, info]


def example():
    # Note: Use with for the example testing, doesn't need to be like this on the README
    config = habitat.get_config("habitat-lab/configs/tasks/objectnav_mp3d.yaml")
    conifg_o3d = {
        "max_depth_diff": 0.01,
    }
    with habitat.Env(config=config) as env:
        print("Environment creation successful")
        observations = env.reset()  # noqa: F841
        b = Buffer(observations["depth"].shape, steps=100)
        # print(observations)
        volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=4.0 / 512.0,
            sdf_trunc=0.04,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

        obs = []
        while not env.episode_over:
            observations = env.step(env.action_space.sample())  # noqa: F841
            obs.append(observations)
            rgb =   o3d.geometry.Image(observations["rgb"])
            depth = o3d.geometry.Image(observations["depth"])
            rgbd =  o3d.geometry.RGBDImage.create_from_color_and_depth(
                rgb,
                depth,
                depth_trunc=1.0,
                convert_rgb_to_intensity=False)
            obs.append(rgbd)
            if len(obs) > 1:
                register_one_rgbd_pair(obs[-2], obs[-1], conifg_o3d)
        print("Episode finished")

        mesh = volume.extract_triangle_mesh()
        mesh.compute_vertex_normals()
        o3d.visualization.draw_geometries([mesh],
                                          front=[0.5297, -0.1873, -0.8272],
                                          lookat=[2.0712, 2.0312, 1.7251],
                                          up=[-0.0558, -0.9809, 0.1864],
                                          zoom=0.47)


if __name__ == "__main__":
    example()
