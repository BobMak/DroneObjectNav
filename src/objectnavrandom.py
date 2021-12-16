import sys
import habitat

import numpy as np
import open3d as o3d

from config import ConfigParser

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
def register_one_rgbd_pair(source_rgbd_image, target_rgbd_image, intrinsic, config):

    option = o3d.pipelines.odometry.OdometryOption()
    option.max_depth_diff = 0.01  # todo what is this

    odo_init = np.identity(4)
    [success, trans, info] = o3d.pipelines.odometry.compute_rgbd_odometry(
        source_rgbd_image, target_rgbd_image, intrinsic, odo_init,
        o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm(), option)
    return [success, trans, info]


def example(conifg_o3d):
    # Note: Use with for the example testing, doesn't need to be like this on the README
    config_hab = habitat.get_config("habitat-lab/configs/tasks/objectnav_mp3d.yaml")
    # conifg_o3d = {
    #     "max_depth_diff": 0.01,
    #     "voxel_size": 0.05,
    #     "block_count": 10000,
    #     "depth_scale":
    # }
    intrinsic = o3d.camera.PinholeCameraIntrinsic()
    device = o3d.core.Device("CPU:0")
    T_frame_to_model = o3d.core.Tensor(np.identity(4))
    model = o3d.t.pipelines.slam.Model(conifg_o3d.voxel_size, 16,
                                       conifg_o3d.block_count, T_frame_to_model,
                                       device)
    with habitat.Env(config=config_hab) as env:
        print("Environment creation successful")
        observations = env.reset()  # noqa: F841
        obs = []
        # inital depth image
        depth_ref = o3d.t.geometry.Image(observations["depth"])
        input_frame = o3d.t.pipelines.slam.Frame(depth_ref.rows, depth_ref.columns,
                                                 intrinsic, device)
        raycast_frame = o3d.t.pipelines.slam.Frame(depth_ref.rows,
                                                   depth_ref.columns, intrinsic,
                                                   device)
        i = 0
        poses = []
        while not env.episode_over:
            color = o3d.t.geometry.Image(observations["rgb"]).to(device)
            depth = o3d.t.geometry.Image(observations["depth"]).to(device)

            input_frame.set_data_from_image('depth', depth)
            input_frame.set_data_from_image('color', color)

            if i > 0:
                result = model.track_frame_to_model(input_frame, raycast_frame,
                                                    conifg_o3d.depth_scale,
                                                    conifg_o3d.depth_max,
                                                    conifg_o3d.odometry_distance_thr)
                T_frame_to_model = T_frame_to_model @ result.transformation

            poses.append(T_frame_to_model.cpu().numpy())
            model.update_frame_pose(i, T_frame_to_model)
            model.integrate(input_frame, conifg_o3d.depth_scale, conifg_o3d.depth_max)
            model.synthesize_model_frame(raycast_frame, conifg_o3d.depth_scale,
                                         conifg_o3d.depth_min, conifg_o3d.depth_max, False)
            obs.append(observations)
            observations = env.step(env.action_space.sample())  # noqa: F841
            i += 1

        print("Episode finished")


if __name__ == "__main__":
    parser = ConfigParser()
    parser.add(
        '--config',
        is_config_file=True,
        help='YAML config file path. Please refer to default_config.yml as a '
             'reference. It overrides the default config file, but will be '
             'overridden by other command line inputs.')
    parser.add('--path_npz',
               help='path to the npz file that stores voxel block grid.',
               default='output.npz')
    config = parser.get_config()
    example(config)
