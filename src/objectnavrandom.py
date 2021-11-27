import sys
import habitat


def example():
    # Note: Use with for the example testing, doesn't need to be like this on the README
    config = habitat.get_config("habitat-lab/configs/tasks/objectnav_mp3d.yaml")
    with habitat.Env(config=config) as env:
        print("Environment creation successful")
        observations = env.reset()  # noqa: F841
        print(observations)
        while not env.episode_over:
            observations = env.step(env.action_space.sample())  # noqa: F841
        print("Episode finished")


if __name__ == "__main__":
    example()
