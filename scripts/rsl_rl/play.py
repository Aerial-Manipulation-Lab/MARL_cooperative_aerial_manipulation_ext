"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--plot_data", type=bool, default=None, help="Plot data of the current run")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""


import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import os
import torch

from rsl_rl.runners import OnPolicyRunner

# Import extensions to set up environment tasks
import MARL_mav_carry_ext.tasks  # noqa: F401

from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_onnx
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg


def main():
    """Play with RSL-RL agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)
    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    policy_nn = ppo_runner.alg.policy
    normalizer = getattr(policy_nn, "actor_obs_normalizer", None)

    # export policy to onnx
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_onnx(policy_nn, export_model_dir, normalizer=normalizer, filename="policy.onnx")

    # reset environment
    obs = env.get_observations()
    timestep = 0

    # create lists for action plots
    drone_1_forces = []
    drone_2_forces = []
    drone_3_forces = []
    drone_1_x_torque = []
    drone_1_y_torque = []
    drone_1_z_torque = []
    drone_2_x_torque = []
    drone_2_y_torque = []
    drone_2_z_torque = []
    drone_3_x_torque = []
    drone_3_y_torque = []
    drone_3_z_torque = []

    # create lists for payload observations
    payload_pos_x = []
    payload_pos_y = []
    payload_pos_z = []
    payload_quat_w = []
    payload_quat_x = []
    payload_quat_y = []
    payload_quat_z = []
    payload_lin_vel_x = []
    payload_lin_vel_y = []
    payload_lin_vel_z = []
    payload_ang_vel_x = []
    payload_ang_vel_y = []
    payload_ang_vel_z = []

    # create lists for drone observations
    drone_1_pos_x = []
    drone_1_pos_y = []
    drone_1_pos_z = []
    drone_1_quat_w = []
    drone_1_quat_x = []
    drone_1_quat_y = []
    drone_1_quat_z = []
    drone_1_lin_vel_x = []
    drone_1_lin_vel_y = []
    drone_1_lin_vel_z = []
    drone_1_ang_vel_x = []
    drone_1_ang_vel_y = []
    drone_1_ang_vel_z = []

    drone_2_pos_x = []
    drone_2_pos_y = []
    drone_2_pos_z = []
    drone_2_quat_w = []
    drone_2_quat_x = []
    drone_2_quat_y = []
    drone_2_quat_z = []
    drone_2_lin_vel_x = []
    drone_2_lin_vel_y = []
    drone_2_lin_vel_z = []
    drone_2_ang_vel_x = []
    drone_2_ang_vel_y = []
    drone_2_ang_vel_z = []

    drone_3_pos_x = []
    drone_3_pos_y = []
    drone_3_pos_z = []
    drone_3_quat_w = []
    drone_3_quat_x = []
    drone_3_quat_y = []
    drone_3_quat_z = []
    drone_3_lin_vel_x = []
    drone_3_lin_vel_y = []
    drone_3_lin_vel_z = []
    drone_3_ang_vel_x = []
    drone_3_ang_vel_y = []
    drone_3_ang_vel_z = []

    # create lists for cable directions
    cable_angle_1_w = []
    cable_angle_1_x = []
    cable_angle_1_y = []
    cable_angle_1_z = []

    cable_angle_2_w = []
    cable_angle_2_x = []
    cable_angle_2_y = []
    cable_angle_2_z = []

    cable_angle_3_w = []
    cable_angle_3_x = []
    cable_angle_3_y = []
    cable_angle_3_z = []

    # create lists for payload errors
    payload_pos_error_x = []
    payload_pos_error_y = []
    payload_pos_error_z = []
    payload_quat_error_w = []
    payload_quat_error_x = []
    payload_quat_error_y = []
    payload_quat_error_z = []

    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, rewards, dones, _ = env.step(actions)
            obs_flat = obs["policy"]
            timestep += 1
            if args_cli.plot_data:

                # append actions
                drone_1_forces.append(actions[:, 0].cpu().numpy())
                drone_2_forces.append(actions[:, 4].cpu().numpy())
                drone_3_forces.append(actions[:, 8].cpu().numpy())
                drone_1_x_torque.append(actions[:, 1].cpu().numpy())
                drone_1_y_torque.append(actions[:, 2].cpu().numpy())
                drone_1_z_torque.append(actions[:, 3].cpu().numpy())
                drone_2_x_torque.append(actions[:, 5].cpu().numpy())
                drone_2_y_torque.append(actions[:, 6].cpu().numpy())
                drone_2_z_torque.append(actions[:, 7].cpu().numpy())
                drone_3_x_torque.append(actions[:, 9].cpu().numpy())
                drone_3_y_torque.append(actions[:, 10].cpu().numpy())
                drone_3_z_torque.append(actions[:, 11].cpu().numpy())

                # append payload observations
                payload_pos_x.append(obs_flat[:, 0].cpu().numpy())
                payload_pos_y.append(obs_flat[:, 1].cpu().numpy())
                payload_pos_z.append(obs_flat[:, 2].cpu().numpy())
                payload_quat_w.append(obs_flat[:, 3].cpu().numpy())
                payload_quat_x.append(obs_flat[:, 4].cpu().numpy())
                payload_quat_y.append(obs_flat[:, 5].cpu().numpy())
                payload_quat_z.append(obs_flat[:, 6].cpu().numpy())
                payload_lin_vel_x.append(obs_flat[:, 7].cpu().numpy())
                payload_lin_vel_y.append(obs_flat[:, 8].cpu().numpy())
                payload_lin_vel_z.append(obs_flat[:, 9].cpu().numpy())
                payload_ang_vel_x.append(obs_flat[:, 10].cpu().numpy())
                payload_ang_vel_y.append(obs_flat[:, 11].cpu().numpy())
                payload_ang_vel_z.append(obs_flat[:, 12].cpu().numpy())

                # append drone observations
                drone_1_pos_x.append(obs_flat[:, 13].cpu().numpy())
                drone_1_pos_y.append(obs_flat[:, 14].cpu().numpy())
                drone_1_pos_z.append(obs_flat[:, 15].cpu().numpy())
                drone_2_pos_x.append(obs_flat[:, 16].cpu().numpy())
                drone_2_pos_y.append(obs_flat[:, 17].cpu().numpy())
                drone_2_pos_z.append(obs_flat[:, 18].cpu().numpy())
                drone_3_pos_x.append(obs_flat[:, 19].cpu().numpy())
                drone_3_pos_y.append(obs_flat[:, 20].cpu().numpy())
                drone_3_pos_z.append(obs_flat[:, 21].cpu().numpy())

                drone_1_quat_w.append(obs_flat[:, 22].cpu().numpy())
                drone_1_quat_x.append(obs_flat[:, 23].cpu().numpy())
                drone_1_quat_y.append(obs_flat[:, 24].cpu().numpy())
                drone_1_quat_z.append(obs_flat[:, 25].cpu().numpy())
                drone_2_quat_w.append(obs_flat[:, 26].cpu().numpy())
                drone_2_quat_x.append(obs_flat[:, 27].cpu().numpy())
                drone_2_quat_y.append(obs_flat[:, 28].cpu().numpy())
                drone_2_quat_z.append(obs_flat[:, 29].cpu().numpy())
                drone_3_quat_w.append(obs_flat[:, 30].cpu().numpy())
                drone_3_quat_x.append(obs_flat[:, 31].cpu().numpy())
                drone_3_quat_y.append(obs_flat[:, 32].cpu().numpy())
                drone_3_quat_z.append(obs_flat[:, 33].cpu().numpy())

                drone_1_lin_vel_x.append(obs_flat[:, 34].cpu().numpy())
                drone_1_lin_vel_y.append(obs_flat[:, 35].cpu().numpy())
                drone_1_lin_vel_z.append(obs_flat[:, 36].cpu().numpy())
                drone_2_lin_vel_x.append(obs_flat[:, 37].cpu().numpy())
                drone_2_lin_vel_y.append(obs_flat[:, 38].cpu().numpy())
                drone_2_lin_vel_z.append(obs_flat[:, 39].cpu().numpy())
                drone_3_lin_vel_x.append(obs_flat[:, 40].cpu().numpy())
                drone_3_lin_vel_y.append(obs_flat[:, 41].cpu().numpy())
                drone_3_lin_vel_z.append(obs_flat[:, 42].cpu().numpy())

                drone_1_ang_vel_x.append(obs_flat[:, 43].cpu().numpy())
                drone_1_ang_vel_y.append(obs_flat[:, 44].cpu().numpy())
                drone_1_ang_vel_z.append(obs_flat[:, 45].cpu().numpy())
                drone_2_ang_vel_x.append(obs_flat[:, 46].cpu().numpy())
                drone_2_ang_vel_y.append(obs_flat[:, 47].cpu().numpy())
                drone_2_ang_vel_z.append(obs_flat[:, 48].cpu().numpy())
                drone_3_ang_vel_x.append(obs_flat[:, 49].cpu().numpy())
                drone_3_ang_vel_y.append(obs_flat[:, 50].cpu().numpy())
                drone_3_ang_vel_z.append(obs_flat[:, 51].cpu().numpy())

                # append cable angles
                cable_angle_1_w.append(obs_flat[:, 92].cpu().numpy())
                cable_angle_1_x.append(obs_flat[:, 93].cpu().numpy())
                cable_angle_1_y.append(obs_flat[:, 94].cpu().numpy())
                cable_angle_1_z.append(obs_flat[:, 95].cpu().numpy())

                cable_angle_2_w.append(obs_flat[:, 96].cpu().numpy())
                cable_angle_2_x.append(obs_flat[:, 97].cpu().numpy())
                cable_angle_2_y.append(obs_flat[:, 98].cpu().numpy())
                cable_angle_2_z.append(obs_flat[:, 99].cpu().numpy())

                cable_angle_3_w.append(obs_flat[:, 100].cpu().numpy())
                cable_angle_3_x.append(obs_flat[:, 101].cpu().numpy())
                cable_angle_3_y.append(obs_flat[:, 102].cpu().numpy())
                cable_angle_3_z.append(obs_flat[:, 103].cpu().numpy())

                # append payload errors
                payload_pos_error_x.append(obs_flat[:, 52].cpu().numpy())
                payload_pos_error_y.append(obs_flat[:, 53].cpu().numpy())
                payload_pos_error_z.append(obs_flat[:, 54].cpu().numpy())
                payload_quat_error_w.append(obs_flat[:, 55].cpu().numpy())
                payload_quat_error_x.append(obs_flat[:, 56].cpu().numpy())
                payload_quat_error_y.append(obs_flat[:, 57].cpu().numpy())
                payload_quat_error_z.append(obs_flat[:, 58].cpu().numpy())

                if dones | timestep == args_cli.video_length:
                    break

        if args_cli.video:
            # timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

    if args_cli.plot_data:
        # Plot action data
        plt.figure(figsize=(10, 8))
        plt.subplot(2, 2, 1)
        plt.plot(drone_1_forces, label="drone 1")
        plt.plot(drone_2_forces, label="drone 2")
        plt.plot(drone_3_forces, label="drone 3")
        plt.title("Drone Forces")
        plt.legend()

        plt.subplot(2, 2, 2)
        plt.plot(drone_1_x_torque, label="drone 1 x torque")
        plt.plot(drone_1_y_torque, label="drone 1 y torque")
        plt.plot(drone_1_z_torque, label="drone 1 z torque")
        plt.title("Drone 1 Torques")
        plt.legend()

        plt.subplot(2, 2, 3)
        plt.plot(drone_2_x_torque, label="drone 2 x torque")
        plt.plot(drone_2_y_torque, label="drone 2 y torque")
        plt.plot(drone_2_z_torque, label="drone 2 z torque")
        plt.title("Drone 2 Torques")
        plt.legend()

        plt.subplot(2, 2, 4)
        plt.plot(drone_3_x_torque, label="drone 3 x torque")
        plt.plot(drone_3_y_torque, label="drone 3 y torque")
        plt.plot(drone_3_z_torque, label="drone 3 z torque")
        plt.title("Drone 3 Torques")
        plt.legend()

        # Plot clipped action data
        plt.figure(figsize=(10, 8))
        plt.subplot(2, 2, 1)
        plt.plot(np.clip(drone_1_forces, 0, 25), label="drone 1")
        plt.plot(np.clip(drone_2_forces, 0, 25), label="drone 2")
        plt.plot(np.clip(drone_3_forces, 0, 25), label="drone 3")
        plt.title("Drone Forces")
        plt.legend()

        plt.subplot(2, 2, 2)
        plt.plot(np.clip(drone_1_x_torque, -0.05, 0.05), label="drone 1 x torque")
        plt.plot(np.clip(drone_1_y_torque, -0.05, 0.05), label="drone 1 y torque")
        plt.plot(np.clip(drone_1_z_torque, -0.05, 0.05), label="drone 1 z torque")
        plt.title("Drone 1 clipped Torques")
        plt.legend()

        plt.subplot(2, 2, 3)
        plt.plot(np.clip(drone_2_x_torque, -0.05, 0.05), label="drone 2 x torque")
        plt.plot(np.clip(drone_2_y_torque, -0.05, 0.05), label="drone 2 y torque")
        plt.plot(np.clip(drone_2_z_torque, -0.05, 0.05), label="drone 2 z torque")
        plt.title("Drone 2 clipped Torques")
        plt.legend()

        plt.subplot(2, 2, 4)
        plt.plot(np.clip(drone_3_x_torque, -0.05, 0.05), label="drone 3 x torque")
        plt.plot(np.clip(drone_3_y_torque, -0.05, 0.05), label="drone 3 y torque")
        plt.plot(np.clip(drone_3_z_torque, -0.05, 0.05), label="drone 3 z torque")
        plt.title("Drone 3 clipped Torques")
        plt.legend()

        # Plot payload position
        plt.figure(figsize=(10, 8))
        plt.subplot(2, 2, 1)
        plt.plot(payload_pos_x, label="x")
        plt.plot(payload_pos_y, label="y")
        plt.plot(payload_pos_z, label="z")
        plt.title("Payload Position")
        plt.legend()

        # Plot payload orientation
        plt.subplot(2, 2, 2)
        plt.plot(payload_quat_w, label="w")
        plt.plot(payload_quat_x, label="x")
        plt.plot(payload_quat_y, label="y")
        plt.plot(payload_quat_z, label="z")
        plt.title("Payload Orientation")
        plt.legend()

        # Plot payload linear velocities
        plt.subplot(2, 2, 3)
        plt.plot(payload_lin_vel_x, label="x")
        plt.plot(payload_lin_vel_y, label="y")
        plt.plot(payload_lin_vel_z, label="z")
        plt.title("Payload Linear Velocities")
        plt.legend()

        # Plot payload angular velocities
        plt.subplot(2, 2, 4)
        plt.plot(payload_ang_vel_x, label="x")
        plt.plot(payload_ang_vel_y, label="y")
        plt.plot(payload_ang_vel_z, label="z")
        plt.title("Payload Angular Velocities")
        plt.legend()

        # Plot drone positions
        plt.figure(figsize=(10, 8))
        plt.subplot(2, 2, 1)
        plt.plot(drone_1_pos_x, label="drone 1 x")
        plt.plot(drone_1_pos_y, label="drone 1 y")
        plt.plot(drone_1_pos_z, label="drone 1 z")
        plt.title("Drone 1 Positions")
        plt.legend()

        plt.subplot(2, 2, 2)
        plt.plot(drone_2_pos_x, label="drone 2 x")
        plt.plot(drone_2_pos_y, label="drone 2 y")
        plt.plot(drone_2_pos_z, label="drone 2 z")
        plt.title("Drone 2 Positions")
        plt.legend()

        plt.subplot(2, 2, 3)
        plt.plot(drone_3_pos_x, label="drone 3 x")
        plt.plot(drone_3_pos_y, label="drone 3 y")
        plt.plot(drone_3_pos_z, label="drone 3 z")
        plt.title("Drone 3 Positions")
        plt.legend()

        # Plot drone orientations
        plt.figure(figsize=(10, 8))
        plt.subplot(2, 2, 1)
        plt.plot(drone_1_quat_w, label="drone 1 w")
        plt.plot(drone_1_quat_x, label="drone 1 x")
        plt.plot(drone_1_quat_y, label="drone 1 y")
        plt.plot(drone_1_quat_z, label="drone 1 z")
        plt.title("Drone 1 Orientations")
        plt.legend()

        plt.subplot(2, 2, 2)
        plt.plot(drone_2_quat_w, label="drone 2 w")
        plt.plot(drone_2_quat_x, label="drone 2 x")
        plt.plot(drone_2_quat_y, label="drone 2 y")
        plt.plot(drone_2_quat_z, label="drone 2 z")
        plt.title("Drone 2 Orientations")
        plt.legend()

        plt.subplot(2, 2, 3)
        plt.plot(drone_3_quat_w, label="drone 3 w")
        plt.plot(drone_3_quat_x, label="drone 3 x")
        plt.plot(drone_3_quat_y, label="drone 3 y")
        plt.plot(drone_3_quat_z, label="drone 3 z")
        plt.title("Drone 3 Orientations")
        plt.legend()

        # Plot drone linear velocities
        plt.figure(figsize=(10, 8))
        plt.subplot(2, 2, 1)
        plt.plot(drone_1_lin_vel_x, label="drone 1 x")
        plt.plot(drone_1_lin_vel_y, label="drone 1 y")
        plt.plot(drone_1_lin_vel_z, label="drone 1 z")
        plt.title("Drone 1 Linear Velocities")
        plt.legend()

        plt.subplot(2, 2, 2)
        plt.plot(drone_2_lin_vel_x, label="drone 2 x")
        plt.plot(drone_2_lin_vel_y, label="drone 2 y")
        plt.plot(drone_2_lin_vel_z, label="drone 2 z")
        plt.title("Drone 2 Linear Velocities")
        plt.legend()

        plt.subplot(2, 2, 3)
        plt.plot(drone_3_lin_vel_x, label="drone 3 x")
        plt.plot(drone_3_lin_vel_y, label="drone 3 y")
        plt.plot(drone_3_lin_vel_z, label="drone 3 z")
        plt.title("Drone 3 Linear Velocities")
        plt.legend()

        # Plot drone angular velocities
        plt.figure(figsize=(10, 8))
        plt.subplot(2, 2, 1)
        plt.plot(drone_1_ang_vel_x, label="drone 1 x")
        plt.plot(drone_1_ang_vel_y, label="drone 1 y")
        plt.plot(drone_1_ang_vel_z, label="drone 1 z")
        plt.title("Drone 1 Angular Velocities")
        plt.legend()

        plt.subplot(2, 2, 2)
        plt.plot(drone_2_ang_vel_x, label="drone 2 x")
        plt.plot(drone_2_ang_vel_y, label="drone 2 y")
        plt.plot(drone_2_ang_vel_z, label="drone 2 z")
        plt.title("Drone 2 Angular Velocities")
        plt.legend()

        plt.subplot(2, 2, 3)
        plt.plot(drone_3_ang_vel_x, label="drone 3 x")
        plt.plot(drone_3_ang_vel_y, label="drone 3 y")
        plt.plot(drone_3_ang_vel_z, label="drone 3 z")
        plt.title("Drone 3 Angular Velocities")
        plt.legend()

        plt.figure(figsize=(10, 8))
        plt.subplot(2, 2, 1)
        plt.plot(cable_angle_1_w, label="cable 1 w")
        plt.plot(cable_angle_1_x, label="cable 1 x")
        plt.plot(cable_angle_1_y, label="cable 1 y")
        plt.plot(cable_angle_1_z, label="cable 1 z")
        plt.title("Cable 1 Angles")
        plt.legend()

        plt.subplot(2, 2, 2)
        plt.plot(cable_angle_2_w, label="cable 2 w")
        plt.plot(cable_angle_2_x, label="cable 2 x")
        plt.plot(cable_angle_2_y, label="cable 2 y")
        plt.plot(cable_angle_2_z, label="cable 2 z")
        plt.title("Cable 2 Angles")
        plt.legend()

        plt.subplot(2, 2, 3)
        plt.plot(cable_angle_3_w, label="cable 3 w")
        plt.plot(cable_angle_3_x, label="cable 3 x")
        plt.plot(cable_angle_3_y, label="cable 3 y")
        plt.plot(cable_angle_3_z, label="cable 3 z")
        plt.title("Cable 3 Angles")
        plt.legend()

        plt.figure(figsize=(10, 8))
        plt.subplot(2, 2, 1)
        plt.plot(payload_pos_error_x, label="x")
        plt.plot(payload_pos_error_y, label="y")
        plt.plot(payload_pos_error_z, label="z")
        plt.title("Payload Position Error")
        plt.legend()

        plt.subplot(2, 2, 2)
        plt.plot(payload_quat_error_w, label="w")
        plt.plot(payload_quat_error_x, label="x")
        plt.plot(payload_quat_error_y, label="y")
        plt.plot(payload_quat_error_z, label="z")
        plt.title("Payload Orientation Error")
        plt.legend()

        plt.tight_layout()
        plt.show()

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main execution
    main()
    # close sim app
    simulation_app.close()
