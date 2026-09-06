"""Headless smoke tests for the DirectMARL tasks.

These are the regression net for the Isaac Lab 3.0 port: they construct each registered
DirectMARL environment, reset it, step it, and assert the returned tensors match the spaces the
config declares. Run with::

    OMNI_KIT_ACCEPT_EULA=YES pytest -q tests/test_directmarl_envs.py

A single Isaac Sim app is shared by every test (Kit cannot be started twice in one process).
"""

from __future__ import annotations

import pytest

TASKS = [
    ("Isaac-flycrane-payload-decentralized-hovering-v0", 3),
    ("Isaac-flycrane-payload-decentralized-hovering-flycart-v0", 4),
    ("Isaac-flycrane-payload-decentralized-hovering-flypent-v0", 5),
]


@pytest.fixture(scope="session")
def sim_app():
    from isaaclab.app import AppLauncher

    # NOTE: deliberately not calling ``app.close()``. Kit tears the process down from inside
    # ``close()``, which kills pytest before it can write its report (the run then looks like a
    # silent pass). The app goes away with the interpreter instead.
    return AppLauncher(headless=True).app


@pytest.fixture(scope="session")
def gym(sim_app):
    import gymnasium

    import MARL_mav_carry_ext.tasks  # noqa: F401  (registers the tasks)

    return gymnasium


@pytest.mark.parametrize("task_id,num_agents", TASKS)
def test_env_resets_and_steps(gym, task_id, num_agents):
    import torch
    from isaaclab_tasks.utils import parse_env_cfg

    env_cfg = parse_env_cfg(task_id, device="cuda:0", num_envs=2)
    env = gym.make(task_id, cfg=env_cfg)
    try:
        unwrapped = env.unwrapped
        assert len(unwrapped.possible_agents) == num_agents

        obs, _ = env.reset()
        assert set(obs) == set(unwrapped.possible_agents)
        for uid in unwrapped.possible_agents:
            expected = unwrapped.observation_spaces[uid].shape[0]
            assert obs[uid].shape == (2, expected), f"{uid}: {obs[uid].shape} != (2, {expected})"
            assert torch.isfinite(obs[uid]).all(), f"{uid} observation has non-finite values after reset"

        for _ in range(5):
            actions = {
                uid: torch.zeros((2, unwrapped.action_spaces[uid].shape[0]), device=unwrapped.device)
                for uid in unwrapped.possible_agents
            }
            obs, rew, terminated, truncated, _ = env.step(actions)
            for uid in unwrapped.possible_agents:
                assert torch.isfinite(obs[uid]).all(), f"{uid} observation went non-finite while stepping"
                assert torch.isfinite(rew[uid]).all(), f"{uid} reward went non-finite while stepping"

        state = unwrapped.state()
        assert state.shape == (2, unwrapped.state_space.shape[0])
    finally:
        env.close()


MANAGER_BASED_TASKS = [
    "Isaac-flycrane-payload-hovering-v0",
    "Isaac-flycrane-payload-hovering-llc-v0",
    "Isaac-flycrane-payload-track-v0",
    "Isaac-flycrane-payload-obstacle-avoidance-v0",
]


@pytest.mark.parametrize("task_id", MANAGER_BASED_TASKS)
def test_manager_based_env_resets_and_steps(gym, task_id):
    """Step the manager-based tasks too.

    These exercise the reward/observation/termination term functions, which the DirectMARL tasks
    do not share. A missing import inside one of those terms only raises when the term is
    actually called, so importing the task is not enough.
    """
    import torch
    from isaaclab_tasks.utils import parse_env_cfg
    from dataclasses import MISSING

    env_cfg = parse_env_cfg(task_id, device="cuda:0", num_envs=2)
    # `control_mode` is deliberately MISSING on the low-level action term; the play script supplies
    # it from --control_mode, so the test has to pick one too.
    low_level = getattr(env_cfg.actions, "low_level_action", None)
    if low_level is not None and isinstance(getattr(low_level, "control_mode", None), type(MISSING)):
        low_level.control_mode = "ACCBR"

    env = gym.make(task_id, cfg=env_cfg)
    try:
        obs, _ = env.reset()
        assert torch.isfinite(obs["policy"]).all(), "observation is non-finite after reset"

        for _ in range(5):
            actions = torch.zeros((2, env.unwrapped.action_manager.total_action_dim), device=env.unwrapped.device)
            obs, rew, terminated, truncated, _ = env.step(actions)
            assert torch.isfinite(obs["policy"]).all(), "observation went non-finite while stepping"
            assert torch.isfinite(rew).all(), "reward went non-finite while stepping"
    finally:
        env.close()
