from isaaclab.utils.configclass import configclass  # explicit: isaaclab.utils lazy-exports this name and it can be shadowed by the submodule
from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


@configclass
class FlycraneHoverPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 150000
    save_interval = 50
    experiment_name = "Flycrane_hover"
    # rsl-rl 5.x: map each network onto the env's observation groups. Both
    # manager-based envs define a single "policy" group.
    obs_groups = {"actor": ["policy"], "critic": ["policy"]}
    # logger = "wandb"
    # resume = True
    wandb_project = "Flycrane_hover"
    # rsl-rl 5.x builds the two networks from separate model configs; the old single
    # RslRlPpoActorCriticCfg is still exported but the runner reads cfg["actor"]/cfg["critic"].
    actor = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
