from hydra import compose, initialize_config_module
from jaxmarl_worker.algorithms.mosaic_multigrid import structured_configs  # noqa: F401 (registers configs)


def test_mappo_config_composes_with_bb_defaults():
    with initialize_config_module(
        config_module="jaxmarl_worker.algorithms.mosaic_multigrid.conf",
        version_base=None,
    ):
        cfg = compose(config_name="mappo_config", overrides=["run_dir=/tmp/test"])
        assert cfg.env.family == "bb"
        assert cfg.env.variant == "G-2v0"
        assert cfg.ppo.n_envs == 256
        assert cfg.ppo.hidden_dim == 256
        assert cfg.agent_id.enabled is False
        assert cfg.training_reward == "cooperative-no-opponent"


def test_mappo_config_agent_id_override():
    with initialize_config_module(
        config_module="jaxmarl_worker.algorithms.mosaic_multigrid.conf",
        version_base=None,
    ):
        cfg = compose(
            config_name="mappo_config",
            overrides=["run_dir=/tmp/test", "agent_id=on", "env=af"],
        )
        assert cfg.agent_id.enabled is True
        assert cfg.env.family == "af"


def test_vdppo_config_has_mixer_fields():
    with initialize_config_module(
        config_module="jaxmarl_worker.algorithms.mosaic_multigrid.conf",
        version_base=None,
    ):
        cfg = compose(config_name="vdppo_config", overrides=["run_dir=/tmp/test"])
        assert cfg.ppo.mixer == "QMIX"
        assert cfg.ppo.mix_hidden == 32
