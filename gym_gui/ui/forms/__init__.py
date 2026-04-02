"""Worker form registration utilities.

The factory itself is always available.  Form widgets register themselves
lazily --- each form module calls ``get_worker_form_factory()`` at module
level to self-register.  The ``ensure_all_forms_registered()`` helper
triggers all registrations at once and is called once by ``MainWindow``.
"""

import warnings

from .factory import WorkerFormFactory, get_worker_form_factory

# Suppress a Pydantic warning triggered by wandb's internal artifact models.
# wandb uses Field(frozen=True, repr=False) inside an Annotated union member,
# which Pydantic v2 flags as unsupported.  This is a wandb upstream bug
# (ray -> rllib -> ray.air.integrations.wandb -> wandb -> pydantic warning).
warnings.filterwarnings(
    "ignore",
    message=r"The '(repr|frozen)' attribute .* was provided to the `Field\(\)` function",
    module=r"pydantic\._internal\._generate_schema",
)


def ensure_all_forms_registered() -> None:
    """Import every worker form module so their registration side-effects run.

    Call this once at application startup (e.g. in MainWindow) rather than
    at package-import time, so that importing the ``forms`` package does
    not eagerly pull in heavy transitive dependencies (ray -> wandb, etc.).
    """
    from gym_gui.ui.widgets import (
        cleanrl_policy_form,  # noqa: F401
        ray_policy_form,  # noqa: F401
    )
    from gym_gui.ui.widgets.cleanrl_resume_form import CleanRlResumeForm  # noqa: F401
    from gym_gui.ui.widgets.cleanrl_script_form import CleanRlScriptForm  # noqa: F401
    from gym_gui.ui.widgets.cleanrl_train_form import CleanRlTrainForm  # noqa: F401
    from gym_gui.ui.widgets.marllib_train_form import MARLlibTrainForm  # noqa: F401
    from gym_gui.ui.widgets.mctx_train_form import MCTXTrainForm  # noqa: F401
    from gym_gui.ui.widgets.ray_evaluation_form import RayEvaluationForm  # noqa: F401
    from gym_gui.ui.widgets.ray_train_form import RayRLlibTrainForm  # noqa: F401
    from gym_gui.ui.widgets.xuance_script_form import XuanCeScriptForm  # noqa: F401
    from gym_gui.ui.widgets.xuance_train_form import XuanCeTrainForm  # noqa: F401
    from gym_gui.ui.widgets.tianshou_train_form import TianshouTrainForm  # noqa: F401
    from gym_gui.ui.widgets.tianshou_script_form import TianshouScriptForm  # noqa: F401
    from gym_gui.ui.widgets.tianshou_resume_form import TianshouResumeForm  # noqa: F401
    from gym_gui.ui.widgets.tianshou_policy_form import TianshouPolicyForm  # noqa: F401
    from gym_gui.ui.widgets.jumanji_train_form import JumanjiTrainForm  # noqa: F401


__all__ = [
    "WorkerFormFactory",
    "get_worker_form_factory",
    "ensure_all_forms_registered",
]
