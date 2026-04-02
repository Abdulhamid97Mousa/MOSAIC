"""Algorithm parameter handling for Ray RLlib.

This module provides schema-based parameter validation and application
for different RL algorithms. Each algorithm (PPO, APPO, IMPALA, DQN, SAC)
has its own set of valid parameters defined in metadata/ray_rllib/schemas.json.

Key Design:
- Parameters are validated against schemas before training
- Only valid parameters for the selected algorithm are applied
- Prevents runtime crashes from invalid parameter combinations

See Also:
    - metadata/ray_rllib/0.1.0/schemas.json for parameter definitions
    - gym_gui/ui/widgets/ray_train_form.py for UI integration
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Schema version
SCHEMA_VERSION = "0.1.0"

# Path to schemas relative to repo root
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
SCHEMAS_PATH = _REPO_ROOT / "metadata" / "ray_rllib" / SCHEMA_VERSION / "schemas.json"

# Cached schema data
_schema_cache: Optional[Dict[str, Any]] = None


def _load_schemas() -> Dict[str, Any]:
    """Load algorithm schemas from JSON file.

    Returns:
        Parsed schema dictionary

    Raises:
        FileNotFoundError: If schemas.json doesn't exist
        json.JSONDecodeError: If JSON is invalid
    """
    global _schema_cache

    if _schema_cache is not None:
        return _schema_cache

    if not SCHEMAS_PATH.exists():
        raise FileNotFoundError(
            f"Algorithm schemas not found at {SCHEMAS_PATH}. "
            "Please ensure metadata/ray_rllib/0.1.0/schemas.json exists."
        )

    with open(SCHEMAS_PATH) as f:
        _schema_cache = json.load(f)

    logger.debug(f"Loaded algorithm schemas from {SCHEMAS_PATH}")
    assert _schema_cache is not None  # json.load always returns a value
    return _schema_cache


def get_available_algorithms() -> List[str]:
    """Get list of available algorithm names.

    Returns:
        List of algorithm names (e.g., ["PPO", "APPO", "IMPALA", "DQN", "SAC"])
    """
    schemas = _load_schemas()
    return list(schemas.get("algorithms", {}).keys())


def get_algorithm_info(algorithm: str) -> Dict[str, Any]:
    """Get algorithm display information.

    Args:
        algorithm: Algorithm name (e.g., "PPO")

    Returns:
        Dict with display_name, description, action_space

    Raises:
        ValueError: If algorithm not found
    """
    schemas = _load_schemas()
    algorithms = schemas.get("algorithms", {})

    if algorithm not in algorithms:
        available = ", ".join(algorithms.keys())
        raise ValueError(f"Unknown algorithm '{algorithm}'. Available: {available}")

    algo_data = algorithms[algorithm]
    return {
        "name": algorithm,
        "display_name": algo_data.get("display_name", algorithm),
        "description": algo_data.get("description", ""),
        "action_space": algo_data.get("action_space", ["discrete", "continuous"]),
    }


def get_algorithm_fields(algorithm: str) -> List[Dict[str, Any]]:
    """Get parameter field definitions for an algorithm.

    Args:
        algorithm: Algorithm name (e.g., "PPO")

    Returns:
        List of field definitions with name, type, default, help, etc.

    Raises:
        ValueError: If algorithm not found
    """
    schemas = _load_schemas()
    algorithms = schemas.get("algorithms", {})

    if algorithm not in algorithms:
        available = ", ".join(algorithms.keys())
        raise ValueError(f"Unknown algorithm '{algorithm}'. Available: {available}")

    return algorithms[algorithm].get("fields", [])


def get_common_fields() -> List[Dict[str, Any]]:
    """Get common parameter fields available to all algorithms.

    Returns:
        List of common field definitions (total_timesteps, num_workers, etc.)
    """
    schemas = _load_schemas()
    return schemas.get("common_fields", {}).get("fields", [])


def get_all_fields(algorithm: str) -> List[Dict[str, Any]]:
    """Get all fields (algorithm-specific + common) for an algorithm.

    Args:
        algorithm: Algorithm name

    Returns:
        Combined list of all parameter fields
    """
    return get_algorithm_fields(algorithm) + get_common_fields()


def get_field_names(algorithm: str) -> List[str]:
    """Get list of valid parameter names for an algorithm.

    Args:
        algorithm: Algorithm name

    Returns:
        List of valid parameter names
    """
    return [f["name"] for f in get_all_fields(algorithm)]


def get_default_params(algorithm: str) -> Dict[str, Any]:
    """Get default parameter values for an algorithm.

    Args:
        algorithm: Algorithm name

    Returns:
        Dict mapping parameter names to default values
    """
    defaults = {}
    for field in get_all_fields(algorithm):
        defaults[field["name"]] = field.get("default")
    return defaults


def validate_params(algorithm: str, params: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Validate parameters against algorithm schema.

    Args:
        algorithm: Algorithm name
        params: Parameter dictionary to validate

    Returns:
        Tuple of (is_valid, list of error messages)
    """
    errors = []
    valid_names = set(get_field_names(algorithm))
    fields_by_name = {f["name"]: f for f in get_all_fields(algorithm)}

    # Check for unknown parameters
    for name in params:
        if name not in valid_names:
            errors.append(f"Unknown parameter '{name}' for algorithm {algorithm}")

    # Validate each parameter
    for name, value in params.items():
        if name not in fields_by_name:
            continue

        field = fields_by_name[name]
        field_type = field.get("type")

        # Type validation
        if field_type == "int" and not isinstance(value, int):
            errors.append(f"Parameter '{name}' must be int, got {type(value).__name__}")
        elif field_type == "float" and not isinstance(value, (int, float)):
            errors.append(f"Parameter '{name}' must be float, got {type(value).__name__}")
        elif field_type == "bool" and not isinstance(value, bool):
            errors.append(f"Parameter '{name}' must be bool, got {type(value).__name__}")

        # Range validation for numeric types
        if field_type in ("int", "float") and isinstance(value, (int, float)):
            min_val = field.get("min")
            max_val = field.get("max")
            if min_val is not None and value < min_val:
                errors.append(f"Parameter '{name}' must be >= {min_val}, got {value}")
            if max_val is not None and value > max_val:
                errors.append(f"Parameter '{name}' must be <= {max_val}, got {value}")

    return len(errors) == 0, errors


def filter_params_for_algorithm(algorithm: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Filter parameters to only include valid ones for an algorithm.

    This is useful when switching algorithms - removes invalid parameters
    and keeps only those defined in the schema.

    Args:
        algorithm: Target algorithm name
        params: Parameter dictionary (may contain extra parameters)

    Returns:
        Filtered dictionary with only valid parameters
    """
    valid_names = set(get_field_names(algorithm))
    filtered = {k: v for k, v in params.items() if k in valid_names}

    removed = set(params.keys()) - valid_names
    if removed:
        logger.debug(f"Filtered out invalid params for {algorithm}: {removed}")

    return filtered


def merge_with_defaults(algorithm: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Merge provided parameters with defaults for missing values.

    Args:
        algorithm: Algorithm name
        params: User-provided parameters

    Returns:
        Complete parameter dictionary with defaults filled in
    """
    defaults = get_default_params(algorithm)
    merged = defaults.copy()

    # Filter and apply user params
    valid_params = filter_params_for_algorithm(algorithm, params)
    merged.update(valid_params)

    return merged


def build_training_params(algorithm: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Build the training config parameters for Ray RLlib.

    This returns only the algorithm-specific training parameters
    (excludes common fields like num_workers, total_timesteps).

    Args:
        algorithm: Algorithm name
        params: Full parameter dictionary

    Returns:
        Dict of training-specific parameters for AlgorithmConfig.training()
    """
    algo_fields = get_algorithm_fields(algorithm)
    algo_field_names = {f["name"] for f in algo_fields}

    training_params = {}
    for name, value in params.items():
        if name in algo_field_names:
            training_params[name] = value

    return training_params


def build_resource_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """Extract resource parameters from params dict.

    Args:
        params: Full parameter dictionary

    Returns:
        Dict with num_workers, num_gpus, etc.
    """
    resource_keys = {"num_workers", "num_gpus"}
    return {k: v for k, v in params.items() if k in resource_keys}


# Ray RLlib parameter name mapping (schema name -> RLlib config name)
# Some parameters have different names in RLlib's config
RLLIB_PARAM_MAPPING = {
    "train_batch_size": "train_batch_size_per_learner",  # Ray 2.x naming
    "sgd_minibatch_size": "minibatch_size",  # PPO specific
    "num_sgd_iter": "num_epochs",  # PPO specific
}


def map_to_rllib_names(params: Dict[str, Any]) -> Dict[str, Any]:
    """Map parameter names from schema to RLlib config names.

    Args:
        params: Parameters with schema names

    Returns:
        Parameters with RLlib config names
    """
    mapped = {}
    for name, value in params.items():
        rllib_name = RLLIB_PARAM_MAPPING.get(name, name)
        mapped[rllib_name] = value
    return mapped


__all__ = [
    "SCHEMA_VERSION",
    "SCHEMAS_PATH",
    "get_available_algorithms",
    "get_algorithm_info",
    "get_algorithm_fields",
    "get_common_fields",
    "get_all_fields",
    "get_field_names",
    "get_default_params",
    "validate_params",
    "filter_params_for_algorithm",
    "merge_with_defaults",
    "build_training_params",
    "build_resource_params",
    "map_to_rllib_names",
    "RLLIB_PARAM_MAPPING",
]
