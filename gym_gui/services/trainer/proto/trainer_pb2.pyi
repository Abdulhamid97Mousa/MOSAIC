from google.protobuf import timestamp_pb2 as _timestamp_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class RunStatus(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    RUN_STATUS_UNSPECIFIED: _ClassVar[RunStatus]
    RUN_STATUS_INIT: _ClassVar[RunStatus]
    RUN_STATUS_HSHK: _ClassVar[RunStatus]
    RUN_STATUS_RDY: _ClassVar[RunStatus]
    RUN_STATUS_EXEC: _ClassVar[RunStatus]
    RUN_STATUS_PAUSE: _ClassVar[RunStatus]
    RUN_STATUS_FAULT: _ClassVar[RunStatus]
    RUN_STATUS_TERM: _ClassVar[RunStatus]
RUN_STATUS_UNSPECIFIED: RunStatus
RUN_STATUS_INIT: RunStatus
RUN_STATUS_HSHK: RunStatus
RUN_STATUS_RDY: RunStatus
RUN_STATUS_EXEC: RunStatus
RUN_STATUS_PAUSE: RunStatus
RUN_STATUS_FAULT: RunStatus
RUN_STATUS_TERM: RunStatus

class SubmitRunRequest(_message.Message):
    __slots__ = ("run_id", "config_json")
    RUN_ID_FIELD_NUMBER: _ClassVar[int]
    CONFIG_JSON_FIELD_NUMBER: _ClassVar[int]
    run_id: str
    config_json: str
    def __init__(self, run_id: _Optional[str] = ..., config_json: _Optional[str] = ...) -> None: ...

class SubmitRunResponse(_message.Message):
    __slots__ = ("run_id", "digest")
    RUN_ID_FIELD_NUMBER: _ClassVar[int]
    DIGEST_FIELD_NUMBER: _ClassVar[int]
    run_id: str
    digest: str
    def __init__(self, run_id: _Optional[str] = ..., digest: _Optional[str] = ...) -> None: ...

class CancelRunRequest(_message.Message):
    __slots__ = ("run_id",)
    RUN_ID_FIELD_NUMBER: _ClassVar[int]
    run_id: str
    def __init__(self, run_id: _Optional[str] = ...) -> None: ...

class CancelRunResponse(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class RegisterWorkerRequest(_message.Message):
    __slots__ = ("run_id", "worker_id", "worker_kind", "proto_version", "schema_id", "schema_version", "supports_pause", "supports_checkpoint")
    RUN_ID_FIELD_NUMBER: _ClassVar[int]
    WORKER_ID_FIELD_NUMBER: _ClassVar[int]
    WORKER_KIND_FIELD_NUMBER: _ClassVar[int]
    PROTO_VERSION_FIELD_NUMBER: _ClassVar[int]
    SCHEMA_ID_FIELD_NUMBER: _ClassVar[int]
    SCHEMA_VERSION_FIELD_NUMBER: _ClassVar[int]
    SUPPORTS_PAUSE_FIELD_NUMBER: _ClassVar[int]
    SUPPORTS_CHECKPOINT_FIELD_NUMBER: _ClassVar[int]
    run_id: str
    worker_id: str
    worker_kind: str
    proto_version: str
    schema_id: str
    schema_version: int
    supports_pause: bool
    supports_checkpoint: bool
    def __init__(self, run_id: _Optional[str] = ..., worker_id: _Optional[str] = ..., worker_kind: _Optional[str] = ..., proto_version: _Optional[str] = ..., schema_id: _Optional[str] = ..., schema_version: _Optional[int] = ..., supports_pause: bool = ..., supports_checkpoint: bool = ...) -> None: ...

class RegisterWorkerResponse(_message.Message):
    __slots__ = ("accepted_version", "session_token")
    ACCEPTED_VERSION_FIELD_NUMBER: _ClassVar[int]
    SESSION_TOKEN_FIELD_NUMBER: _ClassVar[int]
    accepted_version: str
    session_token: str
    def __init__(self, accepted_version: _Optional[str] = ..., session_token: _Optional[str] = ...) -> None: ...

class ControlEvent(_message.Message):
    __slots__ = ("run_id", "grant_steps", "pause", "resume", "max_rate_hz")
    RUN_ID_FIELD_NUMBER: _ClassVar[int]
    GRANT_STEPS_FIELD_NUMBER: _ClassVar[int]
    PAUSE_FIELD_NUMBER: _ClassVar[int]
    RESUME_FIELD_NUMBER: _ClassVar[int]
    MAX_RATE_HZ_FIELD_NUMBER: _ClassVar[int]
    run_id: str
    grant_steps: int
    pause: bool
    resume: bool
    max_rate_hz: int
    def __init__(self, run_id: _Optional[str] = ..., grant_steps: _Optional[int] = ..., pause: bool = ..., resume: bool = ..., max_rate_hz: _Optional[int] = ...) -> None: ...

class RunRecord(_message.Message):
    __slots__ = ("run_id", "status", "digest", "created_at", "updated_at", "last_heartbeat", "gpu_slot", "failure_reason", "gpu_slots", "seq_id")
    RUN_ID_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    DIGEST_FIELD_NUMBER: _ClassVar[int]
    CREATED_AT_FIELD_NUMBER: _ClassVar[int]
    UPDATED_AT_FIELD_NUMBER: _ClassVar[int]
    LAST_HEARTBEAT_FIELD_NUMBER: _ClassVar[int]
    GPU_SLOT_FIELD_NUMBER: _ClassVar[int]
    FAILURE_REASON_FIELD_NUMBER: _ClassVar[int]
    GPU_SLOTS_FIELD_NUMBER: _ClassVar[int]
    SEQ_ID_FIELD_NUMBER: _ClassVar[int]
    run_id: str
    status: RunStatus
    digest: str
    created_at: _timestamp_pb2.Timestamp
    updated_at: _timestamp_pb2.Timestamp
    last_heartbeat: _timestamp_pb2.Timestamp
    gpu_slot: int
    failure_reason: str
    gpu_slots: _containers.RepeatedScalarFieldContainer[int]
    seq_id: int
    def __init__(self, run_id: _Optional[str] = ..., status: _Optional[_Union[RunStatus, str]] = ..., digest: _Optional[str] = ..., created_at: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ..., updated_at: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ..., last_heartbeat: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ..., gpu_slot: _Optional[int] = ..., failure_reason: _Optional[str] = ..., gpu_slots: _Optional[_Iterable[int]] = ..., seq_id: _Optional[int] = ...) -> None: ...

class ListRunsRequest(_message.Message):
    __slots__ = ("status_filter",)
    STATUS_FILTER_FIELD_NUMBER: _ClassVar[int]
    status_filter: _containers.RepeatedScalarFieldContainer[RunStatus]
    def __init__(self, status_filter: _Optional[_Iterable[_Union[RunStatus, str]]] = ...) -> None: ...

class ListRunsResponse(_message.Message):
    __slots__ = ("runs",)
    RUNS_FIELD_NUMBER: _ClassVar[int]
    runs: _containers.RepeatedCompositeFieldContainer[RunRecord]
    def __init__(self, runs: _Optional[_Iterable[_Union[RunRecord, _Mapping]]] = ...) -> None: ...

class WatchRunsRequest(_message.Message):
    __slots__ = ("status_filter", "since_seq")
    STATUS_FILTER_FIELD_NUMBER: _ClassVar[int]
    SINCE_SEQ_FIELD_NUMBER: _ClassVar[int]
    status_filter: _containers.RepeatedScalarFieldContainer[RunStatus]
    since_seq: int
    def __init__(self, status_filter: _Optional[_Iterable[_Union[RunStatus, str]]] = ..., since_seq: _Optional[int] = ...) -> None: ...

class HeartbeatRequest(_message.Message):
    __slots__ = ("run_id",)
    RUN_ID_FIELD_NUMBER: _ClassVar[int]
    run_id: str
    def __init__(self, run_id: _Optional[str] = ...) -> None: ...

class HeartbeatResponse(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class HealthCheckRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class HealthCheckResponse(_message.Message):
    __slots__ = ("pid", "started_at", "listen_address", "healthy")
    PID_FIELD_NUMBER: _ClassVar[int]
    STARTED_AT_FIELD_NUMBER: _ClassVar[int]
    LISTEN_ADDRESS_FIELD_NUMBER: _ClassVar[int]
    HEALTHY_FIELD_NUMBER: _ClassVar[int]
    pid: int
    started_at: _timestamp_pb2.Timestamp
    listen_address: str
    healthy: bool
    def __init__(self, pid: _Optional[int] = ..., started_at: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ..., listen_address: _Optional[str] = ..., healthy: bool = ...) -> None: ...

class StreamStepsRequest(_message.Message):
    __slots__ = ("run_id", "since_seq")
    RUN_ID_FIELD_NUMBER: _ClassVar[int]
    SINCE_SEQ_FIELD_NUMBER: _ClassVar[int]
    run_id: str
    since_seq: int
    def __init__(self, run_id: _Optional[str] = ..., since_seq: _Optional[int] = ...) -> None: ...

class StreamEpisodesRequest(_message.Message):
    __slots__ = ("run_id", "since_seq")
    RUN_ID_FIELD_NUMBER: _ClassVar[int]
    SINCE_SEQ_FIELD_NUMBER: _ClassVar[int]
    run_id: str
    since_seq: int
    def __init__(self, run_id: _Optional[str] = ..., since_seq: _Optional[int] = ...) -> None: ...

class RunStep(_message.Message):
    __slots__ = ("run_id", "episode_index", "step_index", "action_json", "observation_json", "reward", "terminated", "truncated", "timestamp", "policy_label", "backend", "seq_id", "agent_id", "render_hint_json", "frame_ref", "payload_version", "render_payload_json", "episode_seed", "worker_id")
    RUN_ID_FIELD_NUMBER: _ClassVar[int]
    EPISODE_INDEX_FIELD_NUMBER: _ClassVar[int]
    STEP_INDEX_FIELD_NUMBER: _ClassVar[int]
    ACTION_JSON_FIELD_NUMBER: _ClassVar[int]
    OBSERVATION_JSON_FIELD_NUMBER: _ClassVar[int]
    REWARD_FIELD_NUMBER: _ClassVar[int]
    TERMINATED_FIELD_NUMBER: _ClassVar[int]
    TRUNCATED_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    POLICY_LABEL_FIELD_NUMBER: _ClassVar[int]
    BACKEND_FIELD_NUMBER: _ClassVar[int]
    SEQ_ID_FIELD_NUMBER: _ClassVar[int]
    AGENT_ID_FIELD_NUMBER: _ClassVar[int]
    RENDER_HINT_JSON_FIELD_NUMBER: _ClassVar[int]
    FRAME_REF_FIELD_NUMBER: _ClassVar[int]
    PAYLOAD_VERSION_FIELD_NUMBER: _ClassVar[int]
    RENDER_PAYLOAD_JSON_FIELD_NUMBER: _ClassVar[int]
    EPISODE_SEED_FIELD_NUMBER: _ClassVar[int]
    WORKER_ID_FIELD_NUMBER: _ClassVar[int]
    run_id: str
    episode_index: int
    step_index: int
    action_json: str
    observation_json: str
    reward: float
    terminated: bool
    truncated: bool
    timestamp: _timestamp_pb2.Timestamp
    policy_label: str
    backend: str
    seq_id: int
    agent_id: str
    render_hint_json: str
    frame_ref: str
    payload_version: int
    render_payload_json: str
    episode_seed: int
    worker_id: str
    def __init__(self, run_id: _Optional[str] = ..., episode_index: _Optional[int] = ..., step_index: _Optional[int] = ..., action_json: _Optional[str] = ..., observation_json: _Optional[str] = ..., reward: _Optional[float] = ..., terminated: bool = ..., truncated: bool = ..., timestamp: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ..., policy_label: _Optional[str] = ..., backend: _Optional[str] = ..., seq_id: _Optional[int] = ..., agent_id: _Optional[str] = ..., render_hint_json: _Optional[str] = ..., frame_ref: _Optional[str] = ..., payload_version: _Optional[int] = ..., render_payload_json: _Optional[str] = ..., episode_seed: _Optional[int] = ..., worker_id: _Optional[str] = ...) -> None: ...

class RunEpisode(_message.Message):
    __slots__ = ("run_id", "episode_index", "total_reward", "steps", "terminated", "truncated", "metadata_json", "timestamp", "seq_id", "agent_id", "worker_id")
    RUN_ID_FIELD_NUMBER: _ClassVar[int]
    EPISODE_INDEX_FIELD_NUMBER: _ClassVar[int]
    TOTAL_REWARD_FIELD_NUMBER: _ClassVar[int]
    STEPS_FIELD_NUMBER: _ClassVar[int]
    TERMINATED_FIELD_NUMBER: _ClassVar[int]
    TRUNCATED_FIELD_NUMBER: _ClassVar[int]
    METADATA_JSON_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    SEQ_ID_FIELD_NUMBER: _ClassVar[int]
    AGENT_ID_FIELD_NUMBER: _ClassVar[int]
    WORKER_ID_FIELD_NUMBER: _ClassVar[int]
    run_id: str
    episode_index: int
    total_reward: float
    steps: int
    terminated: bool
    truncated: bool
    metadata_json: str
    timestamp: _timestamp_pb2.Timestamp
    seq_id: int
    agent_id: str
    worker_id: str
    def __init__(self, run_id: _Optional[str] = ..., episode_index: _Optional[int] = ..., total_reward: _Optional[float] = ..., steps: _Optional[int] = ..., terminated: bool = ..., truncated: bool = ..., metadata_json: _Optional[str] = ..., timestamp: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ..., seq_id: _Optional[int] = ..., agent_id: _Optional[str] = ..., worker_id: _Optional[str] = ...) -> None: ...

class PublishTelemetryResponse(_message.Message):
    __slots__ = ("accepted", "dropped")
    ACCEPTED_FIELD_NUMBER: _ClassVar[int]
    DROPPED_FIELD_NUMBER: _ClassVar[int]
    accepted: int
    dropped: int
    def __init__(self, accepted: _Optional[int] = ..., dropped: _Optional[int] = ...) -> None: ...

class VideoFrame(_message.Message):
    __slots__ = ("run_id", "width", "height", "rgb_data", "frame_number", "timestamp_ms", "agent_x", "agent_y", "agent_z", "agent_yaw", "agent_pitch", "life", "food", "xp", "reward", "is_alive")
    RUN_ID_FIELD_NUMBER: _ClassVar[int]
    WIDTH_FIELD_NUMBER: _ClassVar[int]
    HEIGHT_FIELD_NUMBER: _ClassVar[int]
    RGB_DATA_FIELD_NUMBER: _ClassVar[int]
    FRAME_NUMBER_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_MS_FIELD_NUMBER: _ClassVar[int]
    AGENT_X_FIELD_NUMBER: _ClassVar[int]
    AGENT_Y_FIELD_NUMBER: _ClassVar[int]
    AGENT_Z_FIELD_NUMBER: _ClassVar[int]
    AGENT_YAW_FIELD_NUMBER: _ClassVar[int]
    AGENT_PITCH_FIELD_NUMBER: _ClassVar[int]
    LIFE_FIELD_NUMBER: _ClassVar[int]
    FOOD_FIELD_NUMBER: _ClassVar[int]
    XP_FIELD_NUMBER: _ClassVar[int]
    REWARD_FIELD_NUMBER: _ClassVar[int]
    IS_ALIVE_FIELD_NUMBER: _ClassVar[int]
    run_id: str
    width: int
    height: int
    rgb_data: bytes
    frame_number: int
    timestamp_ms: int
    agent_x: float
    agent_y: float
    agent_z: float
    agent_yaw: float
    agent_pitch: float
    life: float
    food: float
    xp: int
    reward: float
    is_alive: bool
    def __init__(self, run_id: _Optional[str] = ..., width: _Optional[int] = ..., height: _Optional[int] = ..., rgb_data: _Optional[bytes] = ..., frame_number: _Optional[int] = ..., timestamp_ms: _Optional[int] = ..., agent_x: _Optional[float] = ..., agent_y: _Optional[float] = ..., agent_z: _Optional[float] = ..., agent_yaw: _Optional[float] = ..., agent_pitch: _Optional[float] = ..., life: _Optional[float] = ..., food: _Optional[float] = ..., xp: _Optional[int] = ..., reward: _Optional[float] = ..., is_alive: bool = ...) -> None: ...

class VideoFrameAck(_message.Message):
    __slots__ = ("received", "frame_number")
    RECEIVED_FIELD_NUMBER: _ClassVar[int]
    FRAME_NUMBER_FIELD_NUMBER: _ClassVar[int]
    received: bool
    frame_number: int
    def __init__(self, received: bool = ..., frame_number: _Optional[int] = ...) -> None: ...

class VideoFrameControl(_message.Message):
    __slots__ = ("pause", "resume", "target_fps", "close")
    PAUSE_FIELD_NUMBER: _ClassVar[int]
    RESUME_FIELD_NUMBER: _ClassVar[int]
    TARGET_FPS_FIELD_NUMBER: _ClassVar[int]
    CLOSE_FIELD_NUMBER: _ClassVar[int]
    pause: bool
    resume: bool
    target_fps: int
    close: bool
    def __init__(self, pause: bool = ..., resume: bool = ..., target_fps: _Optional[int] = ..., close: bool = ...) -> None: ...
