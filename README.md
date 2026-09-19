<div align="center">
    <a href="https://github.com/Abdulhamid97Mousa/mosaic"><img width="1000px" height="auto" src="docs/source/_static/figures/A_Full_Architecture.png"></a>
</div>

---

[![arXiv](https://img.shields.io/badge/arXiv-2603.01260-b31b1b.svg)](https://arxiv.org/abs/2603.01260)
[![CI](https://github.com/Abdulhamid97Mousa/mosaic/actions/workflows/ci.yml/badge.svg)](https://github.com/Abdulhamid97Mousa/mosaic/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/Abdulhamid97Mousa/mosaic/branch/main/graph/badge.svg)](https://codecov.io/gh/Abdulhamid97Mousa/mosaic)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%3E%3D2.0.0-red)](https://pytorch.org/get-started/locally/)
[![Gymnasium](https://img.shields.io/badge/gymnasium-%3E%3D0.28.1-blue)](https://gymnasium.farama.org/)
[![PettingZoo](https://img.shields.io/badge/PettingZoo-%3E%3D1.24.0-blue)](https://pettingzoo.farama.org/)
[![Platform](https://img.shields.io/badge/Platform-Linux%20%7C%20Windows-blue)](https://github.com/Abdulhamid97Mousa/mosaic)
[![GitHub license](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/Abdulhamid97Mousa/mosaic/blob/main/LICENSE)
[![Documentation](https://img.shields.io/badge/docs-ReadTheDocs-blue.svg)](https://mosaic-agent-level-interface.readthedocs.io/en/latest/)
[![Contributors](https://img.shields.io/github/contributors/Abdulhamid97Mousa/mosaic)](https://github.com/Abdulhamid97Mousa/mosaic/graphs/contributors)
[![Issues](https://img.shields.io/github/issues/Abdulhamid97Mousa/mosaic)](https://github.com/Abdulhamid97Mousa/mosaic/issues)
[![GitHub stars](https://img.shields.io/github/stars/Abdulhamid97Mousa/mosaic?style=social)](https://github.com/Abdulhamid97Mousa/mosaic/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/Abdulhamid97Mousa/mosaic?style=social)](https://github.com/Abdulhamid97Mousa/mosaic/network/members)






# MOSAIC

**A Universal Agent-Level Interface for Cross-Paradigm Agent Mixing and Human-AI Collaboration**

MOSAIC is a [visual-first platform](https://mosaic-platform.readthedocs.io/en/latest/) for cross-paradigm multi-agent systems, where RL policies, LLMs, VLMs, and human operators act in the same environment. It enables cross-paradigm agent mixing: each agent slot can be assigned to any decision-maker. This is made possible by an [operator](https://mosaic-platform.readthedocs.io/en/latest/documents/architecture/operators/index.html) abstraction that forms a universal agent-level interface by mapping [workers](https://mosaic-platform.readthedocs.io/en/latest/documents/architecture/workers/index.html) to agent slots: each operator, whether backed by an RL policy, an LLM, a VLM, or a human, conforms to the same minimal interface. Like tiles in a mosaic, different paradigms come together to form a complete picture of agent performance.


| **Documentation**: [mosaic-platform.readthedocs.io](https://mosaic-platform.readthedocs.io/en/latest/) | **GitHub**: [github.com/Abdulhamid97Mousa/mosaic](https://github.com/Abdulhamid97Mousa/mosaic) |
| ------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------- |


## Video Showcase

<table>
  <tr>
    <td width="50%">
      <video src="https://github.com/user-attachments/assets/df505885-525f-4f21-bacc-96f5464cf845" controls autoplay muted loop style="width:100%;border-radius:8px;box-shadow:0 2px 8px rgba(0,0,0,0.15);"></video>
      <p align="center"><strong>Google Research Football inside MOSAIC</strong>: human-controlled player, 10 Hz real-time stepping, RGB render view.</p>
    </td>
    <td width="50%">
      <video src="https://github.com/user-attachments/assets/118bbc56-2ca0-4c7b-8c0c-c24464dbad43" controls autoplay muted loop style="width:100%;border-radius:8px;box-shadow:0 2px 8px rgba(0,0,0,0.15);"></video>
      <p align="center"><strong>MAPPO training on GRF</strong> (<code>academy_3_vs_1_with_keeper</code>) via <code>xuance_worker</code> inside MOSAIC.</p>
    </td>
  </tr>
  <tr>
    <td width="50%">
      <video src="https://github.com/user-attachments/assets/6c1bde96-c0ab-401c-ba80-56cdf3ab6807" controls autoplay muted loop style="width:100%;border-radius:8px;box-shadow:0 2px 8px rgba(0,0,0,0.15);"></video>
      <p align="center"><strong>StarCraft Multi-Agent Challenge (SMAC) inside MOSAIC</strong>: cooperative micromanagement via <code>xuance_worker</code>.</p>
    </td>
    <td width="50%">
      <video src="https://github.com/user-attachments/assets/1f1b362d-0046-4393-ba23-675269239b3d" controls autoplay muted loop style="width:100%;border-radius:8px;box-shadow:0 2px 8px rgba(0,0,0,0.15);"></video>
      <p align="center"><strong>MAPPO training on SMAC</strong> via <code>xuance_worker</code> inside MOSAIC.</p>
    </td>
  </tr>
  <tr>
    <td width="50%">
      <video src="https://github.com/user-attachments/assets/dc83a7d3-8088-49f5-a1ad-05acece24b50" controls autoplay muted loop style="width:100%;border-radius:8px;box-shadow:0 2px 8px rgba(0,0,0,0.15);"></video>
      <p align="center"><strong>SMACv2 inside MOSAIC</strong>: procedural unit spawns and randomised team compositions via <code>xuance_worker</code>.</p>
    </td>
    <td width="50%">
      <video src="https://github.com/user-attachments/assets/a752f29d-6b40-4a36-adea-21f3bc09069c" controls autoplay muted loop style="width:100%;border-radius:8px;box-shadow:0 2px 8px rgba(0,0,0,0.15);"></video>
      <p align="center"><strong>MAPPO training on SMACv2</strong> via <code>xuance_worker</code> inside MOSAIC.</p>
    </td>
  </tr>
  <tr>
    <td width="50%">
      <video src="https://github.com/user-attachments/assets/4644af3c-7e87-4ddc-a43b-df0ece99df9f" controls autoplay muted loop style="width:100%;border-radius:8px;box-shadow:0 2px 8px rgba(0,0,0,0.15);"></video>
      <p align="center"><strong>ViZDoom inside MOSAIC</strong>: human-controlled player via keyboard, first-person 3D pixel observations.</p>
    </td>
    <td width="50%">
      <video src="https://github.com/user-attachments/assets/586aed3c-45fd-446e-8eaf-c2fd472bca58" controls autoplay muted loop style="width:100%;border-radius:8px;box-shadow:0 2px 8px rgba(0,0,0,0.15);"></video>
      <p align="center"><strong>PPO_Clip training on ViZDoom-TakeCover-v0</strong> via <code>xuance_worker</code> inside MOSAIC.</p>
    </td>
  </tr>
</table>

## Why MOSAIC?

Today's AI landscape offers powerful but **fragmented** tools: RL frameworks ([CleanRL](https://github.com/vwxyzjn/cleanrl), [RLlib](https://docs.ray.io/en/latest/rllib/index.html), [XuanCe](https://github.com/agi-brain/xuance)), language models (GPT, Claude), and robotics simulators (MuJoCo). Each excels in isolation, but **no platform bridges them together** under a unified, visual-first interface.

**MOSAIC provides:**

- **Visual-First Design**: Configure experiments through an intuitive PyQt6 interface, **almost no code required**.
- **Cross-Paradigm Agent Mixing**: To study cross-paradigm multi-agent systems, MOSAIC enables different types of decision-makers (RL policies, LLMs, VLMs, and humans) to interact with one another inside one environment.
- **Resource Management & Quotas**: GPU allocation, queue limits, credit-based backpressure, health monitoring.
- **Per-Agent Policy Binding**: Route each agent slot to its own worker, or share one policy across slots via link groups ([`PolicyMappingService`](https://mosaic-platform.readthedocs.io/en/latest/documents/architecture/policy_mapping.html)).
- **Worker Lifecycle Orchestration**: Subprocess management with heartbeat monitoring and graceful termination.

**1. Human vs Human:** Two human players competing via dedicated USB keyboards.

<video src="https://private-user-images.githubusercontent.com/80536675/553915983-ded17cdc-f23c-404f-a9f6-074fbe74816c.mp4?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NzIxNjgzMzQsIm5iZiI6MTc3MjE2ODAzNCwicGF0aCI6Ii84MDUzNjY3NS81NTM5MTU5ODMtZGVkMTdjZGMtZjIzYy00MDRmLWE5ZjYtMDc0ZmJlNzQ4MTZjLm1wND9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNjAyMjclMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjYwMjI3VDA0NTM1NFomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPWUwZmZiZTlmZjQ3OGQ4NTk1ZWQzODEzMGVlYTMyY2E0Y2E1ZjJjZTU0NTg0MDMxYmU0OThlMGVkNjc1ZGVjNDgmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.1DWDiXB20L5dr8Hx60M70zd0pL1JKkOO5roOjr2kkaQ" controls autoplay muted loop style="width:100%; max-width:100%; height:auto; border-radius:8px;"></video>
<p align="center"><b>Human vs Human:</b> Two human players competing via dedicated USB keyboards.</p>

**2. Random Agents:** Baseline agents across 33 environment families.

<video src="https://private-user-images.githubusercontent.com/80536675/553916105-2625a8f8-476c-4171-86cc-a9970cbf1665.mp4?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NzIxNjgzMzQsIm5iZiI6MTc3MjE2ODAzNCwicGF0aCI6Ii84MDUzNjY3NS81NTM5MTYxMDUtMjYyNWE4ZjgtNDc2Yy00MTcxLTg2Y2MtYTk5NzBjYmYxNjY1Lm1wND9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNjAyMjclMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjYwMjI3VDA0NTM1NFomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTQwMzljOWNhYWQwNjZjNzRlZDU4ZmM3M2M0YWJlNjYwNWM0NzM2YjkxNmQ4YTQxYmEzYTNmNzJkYWQwZGI3MWQmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.Lvwy-I3p1fNP-kmUhHtUXO7FtJ9Q_K_NCVuGbxZ6VbY" controls autoplay muted loop style="width:100%; max-width:100%; height:auto; border-radius:8px;"></video>
<p align="center"><b>Random Agents:</b> Baseline agents across 33 environment families.</p>

**3. Heterogeneous Multi-Agent Ad-Hoc Teamwork in Adversarial Settings:** Different decision-making paradigms (RL, LLM, Random) competing head-to-head in the same multi-agent environment.

<video src="https://private-user-images.githubusercontent.com/80536675/553916227-f2d79901-a93d-465b-9058-1b9cdabf311a.mp4?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NzIxNjgzMzQsIm5iZiI6MTc3MjE2ODAzNCwicGF0aCI6Ii84MDUzNjY3NS81NTM5MTYyMjctZjJkNzk5MDEtYTkzZC00NjViLTkwNTgtMWI5Y2RhYmYzMTFhLm1wND9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNjAyMjclMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjYwMjI3VDA0NTM1NFomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTU2MzNiMDJjM2ZjOWQwNjA2Y2Q1NWI0MzljMzhmNTNlOTlmMzU4OWNjOGQ1Y2NmMDdlODVkZDBjZTRhOWI4ODgmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.NQwmmI3Yyu_ePoPhLvU8ZAvp-Jn-q3q90j48n7-TEFk" controls autoplay muted loop style="width:100%; max-width:100%; height:auto; border-radius:8px;"></video>
<p align="center"><b>Heterogeneous Multi-Agent Ad-Hoc Teamwork in Adversarial Settings:</b> Different decision-making paradigms (RL, LLM, Random) competing head-to-head in the same multi-agent environment.</p>

**4. Homogeneous Teams: Random vs LLM:** Two homogeneous teams (all-Random vs all-LLM) competing in the same multi-agent environment.

<video src="https://private-user-images.githubusercontent.com/80536675/553916417-2ae1665b-3a57-44be-98a3-4e7223b37628.mp4?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NzIxNjgzMzQsIm5iZiI6MTc3MjE2ODAzNCwicGF0aCI6Ii84MDUzNjY3NS81NTM5MTY0MTctMmFlMTY2NWItM2E1Ny00NGJlLTk4YTMtNGU3MjIzYjM3NjI4Lm1wND9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNjAyMjclMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjYwMjI3VDA0NTM1NFomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTIyYzllNjc4OTYwNzg5MGFkZDA5N2Q4ZTZmMzk4Yzg0ZTYzOGNmYTgxNGZkYTU2OWYzMGZiNDk2NmJjY2FiYjMmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.FK9hfbmCA1GOl80CJkenINfr5iD5CckeipYVYfFy3_Y" controls autoplay muted loop style="width:100%; max-width:100%; height:auto; border-radius:8px;"></video>
<p align="center"><b>Homogeneous Teams: Random vs LLM:</b> Two homogeneous teams (all-Random vs all-LLM) competing in the same multi-agent environment.</p>

## Comparison with Existing Frameworks

Existing frameworks are paradigm-siloed. No prior framework allowed fair, reproducible, head-to-head comparison between RL agents and LLM agents in the same multi-agent environment.

**Column Definitions:**

1. **Platform GUI**: real-time visualization during execution
2. **Cross-Paradigm**: infrastructure for comparing different agent types (e.g., RL vs. LLM) on identical environment instances with shared random seeds for reproducible head-to-head evaluation

**Legend:** ✔️ Supported | ❌ Not supported | 🔵 Partial


| System                   | RL  | LLM | VLM | Human | Framework | Platform GUI | Cross-Paradigm |
| ------------------------ | --- | --- | --- | ----- | --------- | ------------ | -------------- |
| RLlib                    | ✔️  | ❌   | ❌   | ❌     | ✔️        | ❌            | ❌              |
| CleanRL                  | ✔️  | ❌   | ❌   | ❌     | ✔️        | ❌            | ❌              |
| Tianshou                 | ✔️  | ❌   | ❌   | ❌     | ✔️        | ❌            | ❌              |
| Acme                     | ✔️  | ❌   | ❌   | ❌     | ✔️        | ❌            | ❌              |
| XuanCe                   | ✔️  | ❌   | ❌   | ❌     | ✔️        | ❌            | ❌              |
| OpenRL                   | ✔️  | ❌   | ❌   | ❌     | ✔️        | ❌            | ❌              |
| Stable-Baselines3        | ✔️  | ❌   | ❌   | ❌     | ✔️        | ❌            | ❌              |
| Coach                    | ✔️  | ❌   | ❌   | ❌     | ✔️        | ✔️           | ❌              |
| BenchMARL                | ✔️  | ❌   | ❌   | ❌     | ✔️        | ❌            | ❌              |
| HeMAC                    | ✔️  | ❌   | ❌   | ❌     | ✔️        | ❌            | ❌              |
| Overcooked-AI            | ✔️  | ❌   | ❌   | ✔️    | ✔️        | ❌            | ❌              |
| BALROG                   | ❌   | ✔️  | ✔️  | ❌     | ✔️        | ❌            | ❌              |
| TextArena                | ❌   | ✔️  | ❌   | ✔️    | ✔️        | ❌            | ❌              |
| GameBench                | ❌   | ✔️  | ❌   | ❌     | ✔️        | ❌            | ❌              |
| lmgame-Bench             | ❌   | ✔️  | ❌   | ❌     | ✔️        | ❌            | ❌              |
| LLM Chess                | ✔️  | ✔️  | ❌   | ❌     | ✔️        | ❌            | ❌              |
| LLM-Game-Bench           | ❌   | ✔️  | ❌   | ❌     | ✔️        | 🔵           | ❌              |
| AgentBench               | ❌   | ✔️  | ❌   | ❌     | ✔️        | ❌            | ❌              |
| MultiAgentBench          | ❌   | ✔️  | ❌   | ❌     | ✔️        | ❌            | ❌              |
| GAMEBoT                  | ❌   | ✔️  | ❌   | ❌     | ✔️        | ❌            | ❌              |
| Collab-Overcooked        | 🔵  | ✔️  | ❌   | ❌     | ✔️        | ❌            | ❌              |
| BotzoneBench             | ❌   | ✔️  | ❌   | ❌     | ✔️        | ❌            | ❌              |
| AgentGym                 | ❌   | ✔️  | ❌   | ❌     | ✔️        | ❌            | ❌              |
| Game Reasoning Arena     | ✔️  | ✔️  | 🔵  | 🔵    | ✔️        | ❌            | ❌              |
| CREW                     | ✔️  | ❌   | ❌   | ✔️    | ✔️        | ❌            | ❌              |
| LLM-PySC2                | ✔️  | ✔️  | ❌   | ❌     | ✔️        | ❌            | ❌              |
| **MOSAIC (Ours)**        | ✔️  | ✔️  | ✔️  | ✔️    | ❌        | ✔️           | ✔️             |


**MOSAIC introduces an agent-level interface enabling agent-mixing across fundamentally different decision-making paradigms.**

## Policy Mappings for Heterogeneous Multi-Agent Systems

Heterogeneous multi-agent systems require each agent to be configured **independently** while sharing resources where appropriate. MOSAIC enables this through **flexible policy mappings**: one-to-one (independent policies) and one-to-many (shared policies via link groups).

<div align="center">
  <img src="docs/source/_static/figures/policy_mapping_modes.png" alt="Policy Mapping Modes" width="100%">
  <p><em>One-to-One (independent policies) and One-to-Many (shared policies via link groups).</em></p>
</div>

**Why this matters:**

Without flexible policy mappings, you're forced to choose between:

- **Manual configuration:** Copy-paste errors, update fragility, no visual indication of sharing
- **Forced homogeneity:** All agents must use the same worker type, no heterogeneity possible

With flexible policy mappings, you can:

- **Mix paradigms freely**: RL, LLM, Human, Random agents in the same environment
- **Share resources intelligently**: Link groups for RL agents trained together (MAPPO/IPPO)
- **Configure independently**: Each agent slot has its own settings and worker type
- **Update automatically**: Change primary agent's policy, all linked agents update

**Example: Heterogeneous 2v2 Soccer**

```python
# Green team: RL + LLM | Blue team: RL + Random
config = OperatorConfig.multi_agent(
    player_workers={
        "agent_0": WorkerAssignment(worker_id="xuance_worker", ...),  # RL  green agent multigrid_sports
        "agent_1": WorkerAssignment(worker_id="llm_worker", ...),     # LLM
        "agent_2": WorkerAssignment(worker_id="xuance_worker", ...),  # RL  blue agent multigrid_sports
        "agent_3": WorkerAssignment(worker_id="random_worker", ...),  # Random
    },
    link_groups={
        "operator_0_link_0": LinkGroup(
            primary_agent="agent_0",
            linked_agents=["agent_2"],  # Agents 0 and 2 share MAPPO policy
            policy_path="/path/to/mappo_1v1.pth",
        ),
    },
)
```

See the [Policy Mappings documentation](docs/source/documents/architecture/operators/policy_mappings.rst) for complete documentation, including a complex 3vs3 heterogeneous scenario with MAPPO + PPO + Random agents.

## FastLane: Zero-Overhead Live Visualization

Existing RL frameworks either render in-process (blocking training) or stream via network sockets (serialization overhead). MOSAIC's **FastLane** is the first shared-memory frame streaming system in RL: it streams rendered RGB frames from training worker subprocesses directly into POSIX shared memory via a lock-free SPSC ring buffer, achieving ~60 Hz live visualization with **zero measurable training overhead**.

- **Zero serialization**: raw `memcpy` into shared memory, no encoding, no pipes, no sockets
- **Fully decoupled**: the writer never waits for the reader (2.5% throughput variance across no-reader, 1 Hz, and 60 Hz reader conditions)
- **Correct**: zero torn reads across 155K frames and zero memory ordering errors across 700K frames
- **Fast**: 2.9 μs publish latency at 84x84, 46 μs at HD (640x480), 362x faster than the 60 Hz budget

| Metric | Value | Condition |
|--------|-------|-----------|
| Publish latency (p50) | 2.9 μs | 84x84 RGB |
| Throughput at HD | 21,689 fps | 640x480 (921 KB/frame) |
| Writer decoupling | 2.5% variance | No reader / 1 Hz / 60 Hz |
| Torn reads | 0 / 155,000 | Concurrent writer + reader |

![FastLane Writer Decoupling](docs/source/_static/figures/benchmarks/fastlane_fig_b_decoupling.png)

*Writer throughput is independent of reader speed: 337K fps with no reader, 329K fps with a 1 Hz reader, 328K fps with a 60 Hz reader (2.5% variance).*

**Limitation:** FastLane requires the training worker and the GUI to run on the same machine (POSIX shared memory cannot cross network boundaries).

See the [FastLane documentation](docs/source/documents/rendering_tabs/fastlane.rst) for the full architecture, empirical benchmarks, prior art comparison, and limitations.

> **Note:** The complementary Slow Lane is not used during training. It records high-quality human gameplay replays via gRPC and SQLite WAL storage, producing structured datasets suitable for world model training or imitation learning.

## Experimental Configurations

Heterogeneous decision-making enables a systematic ablation matrix for cross-paradigm research. 

### Formal Notation

| Symbol | Description |
| --- | --- |
| **Agent Types** | |
| $\pi^{\text{RL}}_i$ | RL policy trained via reinforcement learning |
| $\bar{\pi}^{\text{RL}}_i$ | Frozen RL policy (parameters $\theta_i$ fixed; no further learning) |
| $\lambda^{\text{LLM}}_j$ | LLM agent (large language model, text-only observations) |
| $\psi^{\text{VLM}}_k$ | VLM agent (vision-language model, multimodal observations) |
| $h_m$ | Human operator (interactive GUI control) |
| $\rho$ | Uniform random baseline policy |
| $\nu$ | No-op baseline policy (null action at every step) |
| **Agent Populations and Sizes** | |
| $\Pi^{\text{RL}} = \{\pi^{\text{RL}}_i\}_{i=1}^{n_{\text{RL}}}$ | Population of RL policies of size $n_{\text{RL}}$ |
| $\Lambda^{\text{LLM}} = \{\lambda^{\text{LLM}}_j\}_{j=1}^{n_{\text{LLM}}}$ | Population of LLM agents of size $n_{\text{LLM}}$ |
| $\Psi^{\text{VLM}} = \{\psi^{\text{VLM}}_k\}_{k=1}^{n_{\text{VLM}}}$ | Population of VLM agents of size $n_{\text{VLM}}$ |
| $\mathcal{H} = \{h_m\}_{m=1}^{n_{\text{H}}}$ | Population of human operators of size $n_{\text{H}}$ |
| $N = n_{\text{RL}} + n_{\text{LLM}} + n_{\text{VLM}} + n_{\text{H}}$ | Total number of agents in the system |
| **Team Partitions** | |
| $K$ | Number of teams ($K \geq 1$) |
| $\mathcal{T}_1, \mathcal{T}_2, \ldots, \mathcal{T}_K$ | Disjoint team partitions covering all agents: $\mathcal{T}_i \cap \mathcal{T}_j = \emptyset$ for $i \neq j$, $\bigcup_{k=1}^{K} \mathcal{T}_k = \{1,\ldots,N\}$ |
| $n_k = \|\mathcal{T}_k\|$ | Size of team $k$; $\sum_{k=1}^{K} n_k = N$ |
| $\mathcal{T}_A \equiv \mathcal{T}_1,\ \mathcal{T}_B \equiv \mathcal{T}_2$; $n_A \equiv n_1,\ n_B \equiv n_2$ | Two-team convention used in the experimental configurations below ($K=2$ case) |
| **Observation and Action Spaces** | |
| $\mathcal{O}^{\text{RL}} = \mathbb{R}^d$ | RL observation space (continuous tensor) |
| $\mathcal{O}^{\text{LLM}} = \Sigma^{*}$ | LLM observation space (strings over alphabet $\Sigma$) |
| $\mathcal{O}^{\text{VLM}} = \Sigma^{*} \times \mathbb{R}^{H \times W \times C}$ | VLM observation space (multimodal: text and RGB image) |
| $\mathcal{O}^{\text{H}} = \mathbb{R}^{H \times W \times C}$ | Human observation space (rendered RGB image) |
| $\mathcal{A} = \{1,2,\dots,K\}$ | Discrete action space (shared after paradigm-specific parsing) |
| $\phi: \Sigma^{*} \to \mathcal{A}$ | Deterministic parsing function mapping LLM/VLM text to actions |
| **Policy Inputs and Action Composition** | |
| $\pi_\theta$ | MARL policy parameterized by $\theta$ (shared across agents or per-agent, depending on algorithm) |
| $\tau_i^t$ | Trajectory (observation-action history) of agent $i$ up to time $t$ |
| $e_i$ | Agent identity/embedding for agent $i$ |
| $a_i^t \sim \pi_\theta(\cdot \mid \tau_i^t, e_i)$ | Action of agent $i$ at time $t$, sampled from the MARL policy |
| $a^{\text{sub},t} = (a_1^{\text{sub},t}, \ldots, a_n^{\text{sub},t})$ | Submitted joint action at time $t$ (input to the environment step) |
| $a_i^{\text{ext},t}$ | External action injected by an operator at slot $i$ during deployment |
| $\text{Merge}(a^{\text{MARL}}, a^{\text{ext}}) \to a^{\text{sub},t}$ | Action Merger composing MARL outputs with external actions before submission |
| $(s^{t+1}, r^t, \text{Done})$ | Environment transition after applying $a^{\text{sub},t}$ |

### Training and Deployment Phases

![Training and Deployment Phases](docs/source/images/architecture/training_deployment_phases.png)

**(A) Cooperative training-phase (Cooperative Self-Play):** All $n$ agents submit
observations $O_i^t, e_i$ to a shared MARL policy $\pi_\theta$ each timestep. The
policy outputs an action $a_i^t \sim \pi_\theta(\cdot \mid \tau_i^t, e_i)$ per
agent; these are assembled into the joint action $a^{\text{sub},t}$ submitted to
the environment. This is standard cooperative multi-agent training (e.g.,
MAPPO, QMIX) on tasks like Basketball, where all agents share a common
objective.

**(B) Adversarial training-phase (Competitive Self-Play):** The same structure
applies to competitive settings such as Soccer, where opposing teams (green vs.
blue) are trained via self-play against each other under the same policy
$\pi_\theta$.

**(C) Cooperative deployment-phase (Cross-Paradigm External Agent):** At
deployment, one or more external operators join the team alongside the MARL
policy's agents. The MARL policy still produces actions for all $n$ of its
trained agents, and each external operator contributes one additional action
per timestep. An operator may be an LLM, or an RL policy trained under a
different algorithm (MAPPO, IPPO, VDPPO, QMIX, CommNet, IC3NET, MAT). The
**Action Merger** combines the MARL policy outputs with the external actions
($\text{Merge}(a^{\text{MARL}}, a^{\text{ext}}) \to a^{\text{sub},t}$) to form
the joint action submitted to the environment. This lets researchers train a
team of $n$ agents and deploy with $n + k$ agents by adding $k$ external
decision-makers at evaluation time, without retraining the original team.

See [Policy Mappings for Heterogeneous Multi-Agent Systems](documents/architecture/operators/policy_mappings.html)
for the full mechanism and configuration reference.

### Adversarial Cross‑Paradigm Matchups

The first set of configurations establishes single-paradigm baselines before introducing cross-paradigm matchups to measure relative performance.
Let $\mathcal{T}_A$ and $\mathcal{T}_B$ denote disjoint team partitions with $|\mathcal{T}_A| = n_A$ and $|\mathcal{T}_B| = n_B$.
For each team $\mathcal{T}_k$ ($k \in A,B$), we define its paradigm composition as $(\Pi^{\text{RL}}_k, \Lambda^{\text{LLM}}_k, \Psi^{\text{VLM}}_k, \mathcal{H}_k)$ where $\Pi^{\text{RL}}_k + \Lambda^{\text{LLM}}_k + \Psi^{\text{VLM}}_k + \mathcal{H}_k = n_k$.


| Config | Team A Composition           | Team B Composition           | Purpose                          |
| ------ | ---------------------------- | ---------------------------- | -------------------------------- |
| **A1** | $\Pi^{\text{RL}}_A = 2$      | $\Pi^{\text{RL}}_B = 2$      | Homogeneous RL baseline          |
| **A2** | $\Lambda^{\text{LLM}}_A = 2$ | $\Lambda^{\text{LLM}}_B = 2$ | Homogeneous LLM baseline         |
| **A3** | $\Psi^{\text{VLM}}_A = 2$    | $\Psi^{\text{VLM}}_B = 2$    | Homogeneous VLM baseline         |
| **A4** | $\Pi^{\text{RL}}_A = 2$      | $\Lambda^{\text{LLM}}_B = 2$ | Cross-paradigm (RL vs LLM)       |
| **A5** | $\Pi^{\text{RL}}_A = 2$      | $\Psi^{\text{VLM}}_B = 2$    | Cross-paradigm (RL vs VLM)       |
| **A6** | $\Lambda^{\text{LLM}}_A = 2$ | $\Psi^{\text{VLM}}_B = 2$    | Cross-paradigm (LLM vs VLM)      |
| **A7** | $\Pi^{\text{RL}}_A = 2$      | $\rho$ baseline ($n_B = 2$)  | Sanity check (trained vs random) |


Configurations A1-A3 measure the performance ceiling for homogeneous teams within each paradigm: RL policies trained via MARL, LLM agents reasoning via text-based decision-making, and VLM agents processing multimodal observations. Configurations A4-A6 address the central cross-paradigm research questions: under identical environmental conditions and shared random seeds, does a team of RL policies outperform teams of LLM or VLM agents, and how do LLM and VLM agents compare head-to-head? A7 serves as a sanity check, confirming that trained agents significantly outperform uniform-random baseline policies.

### Cooperative Heterogeneous Teams

The second set of configurations examines intra-team heterogeneity by mixing paradigms **within** a team. These configurations test whether LLM or VLM agents ($\lambda^{\text{LLM}}$ or $\psi^{\text{VLM}}$) can effectively cooperate with a frozen RL policy $\bar{\pi}^{\text{RL}}$ that was trained without any partner model.


| Config | Team A Composition                              | Team B Composition                           | Research Question                                          |
| ------ | ----------------------------------------------- | -------------------------------------------- | ---------------------------------------------------------- |
| **C1** | $\bar{\pi}^{\text{RL}}$, $\lambda^{\text{LLM}}$ | $\bar{\pi}^{\text{RL}}$, $\rho$ baseline     | Does $\lambda^{\text{LLM}}$ outperform $\rho$ as teammate? |
| **C2** | $\bar{\pi}^{\text{RL}}$, $\lambda^{\text{LLM}}$ | $\bar{\pi}^{\text{RL}}$, $\nu$ baseline      | Does $\lambda^{\text{LLM}}$ actively contribute?           |
| **C3** | $\bar{\pi}^{\text{RL}}$, $\psi^{\text{VLM}}$    | $\bar{\pi}^{\text{RL}}$, $\rho$ baseline     | Does $\psi^{\text{VLM}}$ outperform $\rho$ as teammate?    |
| **C4** | $\bar{\pi}^{\text{RL}}$, $\psi^{\text{VLM}}$    | $\bar{\pi}^{\text{RL}}$, $\nu$ baseline      | Does $\psi^{\text{VLM}}$ actively contribute?              |
| **C5** | $\Pi^{\text{RL}}_A = 2$                         | $\Pi^{\text{RL}}_B = 2$                      | Solo-pair baseline (no co-training)                        |
| **C6** | $\bar{\pi}^{\text{RL}}$, $\lambda^{\text{LLM}}$ | $\Pi^{\text{RL}}_B = 2$ (co-trained)         | Can zero-shot LLM teaming match co-training?               |
| **C7** | $\bar{\pi}^{\text{RL}}$, $\psi^{\text{VLM}}$    | $\Pi^{\text{RL}}_B = 2$ (co-trained)         | Can zero-shot VLM teaming match co-training?               |
| **C8** | $\bar{\pi}^{\text{RL}}$, $\lambda^{\text{LLM}}$ | $\bar{\pi}^{\text{RL}}$, $\psi^{\text{VLM}}$ | LLM vs VLM as heterogeneous teammates                      |


All RL policies are trained solo ($N=1$) and frozen before deployment; LLM/VLM agents are zero-shot. Configurations C1-C2 and C3-C4 test whether LLM and VLM agents can serve as effective teammates for frozen RL policies. C5 serves as the fair comparison baseline: two independently trained solo experts paired at evaluation time. C6-C7 compare zero-shot cross-paradigm teaming against co-trained RL teams. C8 directly compares LLM and VLM agents as teammates within heterogeneous teams.

## Supported Environment Families (33)

| Family | Description | Example | Status |
|--------|-------------|---------|--------|
| **Gymnasium** | Standard single-agent RL (Toy Text, Classic Control, Box2D, MuJoCo) | <img src="docs/source/images/envs/gymnasium/cartpole.gif" width="200"> | Human-Control: ✅, Single-Agent: ✅, Multi-Agent: ❌ |
| **Atari / ALE** | 128 classic Atari 2600 games | <img src="docs/source/images/envs/atari/atari.gif" width="200"> | Human-Control: ✅, Single-Agent: ✅, Multi-Agent: ✅ |
| **MiniGrid** | Procedural grid-world navigation | <img src="docs/source/images/envs/minigrid/minigrid.gif" width="200"> | Human-Control: ✅, Single-Agent: ✅, Multi-Agent: ❌ |
| **BabyAI** | Language-grounded instruction following | <img src="docs/source/images/envs/babyai/GoTo.gif" width="200"> | Human-Control: ✅, Single-Agent: ✅, Multi-Agent: ❌ |
| **Griddly** | High-performance grid worlds with C++ backend & Vulkan GPU rendering (34 envs) | <img src="docs/source/images/envs/griddly/griddly.gif" width="200"> | Human-Control: ✅, Single-Agent: 📋, Multi-Agent: ✅ |
| **ViZDoom** | Doom-based first-person visual RL | <img src="docs/source/images/envs/vizdoom/vizdoom.gif" width="200"> | Human-Control: ✅, Single-Agent: 📋, Multi-Agent: ❌ |
| **MiniHack** | Roguelike sandbox built on NetHack (NLE) | <img src="docs/source/images/envs/minihack/minihack.gif" width="200"> | Human-Control: ✅, Single-Agent: 📋, Multi-Agent: ❌ |
| **NetHack** | Full NetHack roguelike game via NLE | <img src="docs/source/images/envs/nethack/nethack.gif" width="200"> | Human-Control: ✅, Single-Agent: 📋, Multi-Agent: ❌ |
| **Crafter** | Open-world survival benchmark | <img src="docs/source/images/envs/crafter/crafter.gif" width="200"> | Human-Control: ✅, Single-Agent: 📋, Multi-Agent: ❌ |
| **Craftax** | JAX-accelerated open-world survival (Crafter-in-JAX, 22 achievements) | <img src="https://raw.githubusercontent.com/MichaelTMatthews/Craftax/main/images/archery.gif" width="200"> | Human-Control: ✅, Single-Agent: ✅, Multi-Agent: ✅ |
| **Procgen** | 16 procedurally generated environments | <img src="docs/source/images/envs/procgen/coinrun.gif" width="200"> | Human-Control: ✅, Single-Agent: ✅, Multi-Agent: ❌ |
| **BabaIsAI** | Rule-manipulation puzzles | <img src="docs/source/images/envs/babaisai/babaisai.png" width="200"> | Human-Control: ✅, Single-Agent: 📋, Multi-Agent: ❌ |
| **TextWorld** | Text-based interactive fiction (Microsoft Research) | <img src="docs/source/images/envs/textworld/textworld.gif" width="200"> | Human-Control: ✅, Single-Agent: 📋, Multi-Agent: ❌ |
| **Jumanji** | JAX-accelerated logic/routing/packing (25 envs) | <img src="docs/source/images/envs/jumanji/jumanji.gif" width="200"> | Human-Control: ✅, Single-Agent: 📋, Multi-Agent: ❌ |
| **PyBullet Drones** | Quadcopter physics simulation | <img src="docs/source/images/envs/pybullet_drones/pybullet_drones.gif" width="200"> | Human-Control: ❌, Single-Agent: ✅, Multi-Agent: ✅ |
| **PettingZoo Classic** | Turn-based board games (AEC) | <img src="docs/source/images/envs/pettingzoo/pettingzoo.gif" width="200"> | Human-Control: ✅, Single-Agent: ❌, Multi-Agent: ✅ |
| **OpenSpiel** | Board games via DeepMind's OpenSpiel + Shimmy (Chess, Go, Checkers) | <img src="docs/source/images/envs/openspiel/openspiel.gif" width="200"> | Human-Control: ✅, Single-Agent: ✅, Multi-Agent: ❌ |
| **MOSAIC MultiGrid** | Competitive team sports (view_size=3) | <img src="docs/source/images/envs/mosaic_multigrid/mosaic_multigrid.gif" width="200"> | Human-Control: ✅, Single-Agent: ✅, Multi-Agent: ✅ |
| **INI MultiGrid** | Cooperative exploration (view_size=7) | <img src="docs/source/images/envs/multigrid_ini/multigrid_ini.gif" width="200"> | Human-Control: ✅, Single-Agent: ❌, Multi-Agent: ✅ |
| **SocialJax** | JAX-accelerated sequential social dilemmas (9 envs) | <img src="docs/source/images/envs/socialjax/socialjax_common.gif" width="200"> | Human-Control: ❌, Single-Agent: ❌, Multi-Agent: ✅ |
| **Melting Pot** | Social multi-agent scenarios (up to 16 agents) | <img src="docs/source/images/envs/meltingpot/meltingpot.gif" width="200"> | Human-Control: ✅, Single-Agent: ❌, Multi-Agent: ✅ |
| **Overcooked** | Cooperative cooking (2 agents) | <img src="docs/source/images/envs/overcooked/overcooked_layouts.gif" width="200"> | Human-Control: ✅, Single-Agent: ❌, Multi-Agent: ✅ |
| **SMAC** | StarCraft Multi-Agent Challenge (hand-designed maps) | <img src="docs/source/images/envs/smac/smac.gif" width="200"> | Human-Control: ✅, Single-Agent: ✅, Multi-Agent: ✅ |
| **SMACv2** | StarCraft Multi-Agent Challenge v2 (procedural units) | <img src="docs/source/images/envs/smacv2/smacv2.png" width="200"> | Human-Control: ✅, Single-Agent: ✅, Multi-Agent: ✅ |
| **RWARE** | Cooperative warehouse delivery | <img src="docs/source/images/envs/rware/rware.gif" width="200"> | Human-Control: ✅, Single-Agent: ❌, Multi-Agent: ✅ |
| **HeMAC** | Heterogeneous multi-agent challenge (Quadcopters, Observers, Provisioners) | | Human-Control: ✅, Single-Agent: ❌, Multi-Agent: ✅ |
| **MuJoCo** | Continuous-control robotics tasks | <img src="docs/source/images/envs/mujoco/ant.gif" width="200"> | Human-Control: ✅, Single-Agent: ✅, Multi-Agent: ❌ |
| **Google Research Football** | 11-vs-11 football/soccer simulation (Google Research) | <img src="https://1.bp.blogspot.com/-HkcNiCL13cc/XPqSVOgTwMI/AAAAAAAAEM4/OoK_qoM14QA6VNQ79sWeS97TKBhCD7CzQCLcBGAs/s640/image3.gif" width="200"> | Human-Control: ✅, Single-Agent: ✅, Multi-Agent: ✅ |
| **MarLo** *(experimental)* | Multi-Agent RL in Minecraft (2018 MarLo Challenge) | <img src="https://media.giphy.com/media/u45fNQxG59wfnRpzwJ/giphy.gif" width="200"> | Human-Control: ✅, Single-Agent: 🚧, Multi-Agent: 🚧 |
| **Malmo** *(experimental)* | Microsoft Research AI platform in Minecraft | <img src="https://www.microsoft.com/en-us/research/wp-content/uploads/2016/06/malmo_human_ai_interaction-web.png" width="200"> | Human-Control: ✅, Single-Agent: 🚧, Multi-Agent: 🚧 |

## Roadmap

MOSAIC is actively expanding to support more diverse and complex environments, simulators, and algorithms.

### Environments

| Environment | Status | Description |
|-------------|--------|-------------|
| **Minecraft: Malmo** | 🚧 Experimental | Microsoft Research AI platform built on Minecraft |
| **Minecraft: MarLo** | 🚧 Experimental | Multi-Agent RL environments for Minecraft (2018 MarLo Challenge) |
| **Minecraft: MineRL / Mindcraft** | 📋 Planned | Minecraft AI research platforms |

### Simulators

| Simulator | Status | Description |
|-----------|--------|-------------|
| **AirSim** | 📋 Planned | Microsoft Research drone/car simulator for autonomous vehicles |
| **Godot Engine** | 📋 Planned | Free, open-source game engine for custom RL environments |

### Algorithms

More algorithms coming soon, including additional multi-agent and hierarchical RL methods.

---

**Legend:** 🚧 Experimental (under development) | 📋 Planned (on roadmap)



## Two Evaluation Modes

MOSAIC provides two evaluation modes designed for reproducibility:

<video src="https://private-user-images.githubusercontent.com/80536675/553915409-ea9ebc18-2216-4fb2-913c-5d354ebea56e.mp4?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NzIxNjgzMzQsIm5iZiI6MTc3MjE2ODAzNCwicGF0aCI6Ii84MDUzNjY3NS81NTM5MTU0MDktZWE5ZWJjMTgtMjIxNi00ZmIyLTkxM2MtNWQzNTRlYmVhNTZlLm1wND9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNjAyMjclMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjYwMjI3VDA0NTM1NFomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTAyZDkzMWQwZjczOTI1NzFiNjk2MGU4N2I2ZTAxNDE0M2Y2YmQxNjM5ZTAxMTkxYzA5NGU4ZGE3YzZkZmJkZWEmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.9BimKYjxGrdN_g1eDbXC69weDC7-My85Gl-ou2wNzxQ" controls autoplay muted loop style="width:100%; max-width:100%; height:auto; border-radius:8px;"></video>
<p align="center"><b>Manual Mode:</b> Side-by-side lock-step evaluation with shared seeds.</p>

- **Manual Mode:** side-by-side comparison where multiple operators step through the same environment with shared seeds, letting researchers visually inspect decision-making differences between paradigms in real time.

<video src="https://private-user-images.githubusercontent.com/80536675/553915854-a9b3f6f4-661c-492f-b43f-34d7125a6d2e.mp4?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NzIxNjgzMzQsIm5iZiI6MTc3MjE2ODAzNCwicGF0aCI6Ii84MDUzNjY3NS81NTM5MTU4NTQtYTliM2Y2ZjQtNjYxYy00OTJmLWI0M2YtMzRkNzEyNWE2ZDJlLm1wND9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNjAyMjclMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjYwMjI3VDA0NTM1NFomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPWNhMjM0ZjBjYjU1NWFlNmYxOGU2Yzc2N2U0ODE4OTYzZGVkYTc5YTIyMjM5YzRjODU0MTRhODFhOWI4ZDU3NmImWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.tk8Ezu0ivgFlp-xm6YEkIsWlbFPcpOgSq30Hq_JEJvs" controls autoplay muted loop style="width:100%; max-width:100%; height:auto; border-radius:8px;"></video>
<p align="center"><b>Script Mode:</b> Automated batch evaluation with deterministic seed sequences.</p>

- **Script Mode:** automated, long-running evaluation driven by Python scripts that define operator configurations, worker assignments, seed sequences, and episode counts. Scripts execute deterministically with no manual intervention, producing reproducible telemetry logs (JSONL) for every step and episode.

All evaluation runs share **identical conditions**: same environment seeds, same observations, and unified telemetry. Script Mode additionally supports **procedural seeds** (different seed per episode to test generalization) and **fixed seeds** (same seed every episode to isolate agent behaviour), with configurable step pacing for visual inspection or headless batch execution.


## Supported Workers (12)

| Worker | Algorithms | Environments |
| ------ | ---------- | ------------ |
| **[CleanRL](https://github.com/vwxyzjn/cleanrl)** | PPO, DQN, SAC, TD3, DDPG, C51 | Atari, MuJoCo, CartPole, LunarLander |
| **[XuanCe](https://github.com/agi-brain/xuance)** | MAPPO, QMIX, MADDPG, VDN, COMA | MPE, SMAC, MiniGrid, MuJoCo |
| **[Ray RLlib](https://docs.ray.io/en/latest/rllib/)** | PPO, IMPALA, APPO | Atari, MuJoCo, custom Gymnasium envs |
| **[JaxMARL](https://github.com/FLAIROx/JaxMARL)** | IPPO, MAPPO, CommNet, IC3Net, MAT | MultiGrid, SocialJax, MPE, MABrax, Overcooked, Hanabi |
| **[Mava](https://github.com/instadeepai/Mava)** | IPPO, MAPPO, RecurrentIPPO | RobotWarehouse, SocialJax |
| **[BALROG](https://github.com/balrog-ai/BALROG)** | GPT-4o, Claude 3, Gemini | NetHack, BabyAI, Crafter |
| **[Chess LLM](docs/source/documents/architecture/workers/integrated_workers/Chess_LLM_Worker)** | GPT-4o, Claude, Gemini (multi-turn dialog) | PettingZoo Chess |
| **[MOSAIC LLM](docs/source/documents/architecture/workers/integrated_workers/MOSAIC_LLM_Worker)** | CoT, Theory of Mind, coordination strategies | MultiGrid, BabyAI, MeltingPot, PettingZoo |
| **[MOSAIC VLM](docs/source/documents/architecture/workers/integrated_workers/MOSAIC_VLM_Worker)** | CoT, Theory of Mind, coordination strategies | MultiGrid, BabyAI, MeltingPot, PettingZoo |
| **[MOSAIC Human Worker](docs/source/documents/architecture/workers/integrated_workers/MOSAIC_Human_Worker)** | Human keyboard input | MiniGrid, Crafter, Chess, NetHack, GRF |
| **[MOSAIC Random Worker](docs/source/documents/architecture/workers/integrated_workers/MOSAIC_Random_Worker)** | Random policy | All 33 environment families |
| **[MOSAIC Passive Worker](docs/source/documents/architecture/workers/integrated_workers/MOSAIC_Passive_Worker)** | No-op / Still policy | All 33 environment families |


## Installation

```bash
# Clone the repository
git clone https://github.com/Abdulhamid97Mousa/mosaic.git
cd mosaic

# Create virtual environment (Python 3.10-3.12)
python3.11 -m venv .venv
source .venv/bin/activate

# Install core GUI
pip install -e .
```

Install only what you need. **Workers** and **environment families** are independent extras:

```bash
# Worker (CleanRL) + environment family (MiniGrid)
pip install -e ".[cleanrl,minigrid]"

# Multi-agent worker (XuanCe) + competitive environments
pip install -e ".[xuance,multigrid_sports]"

# Everything
pip install -e ".[full]"
```

## Quick Start

```bash
# Launch with trainer daemon (recommended)
./run.sh

# Or launch GUI only
python -m gym_gui
```

## Contributing

Contributions are welcome: new workers, new environment families, baselines,
benchmark runs, bug fixes, and documentation. See
[CONTRIBUTING.md](CONTRIBUTING.md) for the dev setup, lint/test workflow, and
PR checklist, and [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) for community
expectations.

## Citing MOSAIC

If you use MOSAIC in your research, please cite:

```bibtex
@misc{mousa2026mosaicunifiedplatformcrossparadigm,
      title={MOSAIC: A Universal Agent-Level Interface for Cross-Paradigm Agent Mixing and Human-AI Collaboration},
      author={Abdulhamid M. Mousa and Ming Liu and Rakhmonberdi Khajiev and Jalaledin M. Azzabi and Abdulkarim M. Mousa and Peng Yong and Yunusa Haruna and Jinhui Pang},
      year={2026},
      eprint={2603.01260},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2603.01260}, 
}
```

## License

MIT License. See [LICENSE](LICENSE).

## Acknowledgments

- [Gymnasium](https://gymnasium.farama.org/): Standard RL API
- [PettingZoo](https://pettingzoo.farama.org/): Multi-agent environments
- [CleanRL](https://github.com/vwxyzjn/cleanrl): Clean RL implementations
- [XuanCe](https://github.com/agi-brain/xuance): Multi-agent RL algorithms
- [Ray RLlib](https://docs.ray.io/en/latest/rllib/): Distributed RL training
- [JaxMARL](https://github.com/FLAIROx/JaxMARL): JAX-accelerated multi-agent RL
- [Mava](https://github.com/instadeepai/Mava): Distributed JAX MARL framework (InstaDeep)
- [BALROG](https://github.com/balrog-ai/BALROG): LLM/VLM agent benchmark
- [llm_chess](https://github.com/maxim-saplin/llm_chess): LLM chess evaluation
- [Google Research Football](https://github.com/google-research/football): Physics-based football simulation
- [SMAC](https://github.com/oxwhirl/smac): StarCraft Multi-Agent Challenge
- [SMACv2](https://github.com/oxwhirl/smacv2): StarCraft Multi-Agent Challenge v2 with procedural units
- [HeMAC](https://github.com/Thales-MARL/HeMAC): Heterogeneous multi-agent cooperative benchmark (Thales Group)
- [PyQt6](https://www.riverbankcomputing.com/software/pyqt/): GUI framework
