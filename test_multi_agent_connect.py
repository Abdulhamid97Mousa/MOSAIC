#!/usr/bin/env python3
"""Test multi-agent MalmoEnv connection directly (no GUI)."""
import sys, uuid, time
sys.path.insert(0, "3rd_party/environments/malmo/MalmoEnv")

import malmoenv
from pathlib import Path

xml = Path("3rd_party/environments/malmo/MalmoEnv/missions/treasurehunt.xml").read_text()
exp_uid = str(uuid.uuid4())
print(f"exp_uid: {exp_uid}")

print("\n--- Agent 0 (role=0, port=9000) ---")
env0 = malmoenv.Env()
env0.init(xml, 9000, server="localhost", port2=9000, role=0, exp_uid=exp_uid, reshape=True)
print(f"  init OK, agent_count={env0.agent_count}, actions={list(env0.action_space.actions)}")

print("\n--- Agent 1 (role=1, port2=9001) ---")
env1 = malmoenv.Env()
env1.init(xml, 9000, server="localhost", port2=9001, role=1, exp_uid=exp_uid, reshape=True)
print(f"  init OK, agent_count={env1.agent_count}, actions={list(env1.action_space.actions)}")

print("\n--- Reset Agent 0 (creates mission) ---")
try:
    obs0 = env0.reset()
    print(f"  OK, obs shape={obs0.shape}")
except Exception as e:
    print(f"  FAILED: {e}")
    import traceback; traceback.print_exc()
    sys.exit(1)

print("\n--- Reset Agent 1 (joins mission) ---")
try:
    obs1 = env1.reset()
    print(f"  OK, obs shape={obs1.shape}")
except Exception as e:
    print(f"  FAILED: {e}")
    import traceback; traceback.print_exc()
    sys.exit(1)

print("\n=== BOTH AGENTS CONNECTED ===")
env0.close()
env1.close()
print("Done.")
