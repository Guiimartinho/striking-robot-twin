# striking-robot-twin

![Python](https://img.shields.io/badge/python-3.11%2B-blue?logo=python&logoColor=white)
![MuJoCo](https://img.shields.io/badge/MuJoCo-3.10-000000)
![Gymnasium](https://img.shields.io/badge/Gymnasium-1.3-0a7bbb)
![pydantic](https://img.shields.io/badge/pydantic-2.x-e92063?logo=pydantic&logoColor=white)
![tests](https://img.shields.io/badge/tests-35%20passing-brightgreen?logo=pytest&logoColor=white)
![status](https://img.shields.io/badge/status-Phase%200-yellow)
![safety](https://img.shields.io/badge/safety-first%20(pHRI)-red)

Digital-twin-first control stack for a stationary boxing/Muay Thai training
robot. The robot throws padded punches at a trainee and uses computer vision to
teach them to dodge and defend.

Safety (physical human-robot interaction) is the design constraint that
organises the architecture, not a module bolted on. See `CLAUDE.md` for the full
context, the safety contract and the roadmap.

## Status: Phase 0 (foundations + safety)

The whole intelligence stack is built and validated in a MuJoCo digital twin
before any hardware or human. The sim and the real robot are interchangeable
behind a Hardware Abstraction Layer (HAL); nothing above the HAL knows which it
is talking to.

MVP scope: punches only (jab, cross, hook), upper body. No kicks, knees, elbows
or footwork.

## Architecture (plant-agnostic HAL)

```
Services : DrillEngine, Scoring, Telemetry
Domain   : StrikePlanner, TargetSelector, DodgeDetector, GuardDetector
Safety   : SafetyArbiter (keep-out, reach, force cap, latency margin), FaultInjector
---------------------- HAL boundary (sim <-> real) ----------------------
Interfaces : IRobotPlant, ITraineeObserver   (typing.Protocol)
   sim     : MujocoPlant, SimGTObserver
   real    : RealPlant (STM32), CameraPoseObserver (Jetson)   [stubs]
```

Hard rule: Domain, Safety and Services never import `mujoco`, `jax` or `cv2`.
They depend only on `hal.interfaces` and `core.types`.

### The keep-out math (central)

The protected volume around the head is inflated by system latency, because the
head moves between estimating it and stopping the actuator:

```
R_keepout = tracking_error + (latency_total * head_v_max) + margin
```

The `SafetyArbiter` recomputes this every cycle from the observer's live latency
and vetoes any command whose trajectory crosses an inflated protected sphere.

## Setup (Windows native, uv)

```
uv venv
uv pip install -e ".[dev]"
```

JAX+GPU / MJX scaled training does not run on native Windows; use WSL2 or Linux
for that (Phase 3). Develop the logic on Windows, train heavy on Linux.

## Commands

```
pytest                                   # tests + contract + fault injection
python scripts/run_sim.py                # run the twin (safety-gated strikes)
python scripts/viewer.py models/scene.xml
python -m robot_twin.rl.train            # RL training (Phase 3, stub)
```

Inspect the MJCF in the native viewer:

```
E:\Downloads\mujoco-3.10.0-windows-x86_64\bin\simulate.exe models\scene.xml
```

## Phase 0 gate

The SafetyArbiter rejects 100% of commands that violate keep-out / reach / force
under fault injection (high latency, keypoint dropout, the trainee lunging into
the strike). This is enforced by `tests/test_safety_arbiter.py`.

## Roadmap

| Phase | Deliverable | Gate |
|-------|-------------|------|
| 0 | Foundations + safety (HAL, SafetyArbiter, sim plant, env) | 100% rejection of unsafe commands under fault injection |
| 1 | Perception (DodgeDetector, GuardDetector) | reliable detection under injected pose noise |
| 2 | Slow closed loop (one arm, telegraphed strikes, scoring) | full drill end to end, zero safety violations |
| 3 | Multi-striker, combos, RL (DrillEngine curriculum, MJX) | policy respects safety by construction; combos match video timing |
