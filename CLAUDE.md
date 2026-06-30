# CLAUDE.md - Robô de Treino de Boxe/Muay Thai (Digital Twin First)

> Contexto persistente para o Claude Code. Leia por completo antes de qualquer
> tarefa. Em caso de conflito entre velocidade e segurança, segurança vence.

---

## 1. Missão e estado atual

Construir um robô estacionário que desfere golpes acolchoados contra um aluno e
usa visão computacional para que ele aprenda a esquivar e defender.

**Fase atual: digital twin primeiro.** Toda a inteligência (decisão, segurança,
scoring, currículo) é desenvolvida e validada em simulação MuJoCo antes de
qualquer hardware ou humano. O simulador e o robô real são intercambiáveis por
trás de uma camada de abstração (HAL). Nada acima da HAL sabe se está rodando no
sim ou no robô.

**Escopo do MVP: apenas socos (jab, cross, hook), upper body.** Cotovelos,
joelhadas e chutes estão FORA do MVP. Não introduza atuação de perna, footwork ou
balanço de corpo inteiro.

**Plataforma alvo do robô real:** percepção em NVIDIA Jetson Orin; controle de
baixo nível e segurança em STM32; atuadores Series Elastic (SEA) acionados por
BLDC. Nada disso existe ainda. O sim modela a planta SEA para que o código de
cima já nasça compatível.

---

## 2. Restrição que dita todo o resto: segurança (pHRI)

Esta é uma máquina que golpeia um humano. É interação física humano-robô
safety-critical. A camada de segurança não é um módulo entre outros: é a
restrição de projeto que organiza a arquitetura.

Regras inegociáveis que o código deve refletir:

- **Cabeça: sempre sem contato.** O alvo de qualquer golpe à altura da cabeça
  passa por um offset ATRÁS da cabeça, nunca nela. Existe um `keep_out_volume`
  ao redor do keypoint da cabeça que o atuador jamais pode penetrar.
- **Esquiva falha = tag não-impactante** (háptico/buzzer/LED no sim é um evento
  lógico), nunca impacto projetado na cabeça.
- **Corpo: contato leve é opcional e só em fase posterior,** com colete
  acolchoado, cap de energia baixo, e somente depois do envelope de energia
  validado em hardware (fora do escopo do sim).
- **Limitação de força é inerente ao mecanismo,** não só software. A mola do SEA
  é limitador passivo de energia e fonte de sensoriamento de força (via
  deflexão). O E-stop por software é backup, nunca a barreira primária.
- **End-stop mecânico de alcance:** o atuador não pode estender fisicamente além
  de `standoff + margem`, independente do comando.
- **Canal de segurança independente:** no robô real o SafetyArbiter vive no
  STM32, separado do cérebro de percepção. No sim ele é uma camada que pode
  VETAR ou ABORTAR qualquer comando vindo da decisão.

### A matemática do keep-out (central)

O volume de keep-out NÃO é estático. Tem que ser inflado pela latência do
sistema, porque entre "estimei a cabeça" e "o atuador parou" a cabeça se moveu:

```
R_keepout = erro_tracking + (latencia_total * v_max_cabeca) + margem
```

`SafetyArbiter` recebe `latency_s()` do observer e calcula `R_keepout` a cada
ciclo. Se a trajetória comandada cruza a esfera de raio `R_keepout` ao redor de
qualquer keypoint protegido, o comando é rejeitado ou abortado. Cubra isto com
testes e fault injection.

---

## 3. Arquitetura (plant-agnostic HAL)

Dependency inversion aplicada ao sim-to-real. A fronteira é a HAL.

```
Services   : DrillEngine (FSM/currículo), Scoring, Telemetry/Logger
Domain     : StrikePlanner, TargetSelector, DodgeDetector, GuardDetector
Safety     : SafetyArbiter (keep-out, reach, force cap, margem por latência)
             FaultInjector                                  [igual em sim e real]
-------------------- HAL: a fronteira sim <-> real --------------------
Interfaces : IRobotPlant            ITraineeObserver        [Protocols]
   sim     : MujocoPlant            SimGTObserver
   real    : RealPlant (STM32)      CameraPoseObserver (Jetson)
-------------------------------------------------------------------
Plant (sim): modelo MJCF (frame + braços SEA + humanoide aluno),
             física, molas SEA, injeção de latência/dropout/oponente
```

RL track: um `gymnasium.Env` envolve `IRobotPlant + ITraineeObserver` para
otimizar a política de combos. O env usa exatamente as mesmas interfaces, então
o que é treinado roda igual no robô.

Princípios (espelham os valores de engenharia do projeto):
- Dependency Inversion: tudo acima da HAL depende de interface, nunca de
  `MujocoPlant` ou `mujoco` diretamente.
- Single Responsibility por módulo.
- Reliability first: falha previsível e segura > otimismo.
- Measure, don't guess: profiling e dados reais guiam otimização.
- Document the why: o código mostra o "how", comentários explicam o "why".

---

## 4. Contrato da HAL (a espinha do projeto)

Defina como `typing.Protocol` em `src/robot_twin/hal/interfaces.py`. Toda planta
(sim e real) tem que passar pelos mesmos contract tests.

```python
from typing import Protocol
from robot_twin.core.types import Vec3, JointState, TraineePose, StrikeCommand

class IRobotPlant(Protocol):
    def command_strike(self, cmd: StrikeCommand) -> None:
        """Comanda um golpe alto nível: alvo, velocidade, telegraph."""
        ...
    def read_joint_state(self) -> JointState:
        """Pos, vel e força por junta. Força vem da deflexão da mola SEA."""
        ...
    def step(self, dt: float) -> None:
        """Avança a planta (no real é no-op/sincronização)."""
        ...
    def emergency_stop(self) -> None: ...

class ITraineeObserver(Protocol):
    def get_pose(self) -> TraineePose:
        """Keypoints do aluno. Ground truth no sim, pose da câmera no real."""
        ...
    def latency_s(self) -> float:
        """Latência observada do pipeline. Alimenta o R_keepout."""
        ...
```

Regra dura: Domain, Safety e Services NUNCA importam `mujoco`, `jax`, `cv2` nem
qualquer detalhe de planta. Só importam de `hal.interfaces` e `core.types`.

---

## 5. Stack tecnológico

- **Linguagem:** Python 3.11+. Tudo neste repo é Python. (O firmware STM32 em
  C/C++ é outro repo, futuro.)
- **Física/sim:** `mujoco` (engine + bindings). Modelo em **MJCF (XML)**. Mola
  SEA via `tendon`/`spring`/joint stiffness nativos.
- **RL escalado:** **MJX** (MuJoCo em JAX) para milhares de envs em paralelo.
  Ver nota de ambiente na seção 9 (MJX com GPU pede WSL2/Linux).
- **API RL:** **Gymnasium**. Começar com **Stable-Baselines3** (PPO/SAC) para
  resultado rápido, ou **CleanRL** se quiser código legível e hackável. Migrar
  para PPO em JAX quando escalar com MJX.
- **Mineração de vídeo:** pipeline de pose 2D + lifter 2D->3D (MotionBERT ou
  VideoPose3D) para extrair eventos de golpe e timing/cadência dos combos.
- **Config:** dataclasses + pydantic + YAML, env-driven.
- **Telemetria/visualização:** logging estruturado + **rerun.io** (visualizar
  poses, keep-out volume e trajetórias no tempo). TensorBoard ou W&B para runs.
- **Testes:** pytest + harness de fault injection.
- **Repro:** uv (preferido) ou poetry; Docker para o ambiente de treino Linux.

---

## 6. Estrutura do repositório

```
boxing-robot-twin/
├── CLAUDE.md
├── pyproject.toml
├── README.md
├── src/robot_twin/
│   ├── core/
│   │   ├── types.py          # Vec3, JointState, TraineePose, StrikeCommand
│   │   └── result.py         # Result[T] / ErrorCode (sem exceptions no hot path)
│   ├── hal/
│   │   ├── interfaces.py     # IRobotPlant, ITraineeObserver (Protocols)
│   │   ├── mujoco_plant.py
│   │   ├── sim_observer.py
│   │   └── real_plant.py     # stub STM32/UART, levanta NotImplementedError
│   ├── safety/
│   │   ├── arbiter.py        # SafetyArbiter: keep-out, reach, force cap
│   │   └── fault_injection.py
│   ├── domain/
│   │   ├── strike_planner.py
│   │   ├── target_selector.py
│   │   ├── dodge_detector.py
│   │   └── guard.py
│   ├── services/
│   │   ├── drill_engine.py   # FSM / currículo de dificuldade
│   │   ├── scoring.py
│   │   └── telemetry.py
│   ├── rl/
│   │   ├── env.py            # Gymnasium env (envolve plant + observer)
│   │   ├── reward.py
│   │   └── train.py
│   ├── video_mining/
│   │   ├── pose_extract.py
│   │   ├── strike_events.py
│   │   └── combo_stats.py
│   └── config/app_config.py
├── models/
│   ├── robot/striker_arm.xml # braço SEA
│   ├── trainee/humanoid.xml
│   └── scene.xml
├── tests/
│   ├── test_hal_contract.py  # ambas as plantas passam o mesmo contrato
│   ├── test_safety_arbiter.py
│   └── test_keepout.py
├── scripts/
│   ├── run_sim.py
│   └── viewer.py
└── data/
    ├── fight_clips/
    └── combo_distributions/
```

---

## 7. Padrões de código Python (seguir rigorosamente)

- **Type hints sempre**, inclusive retornos. `numpy.typing` para arrays.
- **Docstrings Google/NumPy** com Args/Returns/Raises. Foco no "why".
- **dataclasses** (ou pydantic) para estruturas; valide invariantes em
  `__post_init__`.
- **Enums** para estados, modos e configs. Nada de magic numbers ou `#define`
  improvisado.
- **Protocols/ABCs** para interfaces; o resto programa contra a abstração.
- **pathlib**, nunca `os.path`.
- **Logging estruturado**, nunca `print` em código de produção.
- **Vetorização NumPy/JAX**, evitar loops Python em hot path. Preallocar buffers.
- **Result[T] / ErrorCode no hot path de controle e segurança;** exceptions só
  em init/setup. Reflete o padrão exception-free do firmware.
- **constexpr-style:** o que dá para computar uma vez (LUTs, configs derivadas)
  computa fora do loop.
- **Sem travessões longos no texto e nos comentários.** Use hífen ou dois-pontos.
- Profiling antes de otimizar. `time.perf_counter` + decorator de timing nos
  caminhos quentes.

Red flags para rejeitar em PR: "confia que funciona" sem teste; otimização sem
profiling; planta importada acima da HAL; magic number sem justificativa; edge
case e failure mode ignorados.

---

## 8. Roadmap por fases (com validation gates)

Não pule gates. Cada fase só fecha quando o critério é medido, não suposto.

- **Fase 0 (sim) - Fundações + segurança.** Scaffold do repo, `core.types`, HAL
  com `MujocoPlant` + `SimGTObserver`, `SafetyArbiter` com a matemática de
  keep-out, env Gymnasium mínimo rodando. *Gate: SafetyArbiter rejeita 100% dos
  comandos que violam keep-out/reach/force SOB fault injection (latência alta,
  dropout de keypoint, aluno entrando no golpe).*
- **Fase 1 - Percepção e detecção (sim).** DodgeDetector e GuardDetector sobre o
  observer. *Gate: detecção de esquiva/guarda confiável com pose ruidosa
  injetada.*
- **Fase 2 - Loop fechado lento.** Um braço, golpes telegrafados lentos mirando
  perto do alvo, scoring funcionando. *Gate: aluno simulado consegue treinar o
  drill ponta a ponta; nenhuma violação de segurança em N episódios.*
- **Fase 3 - Multi-striker, combos e RL.** DrillEngine com currículo
  (telegraph time, velocidade, randomização, fintas). Política de combos
  treinada em MJX, semeada pela gramática minerada de vídeo. *Gate: política
  respeita o SafetyArbiter por construção; combos batem as distribuições de
  timing do vídeo.*

Fora do MVP: lower-body (chutes/joelhadas), mobilidade. Não implemente.

---

## 9. Ambiente e setup (Windows + WSL2)

MuJoCo já baixado em: `E:\Downloads\mujoco-3.10.0-windows-x86_64`

Nota importante: para o trabalho Python você NÃO depende desse zip. O pacote
`pip install mujoco` traz a engine 3.x e os bindings. O conteúdo baixado serve
para o viewer interativo standalone e headers C, úteis para inspecionar o MJCF
visualmente.

**Inspecionar um modelo MJCF no viewer nativo:**
```
E:\Downloads\mujoco-3.10.0-windows-x86_64\bin\simulate.exe models\scene.xml
```

**Dev nativo Windows (sim de 1 env, controle, segurança, testes):**
```
uv venv && .venv\Scripts\activate
uv pip install mujoco gymnasium stable-baselines3 numpy pydantic rerun-sdk pytest
```

**MJX para RL paralelo em GPU:** JAX com GPU não roda em Windows nativo (CPU
apenas). Para treino escalado use **WSL2 (Ubuntu) ou uma máquina Linux** com JAX
CUDA + `mujoco-mjx`. Desenvolva a lógica no Windows, treine pesado no WSL2/Linux.
Confirme o estado atual de JAX/Windows e das versões MuJoCo/MJX antes de fixar,
o ecossistema muda rápido.

**Comandos do projeto:**
```
python scripts/run_sim.py            # roda o twin com viewer
python scripts/viewer.py models/scene.xml
pytest                                # testes + contract + fault injection
python -m robot_twin.rl.train         # treino RL da política de combos
```

---

## 10. Sobre o uso de vídeos de luta (não confundir os papéis)

Vídeo treina a POLÍTICA OFENSIVA do robô (o que golpear e quando), nunca a
percepção do aluno (essa precisa da câmera ao vivo no robô real).

- **Transfere de vídeo:** estrutura alto nível. Gramática de combos
  (jab->cross->hook), cadência, distância de disparo, timing de telegraph,
  fintas. Isso semeia a FSM do DrillEngine e vira referência de reward no RL.
- **NÃO transfere:** trajetória de junta. O lutador tem tronco, quadril e
  footwork; o robô é frame fixo com braços SEA de 1-2 DOF. Mismatch de
  morfologia. A trajetória articular é aprendida no sim, não copiada do vídeo.

Comece com FSM/behavior tree autoral alimentada por distribuições mineradas
(determinística, debugável, argumentável em segurança). RL/imitation é
otimização de fase posterior.

---

## 11. Primeira tarefa sugerida

Scaffoldar a Fase 0: criar a árvore de pastas, `core/types.py` e `core/result.py`,
`hal/interfaces.py` com os Protocols, `MujocoPlant` + `SimGTObserver` mínimos, um
MJCF de cena com um braço SEA de teste, o `SafetyArbiter` com o cálculo de
`R_keepout`, um env Gymnasium mínimo, e os testes de contrato + fault injection.
Pare e confirme o design do MJCF do braço SEA antes de detalhar geometria.
