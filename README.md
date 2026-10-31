# Robot Actuator Degradation: PID vs Adaptive vs PPO

A small simulation study of what happens to three different controllers when an electric motor slowly loses strength.

> **Status:** baseline study. It is a first step toward a larger research direction (safe active diagnosis for robots with degrading actuators), not a finished framework. See [Limitations](#limitations).

---

## 1. The idea in plain words

Machines rarely fail all at once. Motors wear out gradually, so a robot may keep running while its actuators quietly get weaker. A controller that was tuned for a healthy motor can then become sluggish without anything "breaking".

This project asks a simple question:

> **When a motor's strength fades over time, how well do a standard controller (PID), an adaptive controller, and a reinforcement learning agent (PPO) keep tracking a target?**

All three controllers face **exactly the same** degradation, sensor noise, load and target, and none of them is told how weak the motor has become.

---

## 2. What was built

| Part | What it is |
|---|---|
| **Simulated motor** | A DC motor model written as a custom [Gymnasium](https://gymnasium.farama.org/) environment |
| **Degradation** | A factor `eta` that multiplies the motor torque (1.0 = healthy, 0.2 = badly degraded) |
| **Controller 1: PID** | A standard controller with fixed gains tuned for the healthy motor |
| **Controller 2: Adaptive** | The same PID, multiplied by a gain that adjusts itself while running |
| **Controller 3: PPO** | A reinforcement learning agent (Stable-Baselines3), trained by trial and error, 3 random seeds |
| **Evaluation** | A sweep from healthy to badly degraded, scoring tracking error, effort and voltage saturation |

---

## 3. The simulated world

### 3.1 The motor

A DC motor with inductance ignored (so current responds instantly to voltage):

```
J * dw/dt = eta(t) * KT * (V - KE*w) / R  -  B*w  -  tau_load
dtheta/dt = w
```

In words: voltage pushes current through the motor, current becomes torque, and **`eta` scales that torque**. Friction and the load take some of it away. What is left speeds the motor up.

| Parameter | Value | Meaning |
|---|---|---|
| `R` | 1.0 | Electrical resistance |
| `KT` | 0.1 | Torque per unit current |
| `KE` | 0.1 | Back-EMF constant (the motor opposes the supply as it spins faster) |
| `J` | 0.01 | Inertia |
| `B` | 0.02 | Viscous friction |
| `V_MAX` | 6.0 V | Supply limit (**voltage saturation**) |

### 3.2 The task

Track a **smoothed square wave** for 10 seconds (500 control steps of 0.02 s). The target swings between `+amp` and `-amp` (amplitude drawn randomly between 0.6 and 1.0 rad per episode). The fast swings demand a lot of torque, which is where a weakened motor struggles.

### 3.3 What the controller sees (5 numbers, the same for everyone)

1. Noisy tracking error
2. Integral of the error (clipped, which acts as anti-windup)
3. Noisy position
4. Noisy velocity (scaled by 1/10)
5. The previous action

**The true `eta` is never given to any controller.** They can only react to how the motor behaves.

### 3.4 Action and saturation

The action is a fraction of the supply voltage, in [-1, 1]. Anything beyond that is clipped. A step counts as **saturated** when the controller asks for 99% of the limit or more.

### 3.5 Disturbances

| Disturbance | Setting |
|---|---|
| Position sensor noise | std 0.005 rad |
| Velocity sensor noise | std 0.2 rad/s |
| Constant load torque | random in ±0.05 N·m per episode |
| Load jitter | std 0.005 N·m per step |

### 3.6 Degradation profiles

| Profile | How `eta` changes |
|---|---|
| `constant` | Stays at its final value for the whole run |
| `slow_linear` | Fades steadily from 1 to the final value over 10 s |
| `fast_linear` | Same fade, finished by 4 s |
| `exponential` | Drops quickly, then levels off |
| `step` | Healthy, then drops suddenly at a random time (3 to 7 s) |

`eta` is never allowed below 0.2.

### 3.7 Reward (used only to train PPO)

```
reward = -( error^2 + 0.01 * u^2 + 0.05 * saturated )
```

Large error is bad, using lots of voltage is slightly bad, and sitting at the voltage limit is penalised. The reward uses the true error. The controllers never see it directly.

---

## 4. The three controllers

### PID (the fixed baseline)

```
u = Kp * e + Ki * e_int - Kd * velocity
```

with `Kp = 5`, `Ki = 2`, `Kd = 1.5`. The gains are set for the healthy motor and **never change**. The derivative term uses measured velocity, which avoids differentiating noisy position.

### Adaptive controller (gain adaptation, MIT-rule style)

It is the same PID multiplied by an adjustable gain `g` (limited to 1 to 4). It also runs a small *reference model* describing how the output **should** move. If the real position lags the model, `g` increases; if the motor runs ahead, `g` decreases:

```
dg/dt = gamma * e_m * u_nominal / (1 + u_nominal^2)
```

It never sees `eta`. It only reacts to the mismatch between expected and actual motion. Because it is literally "PID times g", any difference from PID comes from the adaptation alone.

### PPO (reinforcement learning)

PPO learns a policy by trial and error: it sees the 5 numbers, picks a voltage, receives the reward, and gradually adjusts a neural network so that high-reward actions become more likely.

| Setting | Value |
|---|---|
| Network | MLP, 2 hidden layers of 64 units |
| Parallel environments | 4 |
| Steps per update (per env) | 512 |
| Batch size | 256 |
| Learning rate | 3e-4 |
| Discount (`gamma`) | 0.99 |
| Training length | `TOTAL_TIMESTEPS` per seed (300,000 for the full run, 30,000 to 50,000 for a quick test) |
| Random seeds | 0, 1, 2 (three separately trained agents) |

PPO is trained on the profiles `constant`, `slow_linear`, `fast_linear` and `exponential`, with final `eta` between **0.4 and 1.0**. During testing the deterministic (best) action is used.

---

## 5. Fairness design

- **Same measurements.** All three controllers receive the identical 5-number observation. None is given `eta`.
- **Same conditions.** Each test episode is generated from a seed. The degradation, target, load and noise sequence are drawn in a fixed order, so two controllers given the same seed face exactly the same test.
- **Fresh test data.** Test seeds (1000 and up) are different from anything PPO trained on.
- **Held-out conditions.** `eta` below 0.4 and the `step` profile are never seen by PPO during training. They test whether PPO copes with damage it has not practised.
- **Several PPO seeds.** RL results depend on luck, so three trained agents are reported, not one.

---

## 6. How controllers are scored

For each test episode:

| Metric | Meaning |
|---|---|
| **RMSE** (rad) | Typical tracking error, using the *true* error. Lower is better. This is the main metric. |
| **Effort** | Mean of `u^2` (applied voltage fraction squared). 0 is idle, 1 is flat out. |
| **Saturation** | Fraction of steps where the controller asked for the voltage limit or more. |
| **Performance retained** | `min(1, best healthy RMSE / RMSE at this eta)`. 100% means no loss compared with the best healthy controller. |

> Note: saturation counts the *requested* voltage (before clipping), while effort uses the *applied* voltage (after clipping).

---

## 7. Experiments

1. **Sanity checks.**
   - Open-loop test: the same fixed voltage is applied at different `eta`. Lower `eta` must give slower speed-up and a lower top speed.
   - Closed-loop test: PID and Adaptive run at `eta` = 1.0, 0.5, 0.3 to confirm that degradation really hurts tracking.
2. **Static strength sweep (key experiment).** `eta` is held fixed from 1.0 down to 0.2 in steps of 0.1. Every controller tracks the square wave at every level.
3. **Time-varying profiles (optional extra).** Slow, fast, exponential and step degradation, at final `eta` of 0.5 and 0.3.

---

## 8. Results

> **Fill this in after running the notebook. Only write what you actually observe.**

Reference healthy RMSE: **0.0258**

| eta | PID RMSE | Adaptive RMSE | PPO RMSE (mean, min to max over seeds) |
|---|---|---|---|
| 1.0 | 0.0620 | 0.0604 | 0.0457 (0.0328 to 0.0573) |
| 0.8 | 0.0697 | 0.0663 | 0.0494 (0.0347 to 0.0631) |
| 0.6 | 0.0864 | 0.0782 | 0.0594 (0.0416 to 0.0772) |
| 0.4 | 0.1256 | 0.1065 | 0.0827 (0.0560 to 0.1118) |
| 0.3 | 0.1700 | 0.1403 | 0.1187 (0.0784 to 0.1651) |
| 0.2 | 0.2664 | 0.2204 | 0.2049 (0.1261 to 0.2882) |

Plots are saved in `results/`:

- `sanity_open_loop.png`: same input, weaker motor, slower speed-up
- `training_curves.png`: PPO learning progress per seed
- `performance_retained.png`: RMSE, performance retained and saturation against `eta`
- `trace_step_profile.png`: one example episode with a sudden drop in strength

**Observations** (replace with your own after viewing the output):

- PID: ____
- Adaptive: ____
- PPO, inside the trained range (eta 0.4 to 1.0): ____
- PPO, below the trained range (eta below 0.4): ____

---

## 9. How to run

The notebook is designed for **Google Colab** (CPU is enough, no GPU needed).

1. Open the notebook in Colab.
2. Run the cells from top to bottom:
   1. Install and import (`gymnasium`, `stable-baselines3`)
   2. Settings
   3. Environment
   4. Controllers
   5. Episode runner and sanity checks
   6. Train PPO
   7. Evaluation
   8. Save and download
3. For a quick test first, set `TOTAL_TIMESTEPS = 30_000` and `N_EVAL_EPISODES = 2`, then switch back to full values.

Rough run time on Colab CPU: about 5 to 10 minutes per PPO seed at 300k steps.

All CSVs, plots and trained models are written to `results/` and zipped for download at the end.

### Settings you may want to change

| Setting | Where | Effect |
|---|---|---|
| `MOTOR`, `SIM`, `NOISE` | Settings cell | Physical parameters and noise levels |
| `TRAIN_PROFILES`, `TRAIN_ETA_RANGE` | Settings cell | What PPO practises on |
| `PPO_SEEDS`, `TOTAL_TIMESTEPS` | Settings cell | Number of agents and training length |
| `N_EVAL_EPISODES` | Settings cell | Number of test episodes per condition |
| `Kp`, `Ki`, `Kd`, `g_max`, `gamma` | Controller classes | PID and adaptive tuning |

---

## 10. Limitations

Please read these before drawing conclusions.

- **Simplified plant.** A single 1-D DC motor with inductance ignored. It is not a full robot.
- **One fault type.** Degradation is a multiplier on torque only. Real wear also changes back-EMF, friction, backlash and so on.
- **Possible physical floor.** At low `eta` the motor may not have enough power to follow the fast swings at all, so all controllers can look similarly poor there. That would reflect the physics, not a bug.
- **Adaptive limit.** The adaptive gain is capped at 4, so it cannot fully compensate for `eta` below roughly 0.25.
- **Tuning.** PID gains and adaptive-law settings were chosen by hand and are not optimised. Results depend on them.
- **PPO has no memory.** It sees one snapshot per step, so it has limited ability to infer hidden degradation. PID and Adaptive keep internal state (an integral term, an adaptive gain).
- **Limited statistics.** Three PPO seeds and a modest number of test episodes. Differences smaller than the spread across seeds should not be read as real.
- **Simulation only.** No hardware validation.
- **Not the full framework.** There is no actuator-effectiveness estimator, no constrained-control safety layer and no active diagnosis here.

---

## 11. Possible next steps

- Understand why each controller behaves as it does (PID tuning, MRAC theory, PPO basics).
- Give PPO a short history of observations (for example frame stacking) and test whether this helps, comparing carefully against the plain version.
- Add an estimator of actuator effectiveness.
- Add a constrained-control or shield layer (safe RL, CMDP).
- Test fault models that are not in the training set.
- Validate on real hardware (a motor-driven platform).

---

## 12. Repository contents

```
.
├── actuator_degradation.ipynb   # the full notebook
├── README.md                    # this file
└── results/                     # generated CSVs, plots, saved PPO models
```

## 13. Requirements

Python 3, NumPy, Pandas, Matplotlib, PyTorch, `gymnasium`, `stable-baselines3` (Colab already includes everything except the last two).