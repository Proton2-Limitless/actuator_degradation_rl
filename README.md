# Robust Actuator Degradation Benchmarking: PID vs. Adaptive vs. RL

This repository contains the codebase and scientific benchmarks for evaluating classical, adaptive, and Deep Reinforcement Learning (DRL) controllers under dynamic actuator degradation, sensor noise, and voltage saturation constraints.

### Core Results (Step Profile at severity eta=0.3)
- **PID:** 0.1200 rad
- **Adaptive Gain:** 0.1089 rad
- **Standard PPO (Reactive):** 0.0807 rad
- **Stacked PPO (With Memory):** **0.0580 rad** (28% tracking improvement)
