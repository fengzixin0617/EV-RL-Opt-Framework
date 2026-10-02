# EV-RL-Opt-Framewor

More details of the model will be added soon. 

**An Integrated Simulation–Optimisation Framework for Future Public Electric Vehicle Charger Deployment**

<p align="center">
  <img width="272" height="330" alt="Framework overview" src="https://github.com/user-attachments/assets/e7ce5c64-0076-4c54-a768-3b36aa894dc8" />
</p>

## Overview

Despite the rapid expansion of public electric vehicle (EV) charging infrastructure, charging networks remain far less mature than conventional refuelling systems. Existing studies on public charger optimisation typically overlook EV drivers' **post-deployment behaviour change**.

This repository implements an integrated, **bi-directional simulation–optimisation framework** that optimises future public charger locations and evaluates how drivers respond to newly deployed chargers:

1. **Simulate** – An agent-based reinforcement learning (RL) model simulates the driving and charging behaviour of EV drivers.
2. **Optimise** – A spatial optimisation model allocates new public chargers based on the simulated demand.
3. **Re-evaluate** – The RL model is re-run to assess how drivers adapt their behaviour to the expanded charging network.

## Components

| Component | Description | Availability |
|---|---|---|
| **Improved FRLM** | An enhanced Flow Refuelling Location Model for charger siting | The FRLM is now publicly available in [PySAL / libpysal](https://pysal.org/libpysal/stable/) |
| **RL Model** | Agent-based reinforcement learning model of EV driving and charging behaviour | For full details of the RL model, see: [published paper](https://journals.sagepub.com/doi/full/10.1177/23998083261455937) |



