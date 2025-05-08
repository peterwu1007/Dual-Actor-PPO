#  Dual-Actor PPO for Ball Balancing in Isaac Gym (SKRL)

This project implements a customized Dual-Actor PPO agent for training a humanoid robot arm in NVIDIA Isaac Gym to balance a ball on a tray. It extends the SKRL PPO framework to include two distinct actor policies with log-probability-based soft switching during inference, and independent optimization during training.

---

##  Project Structure

```
dual_actor_rl/
├── dual_actor_ppo.py           # Custom PPO agent with dual-policy logic
├── torch_ball_balacing.py      # Main script for training and evaluation
├── README.md
```

---

## Algorithm: Dual-Actor PPO

The `dual_actor_ppo.py` file defines a modified PPO agent with:

- Two independent policy networks (`policy` and `policy2`)
- Softmax-based probabilistic selection between policies during `act`
- Memory tracking of selected `policy_id` per sample
- Independent policy updates using masks during training
- Shared value network and optimizer

**Selection Logic:**
- Calculates average log-probabilities from both policies
- Applies temperature-controlled softmax to choose one
- Stores selected policy ID for later supervised training

**Training Logic:**
- GAE (Generalized Advantage Estimation) for advantage calculation
- Policy1 and Policy2 update only their corresponding samples
- Shared value network updated with all samples

---

##  Model

**Model (in `torch_ball_balacing.py`):**
- Shared actor-critic model using mixin classes
- Policy head: Gaussian
- Value head: Deterministic
- Architecture: 512 → 256 → 128 (ELU activations)

---


##  How to Run

### Training:

```bash
python torch_ball_balacing.py
```

### Evaluation:

```bash
python torch_ball_balacing.py --test
```

---

##  Dependencies


- [skrl](https://skrl.readthedocs.io/) library



##  License

This project adapts the `skrl` PPO implementation with custom logic.  
Original PPO framework from [`skrl`](https://github.com/Toni-SM/skrl).

---

> Maintained by [@peterwu1007](https://github.com/peterwu1007)# Dual-Actor-PPO
