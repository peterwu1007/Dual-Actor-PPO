
#dual ppo

import isaacgym

import argparse
import torch
import torch.nn as nn
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.envs.loaders.torch import load_isaacgym_env_preview4
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveRL
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed

# Argument parsing
parser = argparse.ArgumentParser(description="Run the RL training or testing with specific configurations.")
parser.add_argument('--headless', action='store_false', help='Run in headless mode')

parser.add_argument('--num_envs', type=int, default=1024, help='Number of environments')
parser.add_argument('--mini_batch_size', type=int, default= 4096, help='Mini-batch size')
parser.add_argument('--horizon_length', type=int, default=64, help='Horizon length')
parser.add_argument('--test', action='store_true', help='Perform testing')
parser.add_argument('--checkpoint', type=str, default='runs/torch/BallBalance/25-03-06_22-37-21-364528_PPO/checkpoints/best_agent.pt', help='Path to the checkpoint file')
args = parser.parse_args()

# Seed for reproducibility
set_seed()
mode = 'test'  # Change this to 'test' when you want to test
checkpoint_path = 'runs/torch/BallBalance/25-04-12_20-13-14-232168_PPO/checkpoints/best_agent.pt'

# define shared model (stochastic and deterministic models) using mixins
class Shared(GaussianMixin, DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False,
                 clip_log_std=True, min_log_std=-20, max_log_std=2, reduction="sum"):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)
        DeterministicMixin.__init__(self, clip_actions)

        self.net = nn.Sequential(nn.Linear(self.num_observations, 512),
                                 nn.ELU(),
                                 nn.Linear(512, 256),
                                 nn.ELU(),
                                 nn.Linear(256, 128),
                                 nn.ELU())

        self.mean_layer = nn.Linear(128, self.num_actions)
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

        self.value_layer = nn.Linear(128, 1)

    def act(self, inputs, role):
        if role == "policy":
            return GaussianMixin.act(self, inputs, role)
        elif role == "value":
            return DeterministicMixin.act(self, inputs, role)

    def compute(self, inputs, role):
        if role == "policy":
            return self.mean_layer(self.net(inputs["states"])), self.log_std_parameter, {}
        elif role == "value":
            return self.value_layer(self.net(inputs["states"])), {}
            

headless = False if mode == 'test' else True  
num_envs = 2 if mode == 'test' else 2048

# Load and wrap the Isaac Gym environment
env = load_isaacgym_env_preview4(task_name="BallBalance", headless=headless, num_envs=num_envs)
env = wrap_env(env)
device = env.device

# Setup for the agent
memory = RandomMemory(memory_size=args.horizon_length, num_envs=env.num_envs, device=device)
models = {
    "policy": Shared(env.observation_space, env.action_space, device),
    "value": Shared(env.observation_space, env.action_space, device),
    "policy2": Shared(env.observation_space, env.action_space, device),
}

# configure and instantiate the agent (visit its documentation to see all the options)
# https://skrl.readthedocs.io/en/latest/api/agents/ppo.html#configuration-and-hyperparameters
cfg = PPO_DEFAULT_CONFIG.copy()
cfg["rollouts"] = 64  # memory_size
cfg["learning_epochs"] = 10
cfg["mini_batches"] = 2  # 24 * 4096 / 32768
cfg["discount_factor"] = 0.99
cfg["lambda"] = 0.95
cfg["learning_rate"] = 1e-4 #3e-4
cfg["learning_rate_scheduler"] = KLAdaptiveRL
cfg["learning_rate_scheduler_kwargs"] = {"kl_threshold": 0.008}
cfg["random_timesteps"] = 0
cfg["learning_starts"] = 0
cfg["grad_norm_clip"] = 1.0
cfg["ratio_clip"] = 0.2
cfg["value_clip"] = 0.2
cfg["clip_predicted_values"] = True
cfg["entropy_loss_scale"] = 0.0
cfg["value_loss_scale"] = 1.0
cfg["kl_threshold"] = 0.008
cfg["rewards_shaper"] = None
cfg["state_preprocessor"] = RunningStandardScaler
cfg["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": device}
cfg["value_preprocessor"] = RunningStandardScaler
cfg["value_preprocessor_kwargs"] = {"size": 1, "device": device}

# logging to TensorBoard and write checkpoints (in timesteps)
cfg["experiment"]["write_interval"] = 20
cfg["experiment"]["checkpoint_interval"] = 200
cfg["experiment"]["directory"] = "runs/torch/BallBalance"

agent = PPO(models=models,
            memory=memory,
            cfg=cfg,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device)

# Trainer configuration and instantiation
trainer_cfg = {"timesteps": 30000, "headless": headless}
trainer = SequentialTrainer(cfg=trainer_cfg, env=env, agents=agent)

# Conditional execution based on mode
if mode == 'test':
    agent.load(checkpoint_path)
    trainer.eval()
else:
    trainer.train()
