from typing import Any, Mapping, Optional, Tuple, Union

import copy
import itertools
import gym
import gymnasium

import torch
import torch.nn as nn
import torch.nn.functional as F

from skrl import config, logger
from skrl.agents.torch import Agent
from skrl.memories.torch import Memory
from skrl.models.torch import Model
from skrl.resources.schedulers.torch import KLAdaptiveLR

# [start-config-dict-torch]
PPO_DEFAULT_CONFIG = {
    "rollouts": 16,                 # number of rollouts before updating
    "learning_epochs": 8,           # number of learning epochs during each update
    "mini_batches": 2,              # number of mini batches during each learning epoch

    "discount_factor": 0.99,        # discount factor (gamma)
    "lambda": 0.95,                 # TD(lambda) coefficient (lam) for computing returns and advantages

    "learning_rate": 1e-3,                  # learning rate
    "learning_rate_scheduler": None,        # learning rate scheduler class (see torch.optim.lr_scheduler)
    "learning_rate_scheduler_kwargs": {},   # learning rate scheduler's kwargs (e.g. {"step_size": 1e-3})

    "state_preprocessor": None,             # state preprocessor class (see skrl.resources.preprocessors)
    "state_preprocessor_kwargs": {},        # state preprocessor's kwargs (e.g. {"size": env.observation_space})
    "value_preprocessor": None,             # value preprocessor class (see skrl.resources.preprocessors)
    "value_preprocessor_kwargs": {},        # value preprocessor's kwargs (e.g. {"size": 1})

    "random_timesteps": 0,          # random exploration steps
    "learning_starts": 0,           # learning starts after this many steps

    "grad_norm_clip": 0.5,              # clipping coefficient for the norm of the gradients
    "ratio_clip": 0.2,                  # clipping coefficient for computing the clipped surrogate objective
    "value_clip": 0.2,                  # clipping coefficient for computing the value loss (if clip_predicted_values is True)
    "clip_predicted_values": False,     # clip predicted values during value loss computation

    "entropy_loss_scale": 0.0,      # entropy loss scaling factor
    "value_loss_scale": 1.0,        # value loss scaling factor

    "kl_threshold": 0,              # KL divergence threshold for early stopping

    "rewards_shaper": None,         # rewards shaping function: Callable(reward, timestep, timesteps) -> reward
    "time_limit_bootstrap": False,  # bootstrap at timeout termination (episode truncation)

    "experiment": {
        "directory": "",            # experiment's parent directory
        "experiment_name": "",      # experiment name
        "write_interval": "auto",   # TensorBoard writing interval (timesteps)

        "checkpoint_interval": "auto",      # interval for checkpoints (timesteps)
        "store_separately": False,          # whether to store checkpoints separately

        "wandb": False,             # whether to use Weights & Biases
        "wandb_kwargs": {}          # wandb kwargs (see https://docs.wandb.ai/ref/python/init)
    }
}
# [end-config-dict-torch]


class PPO(Agent):
    def __init__(self,
                 models: Mapping[str, Model],
                 memory: Optional[Union[Memory, Tuple[Memory]]] = None,
                 observation_space: Optional[Union[int, Tuple[int], gym.Space, gymnasium.Space]] = None,
                 action_space: Optional[Union[int, Tuple[int], gym.Space, gymnasium.Space]] = None,
                 device: Optional[Union[str, torch.device]] = None,
                 cfg: Optional[dict] = None) -> None:
        """Proximal Policy Optimization (PPO)
        https://arxiv.org/abs/1707.06347
        """
        _cfg = copy.deepcopy(PPO_DEFAULT_CONFIG)
        _cfg.update(cfg if cfg is not None else {})
        super().__init__(models=models,
                         memory=memory,
                         observation_space=observation_space,
                         action_space=action_space,
                         device=device,
                         cfg=_cfg)

        # Primary policy and value
        self.policy = self.models.get("policy", None)
        self.value = self.models.get("value", None)

        # Secondary policy (policy2) added
        self.policy2 = self.models.get("policy2", None)

        # checkpoint models
        self.checkpoint_modules["policy"] = self.policy
        self.checkpoint_modules["value"] = self.value
        if self.policy2 is not None:
            self.checkpoint_modules["policy2"] = self.policy2

        # broadcast parameters in distributed runs (if any)
        if config.torch.is_distributed:
            logger.info(f"Broadcasting models' parameters")
            if self.policy is not None:
                self.policy.broadcast_parameters()
                if self.value is not None and self.policy is not self.value:
                    self.value.broadcast_parameters()
            if self.policy2 is not None:
                self.policy2.broadcast_parameters()

        # configuration
        self._learning_epochs = self.cfg["learning_epochs"]
        self._mini_batches = self.cfg["mini_batches"]
        self._rollouts = self.cfg["rollouts"]
        self._rollout = 0

        self._grad_norm_clip = self.cfg["grad_norm_clip"]
        self._ratio_clip = self.cfg["ratio_clip"]
        self._value_clip = self.cfg["value_clip"]
        self._clip_predicted_values = self.cfg["clip_predicted_values"]

        self._value_loss_scale = self.cfg["value_loss_scale"]
        self._entropy_loss_scale = self.cfg["entropy_loss_scale"]

        self._kl_threshold = self.cfg["kl_threshold"]

        self._learning_rate = self.cfg["learning_rate"]
        self._learning_rate_scheduler = self.cfg["learning_rate_scheduler"]

        self._state_preprocessor = self.cfg["state_preprocessor"]
        self._value_preprocessor = self.cfg["value_preprocessor"]

        self._discount_factor = self.cfg["discount_factor"]
        self._lambda = self.cfg["lambda"]

        self._random_timesteps = self.cfg["random_timesteps"]
        self._learning_starts = self.cfg["learning_starts"]

        self._rewards_shaper = self.cfg["rewards_shaper"]
        self._time_limit_bootstrap = self.cfg["time_limit_bootstrap"]

        # set up optimizer for primary policy/value
        if self.policy is not None and self.value is not None:
            if self.policy is self.value:
                self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self._learning_rate)
            else:
                self.optimizer = torch.optim.Adam(itertools.chain(self.policy.parameters(), self.value.parameters()),
                                                  lr=self._learning_rate)
            if self._learning_rate_scheduler is not None:
                self.scheduler = self._learning_rate_scheduler(self.optimizer, **self.cfg["learning_rate_scheduler_kwargs"])

            self.checkpoint_modules["optimizer"] = self.optimizer

        # set up optimizer for secondary policy (policy2)
        if self.policy2 is not None:
            # policy2 不共享 value2 (沒有 value2)，仍使用原本的 value
            # 因此這裡只對 policy2 的參數進行優化
            self.optimizer2 = torch.optim.Adam(self.policy2.parameters(), lr=self._learning_rate)
            if self._learning_rate_scheduler is not None:
                self.scheduler2 = self._learning_rate_scheduler(self.optimizer2, **self.cfg["learning_rate_scheduler_kwargs"])
            self.checkpoint_modules["optimizer2"] = self.optimizer2

        # set up preprocessors
        if self._state_preprocessor:
            self._state_preprocessor = self._state_preprocessor(**self.cfg["state_preprocessor_kwargs"])
            self.checkpoint_modules["state_preprocessor"] = self._state_preprocessor
        else:
            self._state_preprocessor = self._empty_preprocessor

        if self._value_preprocessor:
            self._value_preprocessor = self._value_preprocessor(**self.cfg["value_preprocessor_kwargs"])
            self.checkpoint_modules["value_preprocessor"] = self._value_preprocessor
        else:
            self._value_preprocessor = self._empty_preprocessor

    def init(self, trainer_cfg: Optional[Mapping[str, Any]] = None) -> None:
        super().init(trainer_cfg=trainer_cfg)
        self.set_mode("eval")

        # create tensors in memory
        if self.memory is not None:
            self.memory.create_tensor(name="states", size=self.observation_space, dtype=torch.float32)
            self.memory.create_tensor(name="actions", size=self.action_space, dtype=torch.float32)
            self.memory.create_tensor(name="rewards", size=1, dtype=torch.float32)
            self.memory.create_tensor(name="terminated", size=1, dtype=torch.bool)
            self.memory.create_tensor(name="log_prob", size=1, dtype=torch.float32)
            self.memory.create_tensor(name="values", size=1, dtype=torch.float32)
            self.memory.create_tensor(name="returns", size=1, dtype=torch.float32)
            self.memory.create_tensor(name="advantages", size=1, dtype=torch.float32)
            # 新增 policy_id tensor，用來記錄是哪個 policy 被使用
            self.memory.create_tensor(name="policy_id", size=1, dtype=torch.int)

            # 將 policy_id 也加入 sample 時取出的 tensor 清單中
            self._tensors_names = ["states", "actions", "log_prob", "values", "returns", "advantages", "policy_id"]

        # create temporary variables
        self._current_log_prob = None
        self._current_next_states = None

    def act(self, states: torch.Tensor, timestep: int, timesteps: int) -> torch.Tensor:
        if timestep < self._random_timesteps:
            return self.policy.random_act({"states": self._state_preprocessor(states)}, role="policy")

        state_proc = self._state_preprocessor(states)
        
        actions1, log_prob1, outputs1 = self.policy.act({"states": state_proc}, role="policy")
        actions2, log_prob2, outputs2 = self.policy2.act({"states": state_proc}, role="policy")
        
        # 取每個 policy 的 log_prob 平均值
        p1 = log_prob1.mean().detach().cpu().item()
        p2 = log_prob2.mean().detach().cpu().item()
        
        # 因為 log_prob 通常為負數，取它們的相對大小時可以先做轉換
        # 這裡使用 softmax 將兩個 score 轉換成機率（若數值較大，會更傾向該 policy）
        scores = torch.tensor([p1, p2])
        probs = torch.softmax(scores/5, dim=0)
        # 隨機採樣：若兩個策略類似，則概率接近 0.5；否則較優的策略有更大機率被選中
        choice = torch.multinomial(probs, 1).item()
        
        if choice == 0:
            self._current_log_prob = log_prob1
            self._current_policy_id = torch.zeros_like(log_prob1, dtype=torch.int)  # 設為 0，代表 policy1
            return actions1, log_prob1, outputs1
        else:
            self._current_log_prob = log_prob2
            self._current_policy_id = torch.ones_like(log_prob2, dtype=torch.int)   # 設為 1，代表 policy2
            return actions2, log_prob2, outputs2


    def record_transition(self,
                          states: torch.Tensor,
                          actions: torch.Tensor,
                          rewards: torch.Tensor,
                          next_states: torch.Tensor,
                          terminated: torch.Tensor,
                          truncated: torch.Tensor,
                          infos: Any,
                          timestep: int,
                          timesteps: int) -> None:
        super().record_transition(states, actions, rewards, next_states, terminated, truncated, infos, timestep, timesteps)

        if self.memory is not None:
            self._current_next_states = next_states

            # reward shaping
            if self._rewards_shaper is not None:
                rewards = self._rewards_shaper(rewards, timestep, timesteps)

            # compute values from primary value network
            values, _, _ = self.value.act({"states": self._state_preprocessor(states)}, role="value")
            values = self._value_preprocessor(values, inverse=True)

            # time-limit (truncation) bootstrapping
            if self._time_limit_bootstrap:
                rewards += self._discount_factor * values * truncated

            # store transition in memory (加入 policy_id)
            self.memory.add_samples(states=states, actions=actions, rewards=rewards, next_states=next_states,
                                    terminated=terminated, truncated=truncated, log_prob=self._current_log_prob, values=values,
                                    policy_id=self._current_policy_id)
            for memory in self.secondary_memories:
                memory.add_samples(states=states, actions=actions, rewards=rewards, next_states=next_states,
                                   terminated=terminated, truncated=truncated, log_prob=self._current_log_prob, values=values,
                                   policy_id=self._current_policy_id)

    def pre_interaction(self, timestep: int, timesteps: int) -> None:
        pass

    def post_interaction(self, timestep: int, timesteps: int) -> None:
        self._rollout += 1
        if not self._rollout % self._rollouts and timestep >= self._learning_starts:
            self.set_mode("train")
            self._update(timestep, timesteps)
            self.set_mode("eval")

        super().post_interaction(timestep, timesteps)

    def _update(self, timestep: int, timesteps: int) -> None:
        def compute_gae(rewards: torch.Tensor,
                        dones: torch.Tensor,
                        values: torch.Tensor,
                        next_values: torch.Tensor,
                        discount_factor: float = 0.99,
                        lambda_coefficient: float = 0.95) -> Tuple[torch.Tensor, torch.Tensor]:
            advantage = 0
            advantages = torch.zeros_like(rewards)
            not_dones = dones.logical_not()
            memory_size = rewards.shape[0]

            for i in reversed(range(memory_size)):
                nxt_val = values[i + 1] if i < memory_size - 1 else last_values
                advantage = rewards[i] - values[i] + discount_factor * not_dones[i] * (nxt_val + lambda_coefficient * advantage)
                advantages[i] = advantage
            returns = advantages + values
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            return returns, advantages

        with torch.no_grad():
            self.value.train(False)
            last_values, _, _ = self.value.act({"states": self._state_preprocessor(self._current_next_states.float())}, role="value")
            self.value.train(True)
        last_values = self._value_preprocessor(last_values, inverse=True)

        values = self.memory.get_tensor_by_name("values")
        returns, advantages = compute_gae(rewards=self.memory.get_tensor_by_name("rewards"),
                                          dones=self.memory.get_tensor_by_name("terminated"),
                                          values=values,
                                          next_values=last_values,
                                          discount_factor=self._discount_factor,
                                          lambda_coefficient=self._lambda)

        self.memory.set_tensor_by_name("values", self._value_preprocessor(values, train=True))
        self.memory.set_tensor_by_name("returns", self._value_preprocessor(returns, train=True))
        self.memory.set_tensor_by_name("advantages", advantages)

        # 使用 sample_all 時會同時取出 policy_id (在最後一個位置)
        sampled_batches = self.memory.sample_all(names=self._tensors_names, mini_batches=self._mini_batches)

        cumulative_policy_loss = 0
        cumulative_policy2_loss = 0
        cumulative_value_loss = 0
        cumulative_entropy_loss = 0

        for epoch in range(self._learning_epochs):
            kl_divergences = []
            kl_divergences2 = []
            for batch in sampled_batches:
                # 解包 mini-batch（依據 _tensors_names 順序）
                sampled_states, sampled_actions, sampled_log_prob, sampled_values, sampled_returns, sampled_advantages, sampled_policy_ids = batch

                # 使用 state preprocessor
                sampled_states = self._state_preprocessor(sampled_states, train=not epoch)
                # 將 policy_id squeeze 成一維
                sampled_policy_ids = sampled_policy_ids.squeeze(1)

                # --- 更新 policy (policy1) 只處理 policy_id == 0 的 samples ---
                mask1 = (sampled_policy_ids == 0)
                if mask1.any():
                    s1 = sampled_states[mask1]
                    a1 = sampled_actions[mask1]
                    lp1 = sampled_log_prob[mask1]
                    adv1 = sampled_advantages[mask1]
                    _, next_log_prob, _ = self.policy.act({"states": s1, "taken_actions": a1}, role="policy")
                    ratio = torch.exp(next_log_prob - lp1)
                    surrogate = adv1 * ratio
                    surrogate_clipped = adv1 * torch.clip(ratio, 1.0 - self._ratio_clip, 1.0 + self._ratio_clip)
                    policy_loss = -torch.min(surrogate, surrogate_clipped).mean()

                    self.optimizer.zero_grad()
                    policy_loss.backward()
                    if self._grad_norm_clip > 0:
                        nn.utils.clip_grad_norm_(self.policy.parameters(), self._grad_norm_clip)
                    self.optimizer.step()
                    cumulative_policy_loss += policy_loss.item()

                    kl_div = ((torch.exp(next_log_prob - lp1) - 1) - (next_log_prob - lp1)).mean()
                    kl_divergences.append(kl_div.item())

                # --- 更新 policy2 (secondary) 只處理 policy_id == 1 的 samples ---
                mask2 = (sampled_policy_ids == 1)
                if mask2.any():
                    s2 = sampled_states[mask2]
                    a2 = sampled_actions[mask2]
                    lp2 = sampled_log_prob[mask2]
                    adv2 = sampled_advantages[mask2]
                    _, next_log_prob2, _ = self.policy2.act({"states": s2, "taken_actions": a2}, role="policy")
                    ratio2 = torch.exp(next_log_prob2 - lp2)
                    surrogate2 = adv2 * ratio2
                    surrogate_clipped2 = adv2 * torch.clip(ratio2, 1.0 - self._ratio_clip, 1.0 + self._ratio_clip)
                    policy_loss2 = -torch.min(surrogate2, surrogate_clipped2).mean()

                    self.optimizer2.zero_grad()
                    policy_loss2.backward()
                    if self._grad_norm_clip > 0:
                        nn.utils.clip_grad_norm_(self.policy2.parameters(), self._grad_norm_clip)
                    self.optimizer2.step()
                    cumulative_policy2_loss += policy_loss2.item()

                    kl_div2 = ((torch.exp(next_log_prob2 - lp2) - 1) - (next_log_prob2 - lp2)).mean()
                    kl_divergences2.append(kl_div2.item())

                # --- 更新 Value 網路 (共用) ---
                predicted_values, _, _ = self.value.act({"states": sampled_states}, role="value")
                if self._clip_predicted_values:
                    predicted_values = sampled_values + torch.clip(predicted_values - sampled_values,
                                                                    min=-self._value_clip,
                                                                    max=self._value_clip)
                value_loss = self._value_loss_scale * F.mse_loss(sampled_returns, predicted_values)

                self.optimizer.zero_grad()
                value_loss.backward()
                if self._grad_norm_clip > 0:
                    nn.utils.clip_grad_norm_(self.value.parameters(), self._grad_norm_clip)
                self.optimizer.step()
                cumulative_value_loss += value_loss.item()

            # KL-based learning rate update for both policies
            if self._learning_rate_scheduler:
                if isinstance(self.scheduler, KLAdaptiveLR) and kl_divergences:
                    self.scheduler.step(torch.tensor(kl_divergences, device=self.device).mean().item())
                if isinstance(self.scheduler2, KLAdaptiveLR) and kl_divergences2:
                    self.scheduler2.step(torch.tensor(kl_divergences2, device=self.device).mean().item())

        # Record training metrics
        self.track_data("Loss / Policy loss", cumulative_policy_loss / (self._learning_epochs * self._mini_batches))
        self.track_data("Loss / Policy2 loss", cumulative_policy2_loss / (self._learning_epochs * self._mini_batches))
        self.track_data("Loss / Value loss", cumulative_value_loss / (self._learning_epochs * self._mini_batches))
        if self._entropy_loss_scale:
            self.track_data("Loss / Entropy loss", cumulative_entropy_loss / (self._learning_epochs * self._mini_batches))
        self.track_data("Policy / Standard deviation", self.policy.distribution(role="policy").stddev.mean().item())
        self.track_data("Policy2 / Standard deviation", self.policy2.distribution(role="policy").stddev.mean().item())
