from typing import Any, Dict

import numpy as np
from envs.custom_environments.multiagent_particle_envs.multiagent.environment import (
    MultiAgentEnv,
)

from .base import BaseEnvironment, ResetReturn, StepReturn


class MPEOMAR(BaseEnvironment):
    """MPE Environment wrapper for OMAR offline datasets."""

    def __init__(self, scenario, seed=None):
        # load scenario from script
        if scenario == "simple_spread":
            from envs.custom_environments.multiagent_particle_envs.multiagent.simple_spread import Scenario
        elif scenario == "simple_tag":
            from envs.custom_environments.multiagent_particle_envs.multiagent.scenarios.simple_tag import Scenario
        elif scenario == "simple_world":
            from envs.custom_environments.multiagent_particle_envs.multiagent.scenarios.simple_world import Scenario
        else:
            raise ValueError(f"Unknown scenario: {scenario}")

        scenario_obj = Scenario()
        # create world
        world = scenario_obj.make_world()

        env = MultiAgentEnv(world, scenario_obj.reset_world, scenario_obj.reward, scenario_obj.observation)

        self.environment = env

        self.agents = [f"agent_{n}" for n in range(len(world.agents))]
        self.num_actions = 2
        self.num_agents = len(self.agents)

        self.max_episode_length = 25

        self.t = 0

        # Determine the maximum observation dimension across all agents
        obs = self.environment.reset()
        self.obs_dim = max(o.shape[0] for o in obs)

    def reset(self) -> ResetReturn:
        obs = self.environment.reset()

        observations = {}
        for i, agent in enumerate(self.agents):
            o = obs[i].astype("float32")
            if o.shape[0] < self.obs_dim:
                o = np.concatenate([o, np.zeros(self.obs_dim - o.shape[0], dtype=np.float32)])
            observations[agent] = o

        self.t = 0

        return observations, {}

    def step(self, actions: Dict[str, np.ndarray]) -> StepReturn:
        mpe_actions = []
        for agent in self.agents:
            mpe_actions.append(actions[agent])

        next_observation, reward, done, info = self.environment.step(mpe_actions)

        terminals = {agent: done[i] for i, agent in enumerate(self.agents)}
        trunctations = {agent: False for i, agent in enumerate(self.agents)}

        rewards = {agent: reward[i] for i, agent in enumerate(self.agents)}

        observations = {}
        for i, agent in enumerate(self.agents):
            o = next_observation[i].astype("float32")
            if o.shape[0] < self.obs_dim:
                o = np.concatenate([o, np.zeros(self.obs_dim - o.shape[0], dtype=np.float32)])
            observations[agent] = o

        if self.t == 25:
            terminals = {agent: True for i, agent in enumerate(self.agents)}

        self.t += 1

        return observations, rewards, terminals, trunctations, info  # type: ignore

    def __getattr__(self, name: str) -> Any:
        """Expose any other attributes of the underlying environment."""
        if hasattr(self.__class__, name):
            return self.__getattribute__(name)
        else:
            return getattr(self.environment, name)
