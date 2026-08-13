# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import yaml
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import torchvision.transforms as T
import ray

from agent_system.environments.env_package.alfworld.alfworld.agents.environment import get_environment

ALF_ACTION_LIST=["pass", "goto", "pick", "put", "open", "close", "toggle", "heat", "clean", "cool", "slice", "inventory", "examine", "look"]
# ALF_ITEM_LIST =

def load_config_file(path):
    assert os.path.exists(path), "Invalid config file"
    with open(path) as reader:
        config = yaml.safe_load(reader)
    return config

def get_obs_image(env):
    transform = T.Compose([T.ToTensor()])
    current_frames = env.get_frames()
    image_tensors = [transform(i).cuda() for i in current_frames]
    for i in range(len(image_tensors)):
        image_tensors[i] = image_tensors[i].permute(1, 2, 0)
        image_tensors[i]*= 255
        image_tensors[i] = image_tensors[i].int()
        image_tensors[i] = image_tensors[i][:,:,[2,1,0]]
    image_tensors = torch.stack(image_tensors, dim=0)
    return image_tensors

def compute_reward(info, multi_modal=False):
    if multi_modal:
        reward = 10.0 * float(info['won']) + float(info['goal_condition_success_rate'])
    else:
        reward = 10.0 * float(info['won'])
    return reward

class AlfworldWorker:
    """
    Ray remote actor that replaces the worker function.
    Each actor holds one environment instance.
    """

    def __init__(self, config, seed, base_env, milestone_config=None):
        # The facts-based WiGPO milestone detector runs inside this worker: it
        # holds the task's goal (traj_data.json) and reads the per-step TextWorld
        # facts locally, so only a few scalar fields travel back to the trainer.
        self.milestone_config = milestone_config or {}
        self.use_milestone_detector = bool(self.milestone_config.get("use_milestone_detector", False))
        log_dir = self.milestone_config.get("milestone_log_dir", None)

        self.env = base_env.init_env(batch_size=1, request_facts=self.use_milestone_detector)
        self.env.seed(seed)

        self.milestone_detector = None
        self.milestone_detector_log_dir = log_dir

    def _build_milestone_detector(self, gamefile):
        """Build the facts-based detector from the episode's own traj_data.json."""
        if not self.use_milestone_detector:
            self.milestone_detector = None
            self.milestone_task_type = None
            return
        from gigpo.alfworld_milestone_detector import (
            AlfworldMilestoneDetector,
            load_task_params,
        )

        task_type, goal = load_task_params(gamefile)
        if not task_type or not goal:
            print(f"[Alfworld] milestone detector: no task params found for {gamefile}")
            self.milestone_detector = None
            self.milestone_task_type = None
            return
        self.milestone_task_type = task_type
        self.milestone_detector = AlfworldMilestoneDetector(
            task_type,
            goal,
            log_dir=self.milestone_detector_log_dir,
        )

    @staticmethod
    def _extract_facts(infos):
        """Pull the TextWorld facts out of the info dict.

        With ``request_infos.facts=True`` the gym env returns facts under the
        ``facts`` key. Defensively handle the batch=1 shape (may be wrapped in a
        list/tuple) and an absent key (detector falls back to no progress).
        """
        facts = infos.get("facts", None)
        if facts is None:
            return []
        # textworld gym with batch handles single element sometimes; unwrap.
        if isinstance(facts, (list, tuple)) and len(facts) == 1:
            return facts[0]
        return facts

    def step(self, action):
        """Execute a step in the environment"""
        actions = [action]
        obs, scores, dones, infos = self.env.step(actions)
        infos['observation_text'] = obs

        # WiGPO: feed the step's true-state facts into the detector and expose
        # the milestone result (only when the detector is enabled). Stored as
        # 1-element arrays because the parent AlfworldEnvs flattens every info
        # value with `value[0]` (batch_size=1 convention).
        if self.milestone_detector is not None:
            won = bool(infos.get('won', False))
            facts = self._extract_facts(infos)
            result = self.milestone_detector.process(facts, won=won)
            infos['milestone_achieved'] = np.array([bool(result.achieved)])
            infos['milestone_phase'] = np.array([result.phase.value])
            infos['milestone_name'] = np.array([self._milestone_name()])
            infos['num_milestones'] = np.array([self.milestone_detector.num_milestones])
        return obs, scores, dones, infos

    def _milestone_name(self):
        det = self.milestone_detector
        if det is None:
            return ""
        idx = min(det.completed_count, len(det.milestone_names) - 1)
        return det.milestone_names[idx] if det.milestone_names else ""

    def reset(self):
        """Reset the environment"""
        obs, infos = self.env.reset()
        infos['observation_text'] = obs

        # WiGPO: (re)build the milestone detector for the new episode.
        gamefile = infos.get('extra.gamefile', None)
        self._build_milestone_detector(gamefile)
        if self.milestone_detector is not None:
            infos['milestone_achieved'] = np.array([False])
            infos['milestone_phase'] = np.array([self.milestone_detector.phase.value])
            infos['milestone_name'] = np.array([self._milestone_name()])
            infos['num_milestones'] = np.array([self.milestone_detector.num_milestones])
        return obs, infos

    def getobs(self):
        """Get current observation image"""
        image = get_obs_image(self.env)
        image = image.cpu()
        return image

class AlfworldEnvs(gym.Env):
    def __init__(self, alf_config_path, seed, env_num, group_n, resources_per_worker, is_train=True, env_kwargs={}):
        super().__init__()
        
        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            ray.init()
            
        eval_dataset = env_kwargs.get('eval_dataset', 'eval_in_distribution')
        milestone_config = dict(env_kwargs.get('milestone_config', {}) or {})
        config = load_config_file(alf_config_path)
        env_type = config['env']['type']
        base_env = get_environment(env_type)(config, train_eval='train' if is_train else eval_dataset)
        self.multi_modal = (env_type == 'AlfredThorEnv')
        self.num_processes = env_num * group_n
        self.group_n = group_n

        # Create Ray remote actors instead of processes
        env_worker = ray.remote(**resources_per_worker)(AlfworldWorker)
        self.workers = []
        for i in range(self.num_processes):
            worker = env_worker.remote(config, seed + (i // self.group_n), base_env,
                                       milestone_config=milestone_config)
            self.workers.append(worker)

        self.prev_admissible_commands = [None for _ in range(self.num_processes)]

    def step(self, actions):
        assert len(actions) == self.num_processes, \
            "The num of actions must be equal to the num of processes"

        # Send step commands to all workers
        futures = []
        for i, worker in enumerate(self.workers):
            future = worker.step.remote(actions[i])
            futures.append(future)

        # Collect results
        text_obs_list = []
        image_obs_list = []
        rewards_list = []
        dones_list = []
        info_list = []

        results = ray.get(futures)
        for i, (obs, scores, dones, info) in enumerate(results):
            for k in info.keys():
                info[k] = info[k][0]

            text_obs_list.append(obs[0])
            dones_list.append(dones[0])
            info_list.append(info)

            self.prev_admissible_commands[i] = info['admissible_commands']
            rewards_list.append(compute_reward(info, self.multi_modal))

        if self.multi_modal:
            image_obs_list = self.getobs()
        else:
            image_obs_list = None

        return text_obs_list, image_obs_list, rewards_list, dones_list, info_list

    def reset(self):
        """
        Send the reset command to all workers at once and collect initial obs/info from each environment.
        """
        text_obs_list = []
        image_obs_list = []
        info_list = []

        # Send reset commands to all workers
        futures = []
        for worker in self.workers:
            future = worker.reset.remote()
            futures.append(future)

        # Collect results
        results = ray.get(futures)
        for i, (obs, info) in enumerate(results):
            for k in info.keys():
                info[k] = info[k][0] 
            text_obs_list.append(obs[0])
            self.prev_admissible_commands[i] = info['admissible_commands']
            info_list.append(info)

        if self.multi_modal:
            image_obs_list = self.getobs()
        else:
            image_obs_list = None

        return text_obs_list, image_obs_list, info_list

    def getobs(self):
        """
        Ask each worker to return its current frame image.
        Usually needed only for multi-modal environments; otherwise can return None.
        """
        futures = []
        for worker in self.workers:
            future = worker.getobs.remote()
            futures.append(future)

        images = ray.get(futures)
        return images

    @property
    def get_admissible_commands(self):
        """
        Simply return the prev_admissible_commands stored by the main process.
        You could also design it to fetch after each step or another method.
        """
        return self.prev_admissible_commands

    def close(self):
        """
        Close all workers
        """
        # Kill all Ray actors
        for worker in self.workers:
            ray.kill(worker)

def build_alfworld_envs(alf_config_path, seed, env_num, group_n, resources_per_worker, is_train=True, env_kwargs={}):
    return AlfworldEnvs(alf_config_path, seed, env_num, group_n, resources_per_worker, is_train, env_kwargs)