import os
import numpy as np
import datetime
from typing import Any, List, Tuple, Union
import math

import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
from torch.distributions import Categorical
import torch.autograd as autograd

from multiagent.environment import MultiAgentEnv
import multiagent.scenarios as scenarios

from a2c_model import *


ENV_NAME = "color_social_dilemma" #color_social_dilemma, corssing_greedy, crossing_fully_coop, crossing_partially_coop, paired_by_sharing_goals
NUM_EVALS = 5
PRD_THRESHOLD_MIN = 0.0
PRD_THRESHOLD_MAX = 0.01
PRD_EPISODE = 15000
PRD_THRESHOLD = (PRD_THRESHOLD_MAX-PRD_THRESHOLD_MIN)/PRD_EPISODE


def make_env(scenario_name, benchmark=False):
	# load scenario from script
	scenario = scenarios.load(scenario_name + ".py").Scenario()
	# scenario = Scenario()
	# create world
	world = scenario.make_world()
	# create multiagent environment
	if benchmark:
		env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data, scenario.isFinished)
	else:
		env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, None, scenario.isFinished)
	return env


def split_states(states, num_agents):
	states_critic = []
	states_actor = []
	for i in range(num_agents):
		states_critic.append(states[i][0])
		states_actor.append(states[i][1])

	states_critic = np.asarray(states_critic)
	states_actor = np.asarray(states_actor)

	return states_critic,states_actor



def run(env, max_steps):

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	num_agents = env.n
	num_actions = env.action_space[0].n


	if ENV_NAME == "paired_by_sharing_goals":
		obs_dim = 2*4
	elif ENV_NAME in ["crossing_greedy", "crossing_fully_coop"]:
		obs_dim = 2*3
	elif ENV_NAME == "crossing_partial_coop":
		obs_dim = 2*3+1
	elif ENV_NAME == "color_social_dilemma":
		obs_dim = 2*2 + 1 + 2*3

	if ENV_NAME in ["paired_by_sharing_goals", "crossing_greedy", "color_social_dilemma"]:
		critic_network = GATCritic(obs_dim, 128, obs_dim+num_actions, 128, 128, 1, num_agents, num_actions).to(device)
	else:
		critic_network = DualGATCritic(obs_dim, 128, obs_dim+num_actions, 128, 128, 1, num_agents, num_actions).to(device)

	if ENV_NAME in ["paired_by_sharing_goals", "crossing_greedy", "crossing_fully_coop"]:
		obs_dim = 2*3
	elif ENV_NAME == "color_social_dilemma":
		obs_dim = 2*2 + 1 + 2*3
	elif ENV_NAME == "crossing_partial_coop":
		obs_dim = 2*3+1
	# MLP POLICY
	policy_network = MLPPolicyNetwork(obs_dim, num_agents, num_actions).to(device)

	if ENV_NAME in ["paired_by_sharing_goals", "crossing_greedy"]:
		eps_list = [str(i*1000) for i in range(1,21)]
	elif ENV_NAME == "color_social_dilemma":
		eps_list = [str(i*1000) for i in range(1,51)]
	elif ENV_NAME in ["crossing_fully_coop", "crossing_partial_coop"]:
		eps_list = [str(i*1000) for i in range(1,201)]

	exp_types = ["shared", "greedy", "prd_above_threshold_ascend"]


	for exp_type in exp_types: #prd_above_threshold_run_MAA2C_MC_prd_above_threshold1, shared


		rewards_per_1000_eps = []
		timesteps_per_1000_eps = []
		collison_rate_per_1000_eps = []
		avg_group_size_per_1000_eps = []

		for run_num in range(1,NUM_EVALS+1):			
			policy_eval_dir = '../../../tests/'+'/policy_eval/'+ENV_NAME+'_'+exp_type+'_'+run_num'/'
			try: 
				os.makedirs(policy_eval_dir, exist_ok = True) 
				print("Policy Eval Directory created successfully") 
			except OSError as error: 
				print("Policy Eval Directory can not be created")

			for eps in eps_list:
				# Loading models
				
				model_path_value = "../../../tests/policy_eval/"+"ENV_NAME"+exp_type+"_run1/critic_networks/30-07-2021VN_ATN_FCN_lr0.001_PN_ATN_FCN_lr0.0001_GradNorm0.5_Entropy0.0_trace_decay0.98topK_0select_above_threshold0.01_epsiode"+eps+".pt"
				model_path_policy = "../../../tests/policy_eval/"+"ENV_NAME"+exp_type+"_run1/actor_networks/30-07-2021_PN_ATN_FCN_lr0.0001VN_SAT_FCN_lr0.001_GradNorm0.5_Entropy0.0_trace_decay0.98topK_0select_above_threshold0.01_epsiode"+eps+".pt"
				# For CPU
				# critic_network.load_state_dict(torch.load(model_path_value,map_location=torch.device('cpu')))
				# policy_network.load_state_dict(torch.load(model_path_policy,map_location=torch.device('cpu')))
				# For GPU
				critic_network.load_state_dict(torch.load(model_path_value))
				policy_network.load_state_dict(torch.load(model_path_policy))


				states = env.reset()

				images = []

				states_critic,states_actor = split_states(states,num_agents)

				if int(eps) > PRD_EPISODE:
					PRD_THRESHOLD = ((PRD_THRESHOLD_MAX - PRD_THRESHOLD_MIN) * int(eps))/PRD_EPISODE

				episode_collision_rate = 0
				total_rewards = 0
				avg_group_size = 0
				final_timestep = max_steps

				for step in range(max_steps):

					actions = None
					dists = None
					with torch.no_grad():
						states_actor = torch.FloatTensor([states_actor]).to(device)
						dists, _ = policy_network.forward(states_actor)
						actions = [Categorical(dist).sample().cpu().detach().item() for dist in dists[0]]

						one_hot_actions = np.zeros((num_agents,num_actions))
						for i,act in enumerate(actions):
							one_hot_actions[i][act] = 1

						states_critic = torch.FloatTensor([states_critic]).to(device)
						V_values, weights = critic_network.forward(states_critic, dists.detach(), torch.FloatTensor(one_hot_actions).unsqueeze(0).to(device))

					# Advance a step and render a new image
					next_states,rewards,dones,info = env.step(actions)
					next_states_critic,next_states_actor = split_states(next_states, num_agents)

					if ENV_NAME in ["crossing_greedy", "crossing_fully_coop", "crossing_partial_coop"]:
						collision_rate = [value[1] for value in rewards]
						rewards = [value[0] for value in rewards]
						episode_collision_rate += np.sum(collision_rate)

					if "prd" in exp_type:
						masking_advantage = (weights_prd>PRD_THRESHOLD).int()
						avg_group_size += torch.mean(masking_advantage)

					total_rewards = np.sum(rewards)

					print("*"*100)
					print("TIMESTEP: {} | REWARD: {} \n".format(step,np.round(total_rewards,decimals=4)))
					print("*"*100)


					states_critic,states_actor = next_states_critic,next_states_actor
					states = next_states


					if all(dones):
						final_timestep = step
						break

				avg_group_size /= final_timestep

				rewards_per_1000_eps.append(total_rewards)
				timesteps_per_1000_eps.append(final_timestep)

				if ENV_NAME in ["crossing_greedy", "crossing_fully_coop", "crossing_partial_coop"]:
					collison_rate_per_1000_eps.append(episode_collision_rate)

				if "prd" in exp_type:
					avg_group_size_per_1000_eps.append(avg_group_size)


			np.save(os.path.join(policy_eval_dir,"rewards_per_1000_eps"), np.array(rewards_per_1000_eps), allow_pickle=True, fix_imports=True)
			np.save(os.path.join(policy_eval_dir,"timesteps_per_1000_eps"), np.array(timesteps_per_1000_eps), allow_pickle=True, fix_imports=True)
			if "crossing" in ENV_NAME:
				np.save(os.path.join(policy_eval_dir,"collision_rate_per_1000_eps"), np.array(collison_rate_per_1000_eps), allow_pickle=True, fix_imports=True)
			if "prd" in exp_type:
				np.save(os.path.join(policy_eval_dir,"avg_group_size_per_1000_eps"), np.array(avg_group_size_per_1000_eps), allow_pickle=True, fix_imports=True)			



if __name__ == '__main__':
	env = make_env(scenario_name=ENV_NAME,benchmark=False)

	run(env,100)