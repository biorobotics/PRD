import numpy as np
import torch 
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable
from torch.distributions import Categorical
from a2c_coma import *
import torch.nn.functional as F
import math

class A2CAgent:

	def __init__(
		self, 
		env, 
		dictionary
		):

		self.env = env
		self.env_name = dictionary["env"]
		self.value_lr = dictionary["value_lr"]
		self.policy_lr = dictionary["policy_lr"]
		self.gamma = dictionary["gamma"]
		self.entropy_pen = dictionary["entropy_pen"]
		self.trace_decay = dictionary["trace_decay"]
		self.top_k = dictionary["top_k"]
		self.gae = dictionary["gae"]
		self.critic_loss_type = dictionary["critic_loss_type"]
		self.norm_adv = dictionary["norm_adv"]
		self.norm_rew = dictionary["norm_rew"]
		# Used for masking advantages above a threshold
		self.select_above_threshold = dictionary["select_above_threshold"]

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		# self.device = "cpu"
		
		self.num_agents = self.env.n
		self.num_actions = self.env.action_space[0].n
		self.gif = dictionary["gif"]

		self.experiment_type = dictionary["experiment_type"]
		self.coma_version = dictionary["version"]

		
		# DECAYING ENTROPY PEN 
		self.steps_done = 0
		self.entropy_pen_start = dictionary["entropy_pen"]
		self.entropy_pen_end = 0.0
		self.entropy_pen_decay = 0.0
		# self.entropy_pen = self.entropy_pen_end + (self.entropy_pen_start-self.entropy_pen_end)*math.exp(-1*self.steps_done / self.entropy_pen_decay)

		# TD lambda
		self.lambda_ = 0.8


		self.greedy_policy = torch.zeros(self.num_agents,self.num_agents).to(self.device)
		for i in range(self.num_agents):
			self.greedy_policy[i][i] = 1

		print("EXPERIMENT TYPE:",self.experiment_type)

		obs_dim = 2*4
		# SCALAR DOT PRODUCT
		if self.coma_version == 1:
			self.critic_network = GATCriticV1(obs_dim, 128, obs_dim+self.num_actions, 128, 128, self.num_actions, self.num_agents, self.num_actions).to(self.device)
		elif self.coma_version == 2:
			self.critic_network = GATCriticV2(obs_dim, 128, obs_dim+self.num_actions, 128, 128, self.num_actions, self.num_agents, self.num_actions).to(self.device)
		elif self.coma_version == 3:
			self.critic_network_Q = GATCriticV2(obs_dim, 128, obs_dim+self.num_actions, 128, 128, self.num_actions, self.num_agents, self.num_actions).to(self.device)
			self.critic_network_V = GATCriticV2(obs_dim, 128, obs_dim+self.num_actions, 128, 128, 1, self.num_agents, self.num_actions).to(self.device)
		else:
			self.critic_network_V = GATCriticV2(obs_dim, 128, obs_dim+self.num_actions, 128, 128, 1, self.num_agents, self.num_actions).to(self.device)
		
		
		# MLP POLICY
		obs_dim = 2*3
		self.policy_network = MLPPolicyNetwork(obs_dim, self.num_agents, self.num_actions).to(self.device)

		# GAT POLICY NETWORK
		# self.obs_input_dim = 2*3
		# self.obs_output_dim = 64
		# self.final_input_dim = self.obs_output_dim
		# self.final_output_dim = self.num_actions
		# self.policy_network = ScalarDotProductPolicyNetwork(self.obs_input_dim, self.obs_output_dim, self.final_input_dim, self.final_output_dim, self.num_agents, self.num_actions, self.softmax_cut_threshold).to(self.device)


		# Loading models
		# model_path_value = "../../../models/Scalar_dot_product/collision_avoidance/4_Agents/SingleAttentionMechanism/with_prd_soft_adv/critic_networks/14-05-2021VN_ATN_FCN_lr0.01_PN_FCN_lr0.0002_GradNorm0.5_Entropy0.008_trace_decay0.98lambda_0.001topK_2select_above_threshold0.1softmax_cut_threshold0.1_epsiode29000.pt"
		# model_path_policy = "../../../models/Scalar_dot_product/collision_avoidance/4_Agents/SingleAttentionMechanism/with_prd_soft_adv/actor_networks/14-05-2021_PN_FCN_lr0.0002VN_SAT_FCN_lr0.01_GradNorm0.5_Entropy0.008_trace_decay0.98lambda_0.001topK_2select_above_threshold0.1softmax_cut_threshold0.1_epsiode29000.pt"
		# For CPU
		# self.critic_network.load_state_dict(torch.load(model_path_value,map_location=torch.device('cpu')))
		# self.policy_network.load_state_dict(torch.load(model_path_policy,map_location=torch.device('cpu')))
		# # For GPU
		# self.critic_network.load_state_dict(torch.load(model_path_value))
		# self.policy_network.load_state_dict(torch.load(model_path_policy))

		if self.coma_version == 1 or self.coma_version == 2:
			self.critic_optimizer = optim.Adam(self.critic_network.parameters(),lr=self.value_lr)
		elif self.coma_version == 3:
			self.critic_optimizer_Q = optim.Adam(self.critic_network_Q.parameters(),lr=self.value_lr)
			self.critic_optimizer_V = optim.Adam(self.critic_network_V.parameters(),lr=self.value_lr)
		else:
			self.critic_optimizer_V = optim.Adam(self.critic_network_V.parameters(),lr=self.value_lr)


		self.policy_optimizer = optim.Adam(self.policy_network.parameters(),lr=self.policy_lr)


	def get_action(self,state):
		state = torch.FloatTensor([state]).to(self.device)
		dists, _ = self.policy_network.forward(state)
		index = [Categorical(dist).sample().cpu().detach().item() for dist in dists[0]]
		return index



	def calculate_advantages(self,returns, values, rewards, dones):
		
		advantages = None

		if self.gae:
			advantages = []
			next_value = 0
			advantage = 0
			rewards = rewards.unsqueeze(-1)
			dones = dones.unsqueeze(-1)
			masks = 1 - dones
			for t in reversed(range(0, len(rewards))):
				td_error = rewards[t] + (self.gamma * next_value * masks[t]) - values.data[t]
				next_value = values.data[t]
				
				advantage = td_error + (self.gamma * self.trace_decay * advantage * masks[t])
				advantages.insert(0, advantage)

			advantages = torch.stack(advantages)	
		else:
			advantages = returns - values
		
		if self.norm_adv:
			advantages = (advantages - advantages.mean()) / advantages.std()
		
		return advantages

	def calculate_deltas(self, values, rewards, dones):
		if self.coma_version == 1:
			deltas = []
			next_value = 0
			rewards = rewards
			dones = dones
			masks = 1-dones
			for t in reversed(range(0, len(rewards))):
				td_error = rewards[t] + (self.gamma * next_value * masks[t]) - values.data[t]
				next_value = values.data[t]
				deltas.insert(0,td_error)
			deltas = torch.stack(deltas)

			return deltas


		deltas = []
		next_value = 0
		rewards = rewards.unsqueeze(-1)
		dones = dones.unsqueeze(-1)
		masks = 1-dones
		for t in reversed(range(0, len(rewards))):
			td_error = rewards[t] + (self.gamma * next_value * masks[t]) - values.data[t]
			next_value = values.data[t]
			deltas.insert(0,td_error)
		deltas = torch.stack(deltas)

		return deltas


	def nstep_returns(self,values, rewards, dones):
		deltas = self.calculate_deltas(values, rewards, dones)
		advs = self.calculate_returns(deltas, self.gamma*self.lambda_)
		target_Vs = advs+values
		return target_Vs


	def calculate_returns(self,rewards, discount_factor):
	
		returns = []
		R = 0
		
		for r in reversed(rewards):
			R = r + R * discount_factor
			returns.insert(0, R)
		
		returns_tensor = torch.stack(returns).to(self.device)
		
		if self.norm_rew:
			returns_tensor = (returns_tensor - returns_tensor.mean()) / returns_tensor.std()
			
		return returns_tensor
		
	

	def update(self,states_critic,next_states_critic,one_hot_actions,one_hot_next_actions,actions,states_actor,next_states_actor,rewards,dones):

		probs, weight_policy = self.policy_network.forward(states_actor)


		if self.coma_version == 1:
			Q_values, weights_V = self.critic_network.forward(states_critic, probs.detach(), one_hot_actions)
			Q_values_act_chosen = torch.sum(Q_values.reshape(-1,self.num_agents, self.num_actions) * one_hot_actions, dim=-1)
			V_values_baseline = torch.sum(Q_values.reshape(-1,self.num_agents, self.num_actions) * probs.detach(), dim=-1)
		elif self.coma_version == 2:
			Q_values, weights_V = self.critic_network.forward(states_critic, probs.detach(), one_hot_actions)
			Q_values_act_chosen = torch.sum(Q_values.reshape(-1,self.num_agents,self.num_agents, self.num_actions) * one_hot_actions.unsqueeze(-2), dim=-1)
			V_values_baseline = torch.sum(Q_values.reshape(-1,self.num_agents,self.num_agents, self.num_actions) * probs.detach().unsqueeze(-2), dim=-1)
		elif self.coma_version == 3:
			Q_values, weights_Q = self.critic_network_Q.forward(states_critic, probs.detach(), one_hot_actions)
			Q_values_act_chosen = torch.sum(Q_values.reshape(-1,self.num_agents,self.num_agents, self.num_actions) * one_hot_actions.unsqueeze(-2), dim=-1)
			V_values_baseline, weights_V = self.critic_network_V.forward(states_critic, probs.detach(), one_hot_actions)
			V_values_baseline = V_values_baseline.reshape(-1,self.num_agents,self.num_agents)
		else:
			V_values_baseline, weights_V = self.critic_network_V.forward(states_critic, probs.detach(), one_hot_actions)
			V_values_baseline = V_values_baseline.reshape(-1,self.num_agents,self.num_agents)

		
	# # ***********************************************************************************
	# 	#update critic (value_net)
		if self.coma_version == 1:
			discounted_rewards = self.calculate_returns(rewards,self.gamma).to(self.device)
		else:
			# we need a TxNxN vector so inflate the discounted rewards by N --> cloning the discounted rewards for an agent N times
			discounted_rewards = self.calculate_returns(rewards,self.gamma).unsqueeze(-2).repeat(1,self.num_agents,1).to(self.device)
			discounted_rewards = torch.transpose(discounted_rewards,-1,-2)

		if self.critic_loss_type == "MC":
			if self.coma_version == 1 or self.coma_version == 2:
				value_loss_V = F.smooth_l1_loss(Q_values_act_chosen,discounted_rewards)
			elif self.coma_version == 3:
				value_loss_Q = F.smooth_l1_loss(Q_values_act_chosen,discounted_rewards)
				value_loss_V = F.smooth_l1_loss(V_values_baseline,discounted_rewards)
			else:
				value_loss_V = F.smooth_l1_loss(V_values_baseline,discounted_rewards)
		elif self.critic_loss_type == "TD_lambda":
			if self.coma_version in [1,2]:
				Value_targets = self.nstep_returns(V_values_baseline, rewards, dones)
				value_loss_V = F.smooth_l1_loss(Q_values_act_chosen, Value_targets.detach())
			elif self.coma_version == 3:
				Value_targets = self.nstep_returns(V_values_baseline, rewards, dones)
				value_loss_V = F.smooth_l1_loss(V_values_baseline, Value_targets.detach())
				value_loss_Q = F.smooth_l1_loss(Q_values_act_chosen,Value_targets.detach())
			elif self.coma_version in [4,5,6]:
				Value_targets = self.nstep_returns(V_values_baseline, rewards, dones)
				value_loss_V = F.smooth_l1_loss(V_values_baseline, Value_targets.detach())
			
		
		# # ***********************************************************************************
	# 	#update actor (policy net)
	# # ***********************************************************************************
		entropy = -torch.mean(torch.sum(probs * torch.log(torch.clamp(probs, 1e-10,1.0)), dim=2))

		advantage = None
		
		if self.coma_version == 1:
			advantage = Q_values_act_chosen - V_values_baseline
		elif self.coma_version == 2 or self.coma_version == 3:
			advantage = torch.sum(Q_values_act_chosen - V_values_baseline, dim=-2)
		elif self.coma_version == 4:
			advantage = torch.sum(self.calculate_advantages(discounted_rewards, V_values_baseline, rewards, dones), dim=-2)
		elif self.coma_version == 5:
			advantage = torch.sum(self.calculate_advantages(discounted_rewards, V_values_baseline, rewards, dones), dim=-2)
		elif self.coma_version == 6:
			masking_advantage = (weights_V>self.select_above_threshold).int()
			advantage = torch.sum(self.calculate_advantages(discounted_rewards, V_values_baseline, rewards, dones) * masking_advantage,dim=-2)

		probs = Categorical(probs)
		policy_loss = -probs.log_prob(actions) * advantage.detach()
		policy_loss = policy_loss.mean() - self.entropy_pen*entropy
	# # ***********************************************************************************
		
	# **********************************
		if self.coma_version == 1 or self.coma_version == 2:
			self.critic_optimizer.zero_grad()
			value_loss_V.backward(retain_graph=False)
			grad_norm_value_V = torch.nn.utils.clip_grad_norm_(self.critic_network.parameters(),0.5)
			self.critic_optimizer.step()
		elif self.coma_version == 3:
			self.critic_optimizer_Q.zero_grad()
			value_loss_Q.backward(retain_graph=False)
			grad_norm_value_Q = torch.nn.utils.clip_grad_norm_(self.critic_network_Q.parameters(),0.5)
			self.critic_optimizer_Q.step()

			self.critic_optimizer_V.zero_grad()
			value_loss_V.backward(retain_graph=False)
			grad_norm_value_V = torch.nn.utils.clip_grad_norm_(self.critic_network_V.parameters(),0.5)
			self.critic_optimizer_V.step()
		else:
			self.critic_optimizer_V.zero_grad()
			value_loss_V.backward(retain_graph=False)
			grad_norm_value_V = torch.nn.utils.clip_grad_norm_(self.critic_network_V.parameters(),0.5)
			self.critic_optimizer_V.step()


		self.policy_optimizer.zero_grad()
		policy_loss.backward(retain_graph=False)
		grad_norm_policy = torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(),0.5)
		self.policy_optimizer.step()

		# DECAY ENTROPY PEN
		# self.steps_done += 1
		# self.entropy_pen = self.entropy_pen_end + (self.entropy_pen_start-self.entropy_pen_end)*math.exp(-1*self.steps_done / self.entropy_pen_decay)

		if self.coma_version == 3:
			return value_loss_Q, value_loss_V, policy_loss, entropy, grad_norm_value_Q, grad_norm_value_V, grad_norm_policy, weights_Q, weights_V, weight_policy
		else:
			return value_loss_V, policy_loss, entropy, grad_norm_value_V, grad_norm_policy, weights_V, weight_policy