import numpy as np
import torch
import torch.optim as optim
from torch.distributions import Categorical
from a2c_model import *
import torch.nn.functional as F

class A2CAgent:

	def __init__(
		self, 
		env, 
		arguments
		):

		self.env = env
		self.env_name = arguments.environment
		self.value_lr = arguments.value_lr
		self.policy_lr = arguments.policy_lr
		self.gamma = arguments.gamma
		self.entropy_pen = arguments.entropy_pen
		self.gae_lambda = arguments.gae_lambda
		self.gae = arguments.gae
		self.critic_loss_type = arguments.critic_loss_type
		self.norm_adv = arguments.norm_adv
		self.norm_rew = arguments.norm_rew
		self.gif = arguments.save_gif
		# TD lambda
		self.td_lambda = arguments.td_lambda
		self.experiment_type = arguments.experiment_type
		# Used for masking advantages above a threshold
		self.select_above_threshold = arguments.select_above_threshold
		self.threshold_max = arguments.threshold_max
		self.steps_to_take = arguments.steps_to_take

		if self.experiment_type == "prd":
			self.threshold_delta = (self.threshold_max - self.select_above_threshold)/self.steps_to_take


		self.device = arguments.device
		
		self.num_agents = self.env.n
		self.num_actions = self.env.action_space[0].n

		if self.env_name == "paired_by_switching_goals":
			obs_dim = 2*4
			self.critic_network = GATCritic(obs_dim, 128, obs_dim+self.num_actions, 128, 128, 1, self.num_agents, self.num_actions, self.device).to(self.device)
		elif self.env_name == "crossing_greedy":
			obs_dim = 2*3
			self.critic_network = GATCritic(obs_dim, 128, obs_dim+self.num_actions, 128, 128, 1, self.num_agents, self.num_actions, self.device).to(self.device)
		elif self.env_name == "crossing_fully_coop":
			obs_dim = 2*3
			self.critic_network = DualGATCritic(obs_dim, 128, obs_dim+self.num_actions, 128, 128, 1, self.num_agents, self.num_actions, self.device).to(self.device)
		elif self.env_name == "color_social_dilemma":
			obs_dim = 2*2 + 1 + 2*3
			self.critic_network = GATCritic(obs_dim, 128, obs_dim+self.num_actions, 128, 128, 1, self.num_agents, self.num_actions, self.device).to(self.device)
		elif self.env_name == "crossing_partially_coop":
			obs_dim = 2*3 + 1
			self.critic_network = DualGATCritic(obs_dim, 128, obs_dim+self.num_actions, 128, 128, 1, self.num_agents, self.num_actions, self.device).to(self.device)
		
		
		if self.env_name in ["paired_by_switching_goals", "crossing_greedy", "crossing_fully_coop"]:
			obs_dim = 2*3
		elif self.env_name in ["color_social_dilemma"]:
			obs_dim = 2*2 + 1 + 2*3
		elif self.env_name in ["crossing_partially_coop"]:
			obs_dim = 2*3 + 1

		# MLP POLICY
		self.policy_network = MLPPolicyNetwork(obs_dim, self.num_agents, self.num_actions, self.device).to(self.device)


		self.greedy_policy = torch.zeros(self.num_agents,self.num_agents).to(self.device)
		for i in range(self.num_agents):
			self.greedy_policy[i][i] = 1



		# Loading models
		if arguments.load_models:
			# For CPU
			if self.device == "cpu":
				self.critic_network.load_state_dict(torch.load(arguments.critic_saved_path,map_location=torch.device('cpu')))
				self.policy_network.load_state_dict(torch.load(arguments.actor_saved_path,map_location=torch.device('cpu')))
			# # For GPU
			else:
				self.critic_network.load_state_dict(torch.load(arguments.critic_saved_path))
				self.policy_network.load_state_dict(torch.load(arguments.actor_saved_path))

		
		self.critic_optimizer = optim.Adam(self.critic_network.parameters(),lr=self.value_lr)
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
				
				advantage = td_error + (self.gamma * self.gae_lambda * advantage * masks[t])
				advantages.insert(0, advantage)

			advantages = torch.stack(advantages)	
		else:
			advantages = returns - values
		
		if self.norm_adv:
			
			advantages = (advantages - advantages.mean()) / advantages.std()
		
		return advantages


	def calculate_deltas(self, values, rewards, dones):
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
		advs = self.calculate_returns(deltas, self.gamma*self.td_lambda)
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

		'''
		Getting the probability mass function over the action space for each agent
		'''
		probs, weight_policy = self.policy_network.forward(states_actor)

		'''
		Calculate V values
		'''
		if self.env_name in ["crossing_fully_coop", "crossing_partially_coop"]:
			V_values, weights_preproc, weights_post = self.critic_network.forward(states_critic, probs.detach(), one_hot_actions)
		else:
			V_values, weights = self.critic_network.forward(states_critic, probs.detach(), one_hot_actions)
		V_values = V_values.reshape(-1,self.num_agents,self.num_agents)
	
		# # ***********************************************************************************
		# update critic (value_net)
		# we need a TxNxN vector so inflate the discounted rewards by N --> cloning the discounted rewards for an agent N times
		discounted_rewards = None

		if self.critic_loss_type == "MC":
			discounted_rewards = self.calculate_returns(rewards,self.gamma).unsqueeze(-2).repeat(1,self.num_agents,1).to(self.device)
			discounted_rewards = torch.transpose(discounted_rewards,-1,-2)
			value_loss = F.smooth_l1_loss(V_values,discounted_rewards)
		elif self.critic_loss_type == "TD_1":
			next_probs, _ = self.policy_network.forward(next_states_actor)
			V_values_next, _ = self.critic_network.forward(next_states_critic, next_probs.detach(), one_hot_next_actions)
			V_values_next = V_values_next.reshape(-1,self.num_agents,self.num_agents)
			target_values = torch.transpose(rewards.unsqueeze(-2).repeat(1,self.num_agents,1),-1,-2) + self.gamma*V_values_next*(1-dones.unsqueeze(-1))
			value_loss = F.smooth_l1_loss(V_values,target_values)
		elif self.critic_loss_type == "TD_lambda":
			Value_target = self.nstep_returns(V_values, rewards, dones).detach()
			value_loss = F.smooth_l1_loss(V_values, Value_target)

		
	
		# # ***********************************************************************************
		# update actor (policy net)
		# # ***********************************************************************************
		entropy = -torch.mean(torch.sum(probs * torch.log(torch.clamp(probs, 1e-10,1.0)), dim=2))

		# summing across each agent j to get the advantage
		# so we sum across the second last dimension which does A[t,j] = sum(V[t,i,j] - discounted_rewards[t,i])
		if self.env_name in ["crossing_fully_coop", "crossing_partially_coop"]:
			weights_prd = (weights_post + weights_preproc)/2.0
		else:
			weights_prd = weights

		advantage = None
		masking_advantage = None
		if self.experiment_type == "shared":
			advantage = torch.sum(self.calculate_advantages(discounted_rewards, V_values, rewards, dones),dim=-2)

		elif self.experiment_type == "prd":
			masking_advantage = (weights_prd>self.select_above_threshold).int()
			advantage = torch.sum(self.calculate_advantages(discounted_rewards, V_values, rewards, dones) * torch.transpose(masking_advantage,-1,-2),dim=-2)
			
			if self.threshold_max >= self.select_above_threshold:
				self.select_above_threshold = self.select_above_threshold + self.threshold_delta

			agent_groups_over_episode = torch.sum(torch.sum(masking_advantage.float(), dim=-2),dim=0)/masking_advantage.shape[0]
			avg_agent_group_over_episode = torch.mean(agent_groups_over_episode)

		elif self.experiment_type == "greedy":
			advantage = torch.sum(self.calculate_advantages(discounted_rewards, V_values, rewards, dones) * self.greedy_policy ,dim=-2)
		
			

	
		probs = Categorical(probs)
		policy_loss = -probs.log_prob(actions) * advantage.detach()
		policy_loss = policy_loss.mean() - self.entropy_pen*entropy
		# # ***********************************************************************************
			
		# **********************************
		self.critic_optimizer.zero_grad()
		value_loss.backward(retain_graph=False)
		grad_norm_value = torch.nn.utils.clip_grad_norm_(self.critic_network.parameters(),0.5)
		self.critic_optimizer.step()


		self.policy_optimizer.zero_grad()
		policy_loss.backward(retain_graph=False)
		grad_norm_policy = torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(),0.5)
		self.policy_optimizer.step()

		


		if self.env_name in ["crossing_fully_coop", "crossing_partially_coop"]:
			if self.experiment_type == "prd":
				return value_loss, policy_loss, entropy, grad_norm_value, grad_norm_policy, weights_preproc, weights_post, weight_policy, agent_groups_over_episode, avg_agent_group_over_episode
			else:
				return value_loss, policy_loss, entropy, grad_norm_value, grad_norm_policy, weights_preproc, weights_post, weight_policy
		else:
			if self.experiment_type == "prd":
				return value_loss, policy_loss, entropy, grad_norm_value, grad_norm_policy, weights, weight_policy, agent_groups_over_episode, avg_agent_group_over_episode
			else:
				return value_loss, policy_loss, entropy, grad_norm_value, grad_norm_policy, weights, weight_policy