from typing import Any, List, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import datetime
import math


class MLPPolicyNetwork(nn.Module):
	def __init__(self,state_dim,num_agents,action_dim):
		super(MLPPolicyNetwork,self).__init__()

		self.state_dim = state_dim
		self.num_agents = num_agents		
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


		self.fc1 = nn.Linear(state_dim*num_agents,128)
		self.fc2 = nn.Linear(128,128)
		self.fc3 = nn.Linear(128,action_dim)

	def forward(self, states):

		# T x num_agents x state_dim
		T = states.shape[0]
		
		# [s0;s1;s2;s3]  -> [s0 s1 s2 s3; s1 s2 s3 s0; s2 s3 s1 s0 ....]

		states_aug = [torch.roll(states,i,1) for i in range(self.num_agents)]

		states_aug = torch.cat(states_aug,dim=2)

		x = self.fc1(states_aug)
		x = nn.ReLU()(x)
		x = self.fc2(x)
		x = nn.ReLU()(x)
		x = self.fc3(x)

		Policy = F.softmax(x, dim=-1)

		return Policy, 1/self.num_agents*torch.ones((T,self.num_agents,self.num_agents),device=self.device)


class GATCritic(nn.Module):
	def __init__(self, obs_input_dim, obs_output_dim, obs_act_input_dim, obs_act_output_dim, final_input_dim, final_output_dim, num_agents, num_actions):
		super(GATCritic, self).__init__()
		
		self.num_agents = num_agents
		self.num_actions = num_actions
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

		self.state_embed = nn.Sequential(nn.Linear(obs_input_dim, 128), nn.LeakyReLU())
		self.key_layer = nn.Linear(128, obs_output_dim, bias=False)
		self.query_layer = nn.Linear(128, obs_output_dim, bias=False)
		self.state_act_pol_embed = nn.Sequential(nn.Linear(obs_act_input_dim, 128), nn.LeakyReLU())
		self.attention_value_layer = nn.Linear(128, obs_act_output_dim, bias=False)
		# dimesion of key
		self.d_k_obs_act = obs_output_dim  

		# NOISE
		self.noise_normal = torch.distributions.Normal(loc=torch.tensor([0.0]), scale=torch.tensor([1.0]))
		self.noise_uniform = torch.rand
		# ********************************************************************************************************

		# ********************************************************************************************************
		# FCN FINAL LAYER TO GET VALUES
		self.final_value_layer_1 = nn.Linear(final_input_dim, 64, bias=False)
		self.final_value_layer_2 = nn.Linear(64, final_output_dim, bias=False)
		# ********************************************************************************************************	

		self.place_policies = torch.zeros(self.num_agents,self.num_agents,obs_act_input_dim).to(self.device)
		self.place_actions = torch.ones(self.num_agents,self.num_agents,obs_act_input_dim).to(self.device)
		one_hots = torch.ones(obs_act_input_dim)
		zero_hots = torch.zeros(obs_act_input_dim)

		for j in range(self.num_agents):
			self.place_policies[j][j] = one_hots
			self.place_actions[j][j] = zero_hots


		self.reset_parameters()


	def reset_parameters(self):
		"""Reinitialize learnable parameters."""
		gain_leaky = nn.init.calculate_gain('leaky_relu')

		nn.init.xavier_uniform_(self.state_embed[0].weight, gain=gain_leaky)
		nn.init.xavier_uniform_(self.state_act_pol_embed[0].weight, gain=gain_leaky)

		nn.init.xavier_uniform_(self.key_layer.weight)
		nn.init.xavier_uniform_(self.query_layer.weight)
		nn.init.xavier_uniform_(self.attention_value_layer.weight)


		nn.init.xavier_uniform_(self.final_value_layer_1.weight, gain=gain_leaky)
		nn.init.xavier_uniform_(self.final_value_layer_2.weight, gain=gain_leaky)



	def forward(self, states, policies, actions):
		# EMBED STATES
		states_embed = self.state_embed(states)
		# KEYS
		key_obs = self.key_layer(states_embed)
		# QUERIES
		query_obs = self.query_layer(states_embed)
		# WEIGHT
		weight = F.softmax(torch.matmul(query_obs,key_obs.transpose(1,2))/math.sqrt(self.d_k_obs_act),dim=-1)
		ret_weight = weight

		obs_actions = torch.cat([states,actions],dim=-1)
		obs_policy = torch.cat([states,policies], dim=-1)
		obs_actions = obs_actions.repeat(1,self.num_agents,1).reshape(obs_actions.shape[0],self.num_agents,self.num_agents,-1)
		obs_policy = obs_policy.repeat(1,self.num_agents,1).reshape(obs_policy.shape[0],self.num_agents,self.num_agents,-1)
		obs_actions_policies = self.place_policies*obs_policy + self.place_actions*obs_actions
		# EMBED STATE ACTION POLICY
		obs_actions_policies_embed = self.state_act_pol_embed(obs_actions_policies)
		attention_values = self.attention_value_layer(obs_actions_policies_embed)
		attention_values = attention_values.repeat(1,self.num_agents,1,1).reshape(attention_values.shape[0],self.num_agents,self.num_agents,self.num_agents,-1)
		
		weight = weight.unsqueeze(-2).repeat(1,1,self.num_agents,1).unsqueeze(-1)
		weighted_attention_values = attention_values*weight
		node_features = torch.sum(weighted_attention_values, dim=-2)

		Value = F.leaky_relu(self.final_value_layer_1(node_features))
		Value = self.final_value_layer_2(Value)

		return Value, ret_weight



class DualGATCritic(nn.Module):
	def __init__(self, obs_input_dim, obs_output_dim, obs_act_input_dim, obs_act_output_dim, final_input_dim, final_output_dim, num_agents, num_actions):
		super(DualGATCritic, self).__init__()
		
		self.num_agents = num_agents
		self.num_actions = num_actions
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

		self.state_embed_preproc = nn.Sequential(nn.Linear(obs_input_dim, 128), nn.LeakyReLU())
		self.key_layer_preproc = nn.Linear(128, obs_output_dim, bias=False)
		self.query_layer_preproc = nn.Linear(128, obs_output_dim, bias=False)
		self.attention_value_layer_preproc = nn.Linear(128, obs_output_dim, bias=False)
		self.d_k_obs = obs_output_dim

		self.state_embed = nn.Sequential(nn.Linear(obs_output_dim, 128), nn.LeakyReLU())
		self.key_layer = nn.Linear(128, obs_output_dim, bias=False)
		self.query_layer = nn.Linear(128, obs_output_dim, bias=False)
		self.state_act_pol_embed = nn.Sequential(nn.Linear(obs_act_input_dim, 128), nn.LeakyReLU())
		self.attention_value_layer = nn.Linear(128, obs_act_output_dim, bias=False)
		# dimesion of key
		self.d_k_obs_act = obs_output_dim  

		# NOISE
		self.noise_normal = torch.distributions.Normal(loc=torch.tensor([0.0]), scale=torch.tensor([1.0]))
		self.noise_uniform = torch.rand
		# ********************************************************************************************************

		# ********************************************************************************************************
		# FCN FINAL LAYER TO GET VALUES
		self.final_value_layer_1 = nn.Linear(final_input_dim, 64, bias=False)
		self.final_value_layer_2 = nn.Linear(64, final_output_dim, bias=False)
		# ********************************************************************************************************	

		self.place_policies = torch.zeros(self.num_agents,self.num_agents,obs_act_input_dim).to(self.device)
		self.place_actions = torch.ones(self.num_agents,self.num_agents,obs_act_input_dim).to(self.device)
		one_hots = torch.ones(obs_act_input_dim)
		zero_hots = torch.zeros(obs_act_input_dim)

		for j in range(self.num_agents):
			self.place_policies[j][j] = one_hots
			self.place_actions[j][j] = zero_hots


		self.reset_parameters()


	def reset_parameters(self):
		"""Reinitialize learnable parameters."""
		gain_leaky = nn.init.calculate_gain('leaky_relu')

		nn.init.xavier_uniform_(self.state_embed_preproc[0].weight, gain=gain_leaky)

		nn.init.xavier_uniform_(self.key_layer_preproc.weight)
		nn.init.xavier_uniform_(self.query_layer_preproc.weight)
		nn.init.xavier_uniform_(self.attention_value_layer_preproc.weight)

		nn.init.xavier_uniform_(self.state_embed[0].weight, gain=gain_leaky)
		nn.init.xavier_uniform_(self.state_act_pol_embed[0].weight, gain=gain_leaky)

		nn.init.xavier_uniform_(self.key_layer.weight)
		nn.init.xavier_uniform_(self.query_layer.weight)
		nn.init.xavier_uniform_(self.attention_value_layer.weight)


		nn.init.xavier_uniform_(self.final_value_layer_1.weight, gain=gain_leaky)
		nn.init.xavier_uniform_(self.final_value_layer_2.weight, gain=gain_leaky)



	def forward(self, states, policies, actions):
		# EMBED STATES PREPROC
		states_embed_preproc = self.state_embed_preproc(states)
		# KEYS
		key_obs_preproc = self.key_layer(states_embed_preproc)
		# QUERIES
		query_obs_preproc = self.query_layer(states_embed_preproc)
		# WEIGHT
		weight_preproc = F.softmax(torch.matmul(query_obs_preproc,key_obs_preproc.transpose(1,2))/math.sqrt(self.d_k_obs),dim=-1)
		# ATTENTION VALUES
		attention_values_preproc = self.attention_value_layer_preproc(states_embed_preproc)
		attention_values_preproc = torch.matmul(weight_preproc, states_embed_preproc)


		# EMBED STATES
		states_embed = self.state_embed(attention_values_preproc)
		# KEYS
		key_obs = self.key_layer(states_embed)
		# QUERIES
		query_obs = self.query_layer(states_embed)
		# WEIGHT
		weight = F.softmax(torch.matmul(query_obs,key_obs.transpose(1,2))/math.sqrt(self.d_k_obs_act),dim=-1)
		ret_weight = weight

		obs_actions = torch.cat([states,actions],dim=-1)
		obs_policy = torch.cat([states,policies], dim=-1)
		obs_actions = obs_actions.repeat(1,self.num_agents,1).reshape(obs_actions.shape[0],self.num_agents,self.num_agents,-1)
		obs_policy = obs_policy.repeat(1,self.num_agents,1).reshape(obs_policy.shape[0],self.num_agents,self.num_agents,-1)
		obs_actions_policies = self.place_policies*obs_policy + self.place_actions*obs_actions
		# EMBED STATE ACTION POLICY
		obs_actions_policies_embed = self.state_act_pol_embed(obs_actions_policies)
		attention_values = self.attention_value_layer(obs_actions_policies_embed)
		attention_values = attention_values.repeat(1,self.num_agents,1,1).reshape(attention_values.shape[0],self.num_agents,self.num_agents,self.num_agents,-1)
		
		weight = weight.unsqueeze(-2).repeat(1,1,self.num_agents,1).unsqueeze(-1)
		weighted_attention_values = attention_values*weight
		node_features = torch.sum(weighted_attention_values, dim=-2)

		Value = F.leaky_relu(self.final_value_layer_1(node_features))
		Value = self.final_value_layer_2(Value)

		return Value, weight_preproc, ret_weight