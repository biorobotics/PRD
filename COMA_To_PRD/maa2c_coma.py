import os
import torch
import torch.nn.functional as F 
import torch.optim as optim
from torch.distributions import Categorical
import torch.autograd as autograd
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from a2c_agent_coma import A2CAgent
import datetime



class MAA2C:

	def __init__(self,env, dictionary):
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.env = env
		self.coma_version = dictionary["version"]
		self.gif = dictionary["gif"]
		self.save_model = dictionary["save_model"]
		self.save_model_checkpoint = dictionary["save_model_checkpoint"]
		self.save_tensorboard_plot = dictionary["save_tensorboard_plot"]
		self.learn = dictionary["learn"]
		self.gif_checkpoint = dictionary["gif_checkpoint"]
		self.num_agents = env.n
		self.num_actions = self.env.action_space[0].n
		self.date_time = f"{datetime.datetime.now():%d-%m-%Y}"
		self.env_name = dictionary["env"]
		self.policy_eval_dir = dictionary["policy_eval_dir"]

		self.max_episodes = dictionary["max_episodes"]
		self.max_time_steps = dictionary["max_time_steps"]

		self.weight_dictionary = {}

		for i in range(self.num_agents):
			agent_name = 'agent %d' % i
			self.weight_dictionary[agent_name] = {}
			for j in range(self.num_agents):
				agent_name_ = 'agent %d' % j
				self.weight_dictionary[agent_name][agent_name_] = 0


		# SAVE REWARDS 
		self.rewards = []
		self.rewards_mean_per_1000_eps = []
		self.timesteps = []
		self.timesteps_mean_per_1000_eps = []



		self.agents = A2CAgent(self.env, dictionary)


		if self.save_model:
			critic_dir = dictionary["critic_dir"]
			try: 
				os.makedirs(critic_dir, exist_ok = True) 
				print("Critic Directory created successfully") 
			except OSError as error: 
				print("Critic Directory can not be created") 
			actor_dir = dictionary["actor_dir"]
			try: 
				os.makedirs(actor_dir, exist_ok = True) 
				print("Actor Directory created successfully") 
			except OSError as error: 
				print("Actor Directory can not be created")

			self.policy_eval_dir = dictionary["policy_eval_dir"]
			try: 
				os.makedirs(self.policy_eval_dir, exist_ok = True) 
				print("Policy Eval Directory created successfully") 
			except OSError as error: 
				print("Policy Eval Directory can not be created") 

			# paths for models, tensorboard and gifs
			self.critic_model_path = critic_dir+str(self.date_time)+'VN_ATN_FCN_lr'+str(self.agents.value_lr)+'_PN_ATN_FCN_lr'+str(self.agents.policy_lr)+'_GradNorm0.5_Entropy'+str(self.agents.entropy_pen)+'_trace_decay'+str(self.agents.trace_decay)+"topK_"+str(self.agents.top_k)+"select_above_threshold"+str(self.agents.select_above_threshold)
			self.actor_model_path = actor_dir+str(self.date_time)+'_PN_ATN_FCN_lr'+str(self.agents.policy_lr)+'VN_SAT_FCN_lr'+str(self.agents.value_lr)+'_GradNorm0.5_Entropy'+str(self.agents.entropy_pen)+'_trace_decay'+str(self.agents.trace_decay)+"topK_"+str(self.agents.top_k)+"select_above_threshold"+str(self.agents.select_above_threshold)
			
			
		if self.save_tensorboard_plot:

			tensorboard_dir = dictionary["tensorboard_dir"]

			tensorboard_path = tensorboard_dir+str(self.date_time)+'VN_SAT_FCN_lr'+str(self.agents.value_lr)+'_PN_ATN_FCN_lr'+str(self.agents.policy_lr)+'_GradNorm0.5_Entropy'+str(self.agents.entropy_pen)+'_trace_decay'+str(self.agents.trace_decay)+"topK_"+str(self.agents.top_k)+"select_above_threshold"+str(self.agents.select_above_threshold)
			self.writer = SummaryWriter(tensorboard_path)


			
		if self.gif:
			gif_dir = dictionary["gif_dir"]
			try: 
				os.makedirs(gif_dir, exist_ok = True) 
				print("Gif Directory created successfully") 
			except OSError as error: 
				print("Gif Directory can not be created")
			self.gif_path = gif_dir+str(self.date_time)+'VN_SAT_FCN_lr'+str(self.agents.value_lr)+'_PN_ATN_FCN_lr'+str(self.agents.policy_lr)+'_GradNorm0.5_Entropy'+str(self.agents.entropy_pen)+"topK_"+str(self.agents.top_k)+"select_above_threshold"+str(self.agents.select_above_threshold)+'.gif'




	def get_actions(self,states):
		actions = self.agents.get_action(states)
		return actions


	def calculate_indiv_weights(self,weights):
		weights_per_agent = torch.sum(weights,dim=0) / weights.shape[0]

		for i in range(self.num_agents):
			agent_name = 'agent %d' % i
			for j in range(self.num_agents):
				agent_name_ = 'agent %d' % j
				self.weight_dictionary[agent_name][agent_name_] = weights_per_agent[i][j].item()



	def calculate_weights(self,weights):
		paired_agents_weight = 0
		paired_agents_weight_count = 0
		unpaired_agents_weight = 0
		unpaired_agents_weight_count = 0

		for k in range(weights.shape[0]):
			for i in range(self.num_agents):
				for j in range(self.num_agents):
					if self.num_agents-1-i == j:
						paired_agents_weight += weights[k][i][j]
						paired_agents_weight_count += 1
					else:
						unpaired_agents_weight += weights[k][i][j]
						unpaired_agents_weight_count += 1

		return round(paired_agents_weight.item()/paired_agents_weight_count,4), round(unpaired_agents_weight.item()/unpaired_agents_weight_count,4)




	def update(self,trajectory,episode):

		states_critic = torch.FloatTensor([sars[0] for sars in trajectory]).to(self.device)
		next_states_critic = torch.FloatTensor([sars[1] for sars in trajectory]).to(self.device)

		one_hot_actions = torch.FloatTensor([sars[2] for sars in trajectory]).to(self.device)
		one_hot_next_actions = torch.FloatTensor([sars[3] for sars in trajectory]).to(self.device)
		actions = torch.FloatTensor([sars[4] for sars in trajectory]).to(self.device)

		states_actor = torch.FloatTensor([sars[5] for sars in trajectory]).to(self.device)
		next_states_actor = torch.FloatTensor([sars[6] for sars in trajectory]).to(self.device)

		rewards = torch.FloatTensor([sars[7] for sars in trajectory]).to(self.device)
		dones = torch.FloatTensor([sars[8] for sars in trajectory]).to(self.device)
		
		if self.coma_version == 1 or self.coma_version == 2:
			value_loss,policy_loss,entropy,grad_norm_value,grad_norm_policy,weights,weight_policy = self.agents.update(states_critic,next_states_critic,one_hot_actions,one_hot_next_actions,actions,states_actor,next_states_actor,rewards,dones)
		elif self.coma_version == 3:
			value_loss_Q, value_loss_V, policy_loss, entropy, grad_norm_value_Q, grad_norm_value_V, grad_norm_policy, weights_Q, weights_V, weight_policy = self.agents.update(states_critic,next_states_critic,one_hot_actions,one_hot_next_actions,actions,states_actor,next_states_actor,rewards,dones)
		else:
			value_loss_V, policy_loss, entropy, grad_norm_value_V, grad_norm_policy, weights_V, weight_policy = self.agents.update(states_critic,next_states_critic,one_hot_actions,one_hot_next_actions,actions,states_actor,next_states_actor,rewards,dones)



		if self.save_tensorboard_plot:
			self.writer.add_scalar('Loss/Entropy loss',entropy.item(),episode)
			self.writer.add_scalar('Loss/Policy Loss',policy_loss.item(),episode)
			self.writer.add_scalar('Gradient Normalization/Grad Norm Policy',grad_norm_policy,episode)
			entropy_weights = -torch.mean(torch.sum(weight_policy * torch.log(torch.clamp(weight_policy, 1e-10,1.0)), dim=2))
			self.writer.add_scalar('Weights_Policy/Entropy', entropy_weights.item(), episode)
			
			if self.coma_version == 1 or self.coma_version == 2:
				self.writer.add_scalar('Loss/Value Loss',value_loss.item(),episode)
				self.writer.add_scalar('Gradient Normalization/Grad Norm Value',grad_norm_value,episode)
				paired_agent_avg_weight, unpaired_agent_avg_weight = self.calculate_weights(weights)
				self.writer.add_scalars('Weights/Average_Weights',{'Paired':paired_agent_avg_weight,'Unpaired':unpaired_agent_avg_weight},episode)
				entropy_weights = -torch.mean(torch.sum(weights * torch.log(torch.clamp(weights, 1e-10,1.0)), dim=2))
				self.writer.add_scalar('Weights/Entropy', entropy_weights.item(), episode)
			elif self.coma_version == 3:
				self.writer.add_scalar('Loss/Q-Value Loss',value_loss_Q.item(),episode)
				self.writer.add_scalar('Loss/V-Value Loss',value_loss_V.item(),episode)
				self.writer.add_scalar('Gradient Normalization/Grad Norm V-Value',grad_norm_value_V,episode)
				self.writer.add_scalar('Gradient Normalization/Grad Norm Q-Value',grad_norm_value_Q,episode)
				paired_agent_avg_weight, unpaired_agent_avg_weight = self.calculate_weights(weights_Q)
				self.writer.add_scalars('Weights_Q/Average_Weights',{'Paired':paired_agent_avg_weight,'Unpaired':unpaired_agent_avg_weight},episode)
				entropy_weights = -torch.mean(torch.sum(weights_Q * torch.log(torch.clamp(weights_Q, 1e-10,1.0)), dim=2))
				self.writer.add_scalar('Weights_Q/Entropy', entropy_weights.item(), episode)
				paired_agent_avg_weight, unpaired_agent_avg_weight = self.calculate_weights(weights_V)
				self.writer.add_scalars('Weights_V/Average_Weights',{'Paired':paired_agent_avg_weight,'Unpaired':unpaired_agent_avg_weight},episode)
				entropy_weights = -torch.mean(torch.sum(weights_V * torch.log(torch.clamp(weights_V, 1e-10,1.0)), dim=2))
				self.writer.add_scalar('Weights_V/Entropy', entropy_weights.item(), episode)
			else:
				self.writer.add_scalar('Loss/V-Value Loss',value_loss_V.item(),episode)
				self.writer.add_scalar('Gradient Normalization/Grad Norm V-Value',grad_norm_value_V,episode)
				paired_agent_avg_weight, unpaired_agent_avg_weight = self.calculate_weights(weights_V)
				self.writer.add_scalars('Weights_V/Average_Weights',{'Paired':paired_agent_avg_weight,'Unpaired':unpaired_agent_avg_weight},episode)
				entropy_weights = -torch.mean(torch.sum(weights_V * torch.log(torch.clamp(weights_V, 1e-10,1.0)), dim=2))
				self.writer.add_scalar('Weights_V/Entropy', entropy_weights.item(), episode)


	def split_states(self,states):

		states_critic = []
		states_actor = []
		for i in range(self.num_agents):
			states_critic.append(states[i][0])
			states_actor.append(states[i][1])

		states_critic = np.asarray(states_critic)
		states_actor = np.asarray(states_actor)

		return states_critic,states_actor



	def make_gif(self,images,fname,fps=10, scale=1.0):
		from moviepy.editor import ImageSequenceClip
		"""Creates a gif given a stack of images using moviepy
		Notes
		-----
		works with current Github version of moviepy (not the pip version)
		https://github.com/Zulko/moviepy/commit/d4c9c37bc88261d8ed8b5d9b7c317d13b2cdf62e
		Usage
		-----
		>>> X = randn(100, 64, 64)
		>>> gif('test.gif', X)
		Parameters
		----------
		filename : string
			The filename of the gif to write to
		array : array_like
			A numpy array that contains a sequence of images
		fps : int
			frames per second (default: 10)
		scale : float
			how much to rescale each image by (default: 1.0)
		"""

		# copy into the color dimension if the images are black and white
		if images.ndim == 3:
			images = images[..., np.newaxis] * np.ones(3)

		# make the moviepy clip
		clip = ImageSequenceClip(list(images), fps=fps).resize(scale)
		clip.write_gif(fname, fps=fps)




	def run(self):  
		for episode in range(1,self.max_episodes+1):
			states = self.env.reset()

			images = []
			final_timestep = self.max_time_steps

			states_critic,states_actor = self.split_states(states)

			trajectory = []
			episode_reward = 0
			for step in range(1,self.max_time_steps+1):

				if self.gif:
					# At each step, append an image to list
					images.append(np.squeeze(self.env.render(mode='rgb_array')))
					# Advance a step and render a new image
					with torch.no_grad():
						actions = self.get_actions(states_actor)
				else:
					actions = self.get_actions(states_actor)


				one_hot_actions = np.zeros((self.num_agents,self.num_actions))
				for i,act in enumerate(actions):
					one_hot_actions[i][act] = 1

				next_states,rewards,dones,info = self.env.step(actions)
				next_states_critic,next_states_actor = self.split_states(next_states)

				# Joint reward
				if self.coma_version == 1:
					rewards = [np.sum(rewards)]*self.num_agents

				# next actions
				next_actions = self.get_actions(next_states_actor)


				one_hot_next_actions = np.zeros((self.num_agents,self.num_actions))
				for i,act in enumerate(next_actions):
					one_hot_next_actions[i][act] = 1

				if self.coma_version == 1:
					episode_reward += np.mean(rewards)
				else:
					episode_reward += np.sum(rewards)

				if self.learn:
					if all(dones) or step == self.max_time_steps:
						final_timestep = step

						trajectory.append([states_critic,next_states_critic,one_hot_actions,one_hot_next_actions,actions,states_actor,next_states_actor,rewards,dones])
						print("*"*100)
						print("EPISODE: {} | REWARD: {} | TIME TAKEN: {} / {} \n".format(episode,np.round(episode_reward,decimals=4),step,self.max_time_steps))
						print("*"*100)

						if self.save_tensorboard_plot:
							self.writer.add_scalar('Reward Incurred/Length of the episode',step,episode)
							self.writer.add_scalar('Reward Incurred/Reward',episode_reward,episode)

						break
					else:
						trajectory.append([states_critic,next_states_critic,one_hot_actions,one_hot_next_actions,actions,states_actor,next_states_actor,rewards,dones])
						states_critic,states_actor = next_states_critic,next_states_actor
						states = next_states

				else:
					states_critic,states_actor = next_states_critic,next_states_actor
					states = next_states

			self.rewards.append(episode_reward)
			self.timesteps.append(final_timestep)
			if episode > self.save_model_checkpoint and episode%self.save_model_checkpoint:
				self.rewards_mean_per_1000_eps.append(sum(self.rewards[episode-self.save_model_checkpoint:episode])/self.save_model_checkpoint)
				self.timesteps_mean_per_1000_eps.append(sum(self.timesteps[episode-self.save_model_checkpoint:episode])/self.save_model_checkpoint)

			if not(episode%self.save_model_checkpoint) and self.save_model:
				if self.coma_version == 1 or self.coma_version == 2:
					torch.save(self.agents.critic_network.state_dict(), self.critic_model_path+'_epsiode'+str(episode)+'.pt')
					torch.save(self.agents.policy_network.state_dict(), self.actor_model_path+'_epsiode'+str(episode)+'.pt')
				elif self.coma_version == 3:
					torch.save(self.agents.critic_network_Q.state_dict(), self.critic_model_path+'_epsiode'+str(episode)+'_Q.pt')
					torch.save(self.agents.critic_network_V.state_dict(), self.critic_model_path+'_epsiode'+str(episode)+'_V.pt')
					torch.save(self.agents.policy_network.state_dict(), self.actor_model_path+'_epsiode'+str(episode)+'.pt')  
				else:
					torch.save(self.agents.critic_network_V.state_dict(), self.critic_model_path+'_epsiode'+str(episode)+'_V.pt')
					torch.save(self.agents.policy_network.state_dict(), self.actor_model_path+'_epsiode'+str(episode)+'.pt')  

			if self.learn:
				self.update(trajectory,episode) 
			elif self.gif and not(episode%self.gif_checkpoint):
				print("GENERATING GIF")
				self.make_gif(np.array(images),self.gif_path)


		np.save(os.path.join(self.policy_eval_dir,"paired_by_sharing_goals_reward_list"), np.array(self.rewards), allow_pickle=True, fix_imports=True)
		np.save(os.path.join(self.policy_eval_dir,"paired_by_sharing_goals_mean_rewards_per_1000_eps"), np.array(self.rewards_mean_per_1000_eps), allow_pickle=True, fix_imports=True)
		np.save(os.path.join(self.policy_eval_dir,"paired_by_sharing_goals_timestep_list"), np.array(self.timesteps), allow_pickle=True, fix_imports=True)
		np.save(os.path.join(self.policy_eval_dir,"paired_by_sharing_goals_mean_timestep_per_1000_eps"), np.array(self.timesteps_mean_per_1000_eps), allow_pickle=True, fix_imports=True)