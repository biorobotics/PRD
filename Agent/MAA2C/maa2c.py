from comet_ml import Experiment
import os
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from a2c_agent import A2CAgent
import datetime



class MAA2C:

	def __init__(self, env, arguments):
		self.device = arguments.device
		self.env = env
		self.save_gif = arguments.save_gif
		self.save_model = argumnets.save_model
		self.save_model_checkpoint = arguments.save_model_checkpoint
		self.save_tensorboard_plot = arguments.save_tensorboard_plot
		self.save_comet_ml_plot = arguments.save_comet_ml_plot
		self.learn = arguments.learn
		self.gif_checkpoint = arguments.gif_checkpoint
		self.num_agents = env.n
		self.num_actions = self.env.action_space[0].n
		self.date_time = f"{datetime.datetime.now():%d-%m-%Y}"
		self.env_name = arguments.environment

		self.max_episodes = arguments.max_episodes
		self.max_time_steps = arguments.max_time_steps

		self.experiment_type = arguments.experiment_type

		self.agents = A2CAgent(self.env, arguments)

		self.weight_dictionary = {}

		for i in range(self.num_agents):
			agent_name = 'agent %d' % i
			self.weight_dictionary[agent_name] = {}
			for j in range(self.num_agents):
				agent_name_ = 'agent %d' % j
				self.weight_dictionary[agent_name][agent_name_] = 0

		self.agent_group = {}
		for i in range(self.num_agents):
			agent_name = 'agent'+str(i)
			self.agent_group[agent_name] = 0

		if self.save_tensorboard_plot:
			tensorboard_dir = arguments.tensorboard_dir
			tensorboard_path = tensorboard_dir+str(self.date_time)+'VN_SAT_FCN_lr'+str(self.agents.value_lr)+'_PN_ATN_FCN_lr'+str(self.agents.policy_lr)+'_GradNorm0.5_Entropy'+str(self.agents.entropy_pen)+'_gae_lambda'+str(self.agents.gae_lambda)+"select_above_threshold"+str(self.agents.select_above_threshold)
			self.writer = SummaryWriter(tensorboard_path)

		if self.save_comet_ml_plot:
			self.comet_ml = Experiment("enter_your_key",project_name=self.env_name)
			self.comet_ml.log_parameters(dictionary)

		if self.save_model:
			critic_dir = arguments.critic_dir
			try: 
				os.makedirs(critic_dir, exist_ok = True) 
				print("Critic Directory created successfully") 
			except OSError as error: 
				print("Critic Directory can not be created") 
			actor_dir = arguments.actor_dir
			try: 
				os.makedirs(actor_dir, exist_ok = True) 
				print("Actor Directory created successfully") 
			except OSError as error: 
				print("Actor Directory can not be created")

			
			# paths for models, tensorboard and gifs
			self.critic_model_path = critic_dir+str(self.date_time)+'VN_ATN_FCN_lr'+str(self.agents.value_lr)+'_PN_ATN_FCN_lr'+str(self.agents.policy_lr)+'_GradNorm0.5_Entropy'+str(self.agents.entropy_pen)+'_gae_lambda'+str(self.agents.gae_lambda)+"select_above_threshold"+str(self.agents.select_above_threshold)
			self.actor_model_path = actor_dir+str(self.date_time)+'_PN_ATN_FCN_lr'+str(self.agents.policy_lr)+'VN_SAT_FCN_lr'+str(self.agents.value_lr)+'_GradNorm0.5_Entropy'+str(self.agents.entropy_pen)+'_gae_lambda'+str(self.agents.gae_lambda)+"select_above_threshold"+str(self.agents.select_above_threshold)
			

		if self.save_gif:
			gif_dir = arguments.gif_dir
			try: 
				os.makedirs(gif_dir, exist_ok = True) 
				print("Gif Directory created successfully") 
			except OSError as error: 
				print("Gif Directory can not be created")
			self.gif_path = gif_dir+str(self.date_time)+'VN_SAT_FCN_lr'+str(self.agents.value_lr)+'_PN_ATN_FCN_lr'+str(self.agents.policy_lr)+'_GradNorm0.5_Entropy'+str(self.agents.entropy_pen)+"select_above_threshold"+str(self.agents.select_above_threshold)+'.gif'



	def get_actions(self,states):
		actions = self.agents.get_action(states)
		return actions


	# FOR PAIRED AGENT ENVIRONMENTS
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


	# FOR OTHER ENV
	def calculate_indiv_weights(self,weights):
		weights_per_agent = torch.sum(weights,dim=0) / weights.shape[0]

		for i in range(self.num_agents):
			agent_name = 'agent %d' % i
			for j in range(self.num_agents):
				agent_name_ = 'agent %d' % j
				self.weight_dictionary[agent_name][agent_name_] = weights_per_agent[i][j].item()


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
		
		if self.env_name in ["crossing_fully_coop", "crossing_partially_coop"]:
			if self.experiment_type == "prd":
				value_loss, policy_loss, entropy, grad_norm_value, grad_norm_policy, weights_preproc, weights_post, weight_policy, agent_groups_over_episode, avg_agent_group_over_episode = self.agents.update(states_critic,next_states_critic,one_hot_actions,one_hot_next_actions,actions,states_actor,next_states_actor,rewards,dones)
			else:
				value_loss, policy_loss, entropy, grad_norm_value, grad_norm_policy, weights_preproc, weights_post, weight_policy = self.agents.update(states_critic,next_states_critic,one_hot_actions,one_hot_next_actions,actions,states_actor,next_states_actor,rewards,dones)
		else:
			if self.experiment_type == "prd":
				value_loss,policy_loss,entropy,grad_norm_value,grad_norm_policy,weights,weight_policy, agent_groups_over_episode, avg_agent_group_over_episode = self.agents.update(states_critic,next_states_critic,one_hot_actions,one_hot_next_actions,actions,states_actor,next_states_actor,rewards,dones)
			else:
				value_loss,policy_loss,entropy,grad_norm_value,grad_norm_policy,weights,weight_policy = self.agents.update(states_critic,next_states_critic,one_hot_actions,one_hot_next_actions,actions,states_actor,next_states_actor,rewards,dones)

		if self.save_tensorboard_plot:
			
			self.writer.add_scalar('Loss/Entropy loss',entropy.item(),episode)
			self.writer.add_scalar('Loss/Value Loss',value_loss.item(),episode)
			self.writer.add_scalar('Loss/Policy Loss',policy_loss.item(),episode)
			self.writer.add_scalar('Gradient Normalization/Grad Norm Value',grad_norm_value,episode)
			self.writer.add_scalar('Gradient Normalization/Grad Norm Policy',grad_norm_policy,episode)

			# self.calculate_indiv_weights(weights)
			# for i in range(self.num_agents):
			# 	agent_name = 'agent %d' % i
			# 	self.writer.add_scalars('Weights_Critic/Average_Weights/'+agent_name,self.weight_dictionary[agent_name],episode)

			# self.calculate_indiv_weights(weight_policy)
			# for i in range(self.num_agents):
			# 	agent_name = 'agent %d' % i
			# 	self.writer.add_scalars('Weights_Policy/Average_Weights/'+agent_name,self.weight_dictionary[agent_name],episode)
			

			# entropy_weights = -torch.mean(torch.sum(weight_policy * torch.log(torch.clamp(weight_policy, 1e-10,1.0)), dim=2))
			# self.writer.add_scalar('Weights_Policy/Entropy', entropy_weights.item(), episode)

			if self.experiment_type == "prd":
				for i in range(self.num_agents):
					agent_name = "agent"+str(i)
					self.agent_group[agent_name] = agent_groups_over_episode[i].item()
				self.writer.add_scalars('Reward Incurred/Group Size', self.agent_group, episode)
				self.writer.add_scalar('Reward Incurred/Avg Group Size', avg_agent_group_over_episode.item(), episode)

			
			if self.env_name in ["crossing_partially_coop", "crossing_fully_coop"]:
				# ENTROPY OF WEIGHTS
				entropy_weights = -torch.mean(torch.sum(weights_preproc * torch.log(torch.clamp(weights_preproc, 1e-10,1.0)), dim=2))
				self.writer.add_scalar('Weights_Critic/Entropy_Preproc', entropy_weights.item(), episode)
				entropy_weights = -torch.mean(torch.sum(weights_post * torch.log(torch.clamp(weights_post, 1e-10,1.0)), dim=2))
				self.writer.add_scalar('Weights_Critic/Entropy_Post', entropy_weights.item(), episode)
			else:
				# ENTROPY OF WEIGHTS
				entropy_weights = -torch.mean(torch.sum(weights * torch.log(torch.clamp(weights, 1e-10,1.0)), dim=2))
				self.writer.add_scalar('Weights_Critic/Entropy', entropy_weights.item(), episode)


		if self.save_comet_ml_plot:
			self.comet_ml.log_metric('Entropy_Loss',entropy.item(),episode)
			self.comet_ml.log_metric('Value_Loss',value_loss.item(),episode)
			self.comet_ml.log_metric('Policy_Loss',policy_loss.item(),episode)
			self.comet_ml.log_metric('Grad_Norm_Value',grad_norm_value,episode)
			self.comet_ml.log_metric('Grad_Norm_Policy',grad_norm_policy,episode)

			if self.experiment_type == "prd":
				for i in range(self.num_agents):
					agent_name = "agent"+str(i)
					self.comet_ml.log_metric('Group_Size_'+agent_name, agent_groups_over_episode[i].item(), episode)

				self.comet_ml.log_metric('Avg_Group_Size', avg_agent_group_over_episode.item(), episode)

			if self.env_name in ["crossing_fully_coop", "crossing_partially_coop"]:
				# ENTROPY OF WEIGHTS
				entropy_weights = -torch.mean(torch.sum(weights_preproc * torch.log(torch.clamp(weights_preproc, 1e-10,1.0)), dim=2))
				self.comet_ml.log_metric('Critic_Weight_Entropy_Preproc', entropy_weights.item(), episode)
				entropy_weights = -torch.mean(torch.sum(weights_post * torch.log(torch.clamp(weights_post, 1e-10,1.0)), dim=2))
				self.comet_ml.log_metric('Critic_Weight_Entropy_Post', entropy_weights.item(), episode)
			else:
				# ENTROPY OF WEIGHTS
				entropy_weights = -torch.mean(torch.sum(weights * torch.log(torch.clamp(weights, 1e-10,1.0)), dim=2))
				self.comet_ml.log_metric('Critic_Weight_Entropy', entropy_weights.item(), episode)



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

			states_critic,states_actor = self.split_states(states)

			trajectory = []
			episode_reward = 0
			episode_collision_rate = 0
			final_timestep = self.max_time_steps
			for step in range(1, self.max_time_steps+1):

				if self.save_gif:
					# At each step, append an image to list
					if not(episode%self.gif_checkpoint):
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

				# next actions
				next_actions = self.get_actions(next_states_actor)


				one_hot_next_actions = np.zeros((self.num_agents,self.num_actions))
				for i,act in enumerate(next_actions):
					one_hot_next_actions[i][act] = 1


				if self.env_name in ["crossing_greedy", "crossing_fully_coop", "crossing_partially_coop"]:
					collision_rate = [value[1] for value in rewards]
					rewards = [value[0] for value in rewards]
					episode_collision_rate += np.sum(collision_rate)

				episode_reward += np.sum(rewards)


				if self.learn:
					if all(dones) or step == self.max_time_steps:

						trajectory.append([states_critic,next_states_critic,one_hot_actions,one_hot_next_actions,actions,states_actor,next_states_actor,rewards,dones])
						print("*"*100)
						print("EPISODE: {} | REWARD: {} | TIME TAKEN: {} / {} \n".format(episode,np.round(episode_reward,decimals=4),step,self.max_time_steps))
						print("*"*100)

						final_timestep = step

						if self.save_tensorboard_plot:
							self.writer.add_scalar('Reward Incurred/Length of the episode',step,episode)
							self.writer.add_scalar('Reward Incurred/Reward',episode_reward,episode)

						if self.save_comet_ml_plot:
							self.comet_ml.log_metric('Episode_Length', step, episode)
							self.comet_ml.log_metric('Reward', episode_reward, episode)

						break
					else:
						trajectory.append([states_critic,next_states_critic,one_hot_actions,one_hot_next_actions,actions,states_actor,next_states_actor,rewards,dones])
						states_critic,states_actor = next_states_critic,next_states_actor
						states = next_states

				else:
					states_critic,states_actor = next_states_critic,next_states_actor
					states = next_states


			if not(episode%self.save_model_checkpoint) and episode!=0 and self.save_model:	
				torch.save(self.agents.critic_network.state_dict(), self.critic_model_path+'_epsiode'+str(episode)+'.pt')
				torch.save(self.agents.policy_network.state_dict(), self.actor_model_path+'_epsiode'+str(episode)+'.pt')  

			if self.learn:
				self.update(trajectory,episode) 
			elif self.save_gif and not(episode%self.gif_checkpoint):
				print("GENERATING GIF")
				self.make_gif(np.array(images),self.gif_path)