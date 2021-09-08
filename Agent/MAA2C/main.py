from maa2c import MAA2C

from multiagent.environment import MultiAgentEnv
import multiagent.scenarios as scenarios
import argparse

def make_env(scenario_name, benchmark=False):
	scenario = scenarios.load(scenario_name + ".py").Scenario()
	world = scenario.make_world()
	if benchmark:
		env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data, scenario.isFinished)
	else:
		env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, None, scenario.isFinished)
	return env


def run_file(dictionary):
	env = make_env(scenario_name=dictionary["env"],benchmark=False)
	ma_controller = MAA2C(env,dictionary)
	ma_controller.run()


if __name__ == '__main__':
	
	parser = argparse.ArgumentParser(description='PARTIAL REWARD DECOUPLING')

	# ENVIRONMENT NAMES: crossing_greedy/ crossing_fully_coop /  paired_by_sharing_goals/ crossing_partially_coop/ color_social_dilemma 
	parser.add_argument("--environment", default="paired_by_sharing_goals", type=str, help='Choose an environment: crossing_greedy/ crossing_fully_coop /  paired_by_sharing_goals/ crossing_partially_coop/ color_social_dilemma') 
	# EXPERIMENT TYPE: prd/ shared/ greedy
	parser.add_argument("--experiment_type", default="prd", type=str, help='Choose strategy out of the following: shared/ greedy/ prd')

	parser.add_argument("--save_model", default=False , type=bool, help='Set "True" to save models')
	parser.add_argument("--save_model_checkpoint", default=1000, type=int, , help='Enter number of episodes')
	parser.add_argument("--save_tensorboard_plot", default=False, type=bool, help='Set "True" to save tensorboard plots')
	parser.add_argument("--save_comet_ml_plot", default=False, type=bool, help='Set "True" to save comet_ml plots')
	parser.add_argument("--learn", default=True , type=bool, help='Set "True" to allow backprop')
	parser.add_argument("--save_gif", default=False , type=bool, , help='Set "True" to generate gifs')
	parser.add_argument("--gif_checkpoint", default= 1, type=int, help='Enter number of episodes')
	parser.add_argument("--norm_adv", default= False, type=bool, help='Set "True" to normalise advantages')
	parser.add_argument("--norm_rew", default= False, type=bool, help='Set "True" to normalise reward signal')
	parser.add_argument("--gae", default= True, type=bool, help='Set "True" to use GAE for Advantage calculations')

	parser.add_argument("--critic_dir", default="../../../critic_networks/", type=str, help='Path to save critic network models')
	parser.add_argument("--actor_dir", default="../../../actor_networks/", type=str, help='Path to save actor network models')
	parser.add_argument("--tensorboard_dir", default="../../../tensorboard/", type=str, help='Path to save tensorboard files')
	parser.add_argument("--gif_dir", default="../../../gifs/", type=str, help='Path to save gifs')

	parser.add_argument("--max_episodes", default=200000, type=int, help='Enter number of episodes to run')
	parser.add_argument("--max_time_steps", default=4, type=int, help='Enter number of timesteps to roll an episode')
	parser.add_argument("--value_lr", default=1e-2, type=float, help='Enter critic learning rate')
	parser.add_argument("--policy_lr", default=1e-3, type=float, help='Enter policy learning rate')
	parser.add_argument("--entropy_pen", default=1e-3, type=float, help='Enter entropy penalty coefficient')
	parser.add_argument("--gae_lambda", default=0.98, type=float, help='Enter lambda for gae')
	parser.add_argument("--gamma", default=0.99, type=float, help='Enter discount factor')
	parser.add_argument("--select_above_threshold", default=0.0, type=float, help='Enter starting threshold value for prd')
	parser.add_argument("--threshold_max", default=0.01, type=float, help='Enter final threshold value for prd')
	parser.add_argument("--steps_to_take", default=15000, type=int, help='Enter number of episodes to linearly increase the threshold from initial value to final value')
	parser.add_argument("--td_lambda", default= 0.8, type=float, help='Enter TD lambda value')
	parser.add_argument("--critic_loss_type", default= "TD_lambda", type=str, help='Choose target value type: MC/ TD_lambda/ TD_1')
	
	
	parser.add_argument("--load_models", default= False, type=bool, help='Set "True" to load models')
	parser.add_argument("--critic_saved_path", default= "../../../critic_networks/critic.pt", type=str, help='Enter critic model path')
	parser.add_argument("--actor_saved_path", default= "../../../actor.pt", type=str, help='Enter actor model path')

	parser.add_argument("--device", default="cuda", type=str, help='Choose device to train on: cpu/ cuda')
	
	

	arguments = parser.parse_args()

	env = make_env(scenario_name=argument.environment,benchmark=False)
	ma_controller = MAA2C(env,arguments)
	ma_controller.run()