# from maa2c import MAA2C
from maa2c_coma import MAA2C

from multiagent.environment import MultiAgentEnv
# from multiagent.scenarios.simple_spread import Scenario
import multiagent.scenarios as scenarios
import torch 
import numpy as np

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


def run_file(dictionary):
	env = make_env(scenario_name=dictionary["env"],benchmark=False)
	ma_controller = MAA2C(env,dictionary)
	ma_controller.run()



if __name__ == '__main__':
	for i in range(1,6):
		extension = "run"+str(i)
		version = 6
		env_name = "paired_by_sharing_goals" # paired_by_sharing_goals, color_social_dilemma, crossing
		experiment_type = "coma_v"+str(version)

		# VALUE LR: v1:1e-2, v2:1e-2, v3:1e-2, v4:1e-2, v5:1e-2, v6:1e-2; actual: 1e-3
		# POLICY LR: v1:1e-4, v2:5e-4, v3:1e-3, v4:1e-3, v5:1e-3, v6:1e-3; actual: 5e-4
		# ENTROPY: v1:8e-4, v2:0.0, v3:8e-3, v4:8e-3, v5:8e-3, v6:8e-3; actual: 8e-3
		
		dictionary = {
			"critic_dir": '../../tests/'+str(version)+'/models/'+env_name+'_'+experiment_type+'_'+extension+'/critic_networks/',
			"actor_dir": '../../tests/'+str(version)+'/models/'+env_name+'_'+experiment_type+'_'+extension+'/actor_networks/',
			"tensorboard_dir":'../../tests/'+str(version)+'/runs/'+env_name+'_'+experiment_type+'_'+extension+'/',
			"gif_dir": '../../tests/'+str(version)+'/gifs/'+env_name+'_'+experiment_type+'_'+extension+'/',
			"policy_eval_dir": '../../tests/'+str(version)+'/policy_eval/'+env_name+'_'+experiment_type+'_'+extension+'/',
			"env": env_name, 
			"version": version,
			"value_lr": 1e-3, 
			"policy_lr": 5e-4, 
			"entropy_pen": 8e-3, 
			"critic_loss_type": "TD_lambda",
			"gamma": 0.99, 
			"trace_decay": 0.98,
			"select_above_threshold": 0.01,
			"top_k": 0,
			"gif": False,
			"save_model": True,
			"save_model_checkpoint": 1000,
			"save_tensorboard_plot": True,
			"learn":True,
			"max_episodes": 200000,
			"max_time_steps": 100,
			"experiment_type": experiment_type,
			"gif_checkpoint":1,
			"gae": True,
			"norm_adv": False,
			"norm_rew": False,
		}

		env = make_env(scenario_name=dictionary["env"],benchmark=False)
		ma_controller = MAA2C(env,dictionary)
		ma_controller.run()