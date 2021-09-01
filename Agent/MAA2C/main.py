from maa2c import MAA2C

from multiagent.environment import MultiAgentEnv
import multiagent.scenarios as scenarios

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

	# color_social_dilemma
	for i in range(1,6):
		extension = "run"+str(i)
		test_num = "color_social_dilemma_8_Agents_50K_policy_eval"
		env_name = "color_social_dilemma" 
		experiment_type = "prd_above_threshold_decay" # prd_above_threshold_decay, greedy, shared, prd_above_threshold_ascend, prd_above_threshold_l1_pen_decay

		dictionary = {
				"critic_dir": '../../../tests/'+test_num+'/models/'+env_name+'_'+experiment_type+'_'+extension+'/critic_networks/',
				"actor_dir": '../../../tests/'+test_num+'/models/'+env_name+'_'+experiment_type+'_'+extension+'/actor_networks/',
				"run_dir":'../../../tests/'+test_num+'/runs/'+env_name+'_'+experiment_type+'_'+extension+'/',
				"policy_eval_dir":'../../../tests/'+test_num+'/policy_eval/'+env_name+'_'+experiment_type+'_'+extension+'/',
				"gif_dir": '../../../tests/'+test_num+'/gifs/'+env_name+'_'+experiment_type+'_'+extension+'/',
				"env": env_name, 
				"test_num":test_num,
				"value_lr": 1e-3, 
				"policy_lr": 8e-5, #1e-4
				"entropy_pen": 8e-3, 
				"entropy_pen_min": 8e-3,
				"l1_pen": 0.0,
				"critic_entropy_pen": 0.0,
				"critic_loss_type": "TD_lambda", #MC
				"gamma": 0.99, 
				"trace_decay": 0.98,
				"lambda": 0.8, #0.8
				"select_above_threshold": 0.01,
				"threshold_min": 0.0, #0.0001
				"threshold_max": 0.0,
				"steps_to_take": 15000,
				"l1_pen_min": 0.0,
				"l1_pen_steps_to_take": 0,
				"top_k": 0,
				"gif": False,
				"save_model": True,
				"eval_policy": True,
				"save_model_checkpoint": 100,
				"save_tensorboard_plot": False,
				"save_comet_ml_plot": True,
				"learn":True,
				"max_episodes": 50000,
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


	# crossing_greedy/ crossing_fully_coop /  paired_by_sharing_goals/ crossing_partially_coop
	# for i in range(1,6):
	# 	extension = "run"+str(i)
	# 	test_num = "team_crossing_8_agents_pen_non_colliding_team_members_dualgat_policy_eval" #crossing_8_agents_pen_non_colliding_agents_policy_eval
	# 	env_name = "crossing_partially_coop"
	# 	experiment_type = "prd_above_threshold_ascend" # prd_above_threshold_decay_episodic, greedy, shared

	# 	dictionary = {
	# 			"critic_dir": '../../../tests/'+test_num+'/models/'+env_name+'_'+experiment_type+'_'+extension+'/critic_networks/',
	# 			"actor_dir": '../../../tests/'+test_num+'/models/'+env_name+'_'+experiment_type+'_'+extension+'/actor_networks/',
	# 			"run_dir":'../../../tests/'+test_num+'/runs/'+env_name+'_'+experiment_type+'_'+extension+'/',
	# 			"gif_dir": '../../../tests/'+test_num+'/gifs/'+env_name+'_'+experiment_type+'_'+extension+'/',
	# 			"policy_eval_dir":'../../../tests/'+test_num+'/policy_eval/'+env_name+'_'+experiment_type+'_'+extension+'/',
	# 			"env": env_name, 
	# 			"test_num":test_num,
	# 			"extension":extension,
	# 			"value_lr": 1e-3, 
	# 			"policy_lr": 1e-4,
	# 			"entropy_pen": 8e-3, 
	# 			"entropy_pen_min": 1e-3,
	# 			"l1_pen": 0.0,
	# 			"critic_entropy_pen": 0.0,
	# 			"critic_loss_type": "TD_lambda",
	# 			"gamma": 0.99, 
	# 			"trace_decay": 0.98,
	# 			"lambda": 0.8, #0.8
	# 			"select_above_threshold": 0.0,
	# 			"threshold_min": 0.0, 
	# 			"threshold_max": 0.01,
	# 			"steps_to_take": 15000, 
	# 			"l1_pen_min": 0.0,
	# 			"l1_pen_steps_to_take": 0,
	# 			"top_k": 0,
	# 			"gif": False,
	# 			"eval_policy": True,
	# 			"save_model": True,
	# 			"save_model_checkpoint": 1000,
	# 			"save_tensorboard_plot": False,
	# 			"save_comet_ml_plot": True,
	# 			"learn":True,
	# 			"max_episodes": 200000,
	# 			"max_time_steps": 100,
	# 			"experiment_type": experiment_type,
	# 			"gif_checkpoint":1,
	# 			"gae": True,
	# 			"norm_adv": False,
	# 			"norm_rew": False,
	# 		}
	# 	env = make_env(scenario_name=dictionary["env"],benchmark=False)
	# 	ma_controller = MAA2C(env,dictionary)
	# 	ma_controller.run()


# export CUDA_VISIBLE_DEVICES=1
# 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