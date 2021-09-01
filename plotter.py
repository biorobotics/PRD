import numpy as np
import os
import matplotlib.pyplot as plt

data_dir = 'data'
# exp_names = ['policy_eval_v1','policy_eval_v2','policy_eval_v3','policy_eval_v4','policy_eval_v5','policy_eval_v6']

# exp_names = ['policy_eval_v1','policy_eval_v4','policy_eval_v5','policy_eval_v6']

###############################################
exp_names = ['crossing_greedy','crossing_prd_above_threshold','crossing_shared']
legend_entries =['Greedy','PRD','Shared']
trim_inds = [5100,6800,6800]
# trim_inds =[5100,10000,6800]
# trim_inds = [None,None,None]
xlabel = 'Episodes'
ylabel = 'Mean Episode Reward'
title = 'Crossing, Penalty on Collision, Reward vs. Episodes (Higher is better)'
filename = 'crossing_8_Agents_pen_colliding_agents_policy_eval'
legend_loc = 'lower right'
################################################

def trim(list_of_arrays,trim_ind=None):
    lengths = []
    for a in list_of_arrays:
        lengths.append(a.shape[0])
    
    min_length = np.min(lengths)
    trimmed = []
    for a in list_of_arrays:
        if trim_ind is None:
            trimmed.append(a[:min_length])
        else:
            trimmed.append(a[:trim_ind])

    return trimmed

def plot_with_var(data_mat):

    means = np.mean(data_mat,axis=0)
    stds  = np.std(data_mat,axis=0)

    ucb = means + stds
    lcb = means - stds
    
    x = np.arange(data_mat.shape[-1])

    plt.plot(x,means)
    plt.fill_between(x,lcb,ucb,alpha=.5)

plt.figure()

# trim_ind = 5500

for i,exp_name in enumerate(exp_names):
    exp_dir = data_dir + '/' + exp_name
    runs = os.listdir(exp_dir)
    rewards_list = []
    for run in runs:
        # print('run: ', run)

        if os.path.isdir(exp_dir + '/'+run):
            metrics = os.listdir(exp_dir + '/'+run)
            # print('metrics: ', metrics)
            for metric in metrics:
                if 'mean_reward' in metric:
                    # load it
                    data_file = exp_dir + '/'+run + '/' + metric
                    print('data_file: ', data_file)

                    rewards = np.load(data_file)
                    rewards_list.append(rewards)
                    # print('rewards.shape: ',rewards.shape)

    rewards_mat = np.stack(trim(rewards_list,trim_ind=trim_inds[i]))

    plot_with_var(rewards_mat)

    print('rewards_mat.shape: ', rewards_mat.shape)


plt.legend(legend_entries,loc=legend_loc)
plt.xlabel(xlabel)
plt.ylabel(ylabel)
plt.title(title,fontdict={'fontsize':12,'fontweight':'bold'})
# plt.tight_layout()

plt.savefig(filename,bbox_inches='tight')
