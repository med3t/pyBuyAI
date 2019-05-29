import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import environment as env
import agent
import os
import itertools
from numpy import nan
import glob

def path_graphics(path_df,alpha=0.5,sub_plots=5,trial_intervals=None):
    """
    Function returns a graph of bidding paths taken by a single agent
    :param path_df: DataFrame containing actions taken by agent: must be of format produced by __main__ script
    :param alpha: line opacity for plot
    :param sub_plots: number of sub-plots
    :param trial_intervals: optional override for trial numbers per plot
    :return:
    """

    first = path_df['episode'].min()
    last = path_df['episode'].max()
    #cannot plot nan actions: replace these with -1
    path_df['bid'] = path_df['bid'].fillna(-1)

    if trial_intervals is None:
        breaks = list(range(first, last, int(round((last - first) / sub_plots)))) + [last]
        trial_intervals = [(breaks[i], breaks[i + 1]) for i in range(len(breaks) - 1)]

    fig, axs = plt.subplots(len(trial_intervals), 1, figsize=(15, 15), sharex=True, sharey=True,
                            tight_layout=True)

    if len(path_df) == 0:
        logging.error('Agent.get_path_graphics : agent has empty path_df')
        return (fig,axs)

    for i,intv in enumerate(trial_intervals):
        if path_df[path_df['episode']==min(intv)]['episode'].count() > 0:
            eps = path_df[path_df['episode']==min(intv)].head(1)['epsilon'].values[0]
        else:
            eps = np.nan
        axs[i].set_title('Trials {0} to {1} using epsilon = {2}'.format(intv[0],intv[1],eps))
        axs[i].set_xlabel('Bid period')
        axs[i].set_ylabel('Bid Amount')
        for t in range(intv[0],intv[1]):
            axs[i].plot(path_df[path_df['episode']==t]['bidding_round'],path_df[path_df['episode']==t]['bid'],alpha=alpha)

    fig.tight_layout()

    return (fig,axs)

def plot_grid_search_heatmap(param1,param2,dependent_var,df):
    """
    Plot output of grid search results
    """
    fig, axs = plt.subplots(1)
    df2 = df.pivot(param1,param2,dependent_var)
    sns.heatmap(df2,ax=axs)
    return (fig,axs)

def plot_final_bids_heatmap(bids_df,price_levels,title=''):
    """
    Plot heatmap of final bids placed
    :param bids_df:
    :param price_levels:
    :param title:
    :return:
    """
    bids_df['freq'] = 1
    prices = [-1] + list(range(price_levels))
    values = [list(bids_df[col].drop_duplicates()) for col in bids_df.columns]
    cartesian_values = list(itertools.product(prices, prices))
    cartesian_values = [(x) + (0,) for x in cartesian_values]
    df2 = pd.DataFrame(columns=bids_df.columns, data=cartesian_values)
    bids_df = pd.concat([bids_df,df2],axis=0)
    piv = bids_df.fillna(-1).pivot_table(index=bids_df.columns[0], columns=bids_df.columns[1], values='freq', aggfunc='sum')
    fig, axs = plt.subplots(1,figsize=(5,5))
    axs.set_title(title)
    sns.heatmap(piv, cmap="YlGnBu", ax=axs)
    axs.invert_yaxis()
    #axs.xaxis.tick_top()
    return (fig,axs)

def plot_rewards_per_episode(df,alpha=(0.7,1),colours=('tab:blue','tab:purple'),rolling_mean_window=10):
    """
    Line plot of rewards per episode
    :param df:
    :param alpha:
    :param colours:
    :param rolling_mean_window:
    :return:
    """
    df2 = df[df['bidding_round'] == max(df['bidding_round'])]
    df2['reward_ma'] = df2['reward'].rolling(rolling_mean_window,1).mean()

    fig, axs = plt.subplots(1)
    plt.plot(df2['episode'], df2['reward'], alpha=alpha[0], color='tab:blue', label='Reward')
    plt.plot(df2['episode'], df2['reward_ma'], alpha=alpha[1], color='tab:purple', label='{} game moving avg'.format(rolling_mean_window))
    plt.legend(loc='lower right')
    axs.set_xlabel('Bid Episode')
    axs.set_ylabel('Reward')
    return (fig,axs)

def plot_game_ending_periods(df,alpha=(0.7,1),colours=('tab:blue')):
    """
    Line plot of rewards per episode
    :param df:
    :param alpha:
    :param colours:
    :return:
    """
    
    df1 = df[["episode","bidding_round","is_game_ending"]]
    df1 = df1[df1["bidding_round"]>0]

    df2 = df1[df1["is_game_ending"]==True].groupby("episode")["bidding_round"].min().reset_index(name ='period_converged')
    fig, axs = plt.subplots(1)
    plt.plot(df2['episode'], df2['period_converged'], alpha=alpha[0], color='tab:blue', label='Converged Period ')
    plt.legend(loc='lower right')
    axs.set_xlabel('Bid Episode')
    axs.set_ylabel('Bid Period')
    return (fig,axs)

if __name__ == '__main__':

    #procedural code to produce various ad-hoc plots

    #get multiagent final bid heatmaps
    player0 = 'dd6daa2f-cc2c-40db-a42f-6e4ab6710f81'
    player1 = '8b68992c-c44b-475d-b3ee-c67a9b3b8ab6'
    folder = r"/Users/med/Desktop/MMI/MMI706/pyBuyAI-master/parameter_grid_search/results/alpha_gamma_gridsearch_10k_2bp"

    player0_serialised = os.path.join(folder,'player_0_'+player0+'.npy')
    #player0_path = os.path.join(folder,'player_0_'+game_id+'.hdf')
    player1_serialised = os.path.join(folder,'player_0_'+player1+'.npy')
    #player1_path = os.path.join(folder, 'player_1_' + game_id + '.hdf')

    player0 = agent.Player()
    player0.load_serialised_agent(player0_serialised)

    player1 = agent.Player()
    player1.load_serialised_agent(player1_serialised)

    #get results paths graphics for a good agent
    results_folder = r"/Users/med/Desktop/MMI/MMI706/pyBuyAI-master/parameter_grid_search2/results/epsilon_decay_gridsearch_10kg_4bp"
    game_id = '533e37b7-983c-4e45-93c7-7cb02d67048d'
    path_hdf = os.path.join(results_folder,'player_0_'+game_id+'.hdf')
    path_df = pd.read_csv(path_hdf,sep='#')
    path_df = path_df[path_df['episode']<3500]

    fig, axs = plot_rewards_per_episode(path_df, alpha=(0.5, 1), colours=('tab:blue', 'tab:purple'),
                                        rolling_mean_window=10)
    axs.set_title('Rewards per Episode')
    axs.set_xlabel('Episodes')
    axs.set_ylabel('Final Reward')


    # get grid search heatmaps
    #results_template = "/Users/med/Desktop/MMI/MMI706/pyBuyAI-master/parameter_grid_search2/results/*results*csv"
    results_files = glob.glob(results_template)

    df = pd.read_csv(results_files[0])
    for file in results_files[1:]:
        df = pd.concat([df,pd.read_csv(file)],axis=0)

    value_col = 'avg_rwd_100g'
    df['avg_rwd_100g'] = df['Avg Reward Vector'].apply(lambda x: x.split('_')[len(x.split('_'))-1])

    # now for epsilon decay
    grid_params = ['epsilon_decay_1','epsilon_decay_2']
    metric_cols = ['avg_rwd_100g','Period Converged','final_epsilon']
    metric_titles = ['Average Reward for Final 100 Games','Periods Until Q-convergence','Final Epsilon Value']
    cmap_list = ['Blues','Greens','Reds']

    #fig, axs = plt.subplots(1, 2, sharex=True, sharey=True, tight_layout=True)
    fig, axs = plt.subplots(1, 3, figsize=(10, 10), tight_layout=True)
    for i,metric in enumerate(metric_cols):
        df2 = df[grid_params+[metric]].astype(float)
        piv = df2.fillna(-1).pivot_table(
            index=grid_params[0],
            columns=grid_params[1],
            values=metric,
            aggfunc='mean',
            fill_value=0
        )
        #fig, axs = plt.subplots(1)
        #fig.suptitle('Grid search on: {}'.format(', '.join(grid_params)))
        axs[i].xaxis.tick_top()
        axs[i].set_title(metric_titles[i])
        sns.heatmap(piv, cmap=cmap_list[i], ax=axs[i], linewidths=1)

    df = pd.read_csv(results_files[0])
    for file in results_files[1:]:
        df = pd.concat([df,pd.read_csv(file)],axis=0)

    df['avg_rwd_100g'] = df['Avg Reward Vector'].apply(lambda x: x.split('_')[len(x.split('_')) - 1])

    grid_params = ['alpha','gamma']
    metric_cols = ['avg_rwd_100g','Period Converged']
    metric_titles = ['Average Reward for Final 100 Games','Periods Until Q-convergence','Final Epsilon Value']
    cmap_list = ['Blues','Greens','Reds']

    #fig, axs = plt.subplots(1, 2, sharex=True, sharey=True, tight_layout=True)
    fig, axs = plt.subplots(1, 2, figsize=(10, 10), tight_layout=True)
    for i,metric in enumerate(metric_cols):
        df2 = df[grid_params+[metric]].astype(float)
        piv = df2.fillna(-1).pivot_table(
            index=grid_params[0],
            columns=grid_params[1],
            values=metric,
            aggfunc='mean',
            fill_value=0
        )
        #fig, axs = plt.subplots(1)
        #fig.suptitle('Grid search on: {}'.format(', '.join(grid_params)))
        axs[i].xaxis.tick_top()
        axs[i].set_title(metric_titles[i])
        sns.heatmap(piv, cmap=cmap_list[i], ax=axs[i], linewidths=1)

        #plt.show()

    # now for epsilon decay
    grid_params = ['epsilon_decay_1','epsilon_decay_2']
    metric_cols = ['avg_rwd_100g','Period Converged','final_epsilon']
    metric_titles = ['Average Reward for Final 100 Games','Periods Until Q-convergence','Final Epsilon Value']
    cmap_list = ['Blues','Greens','Reds']

    #fig, axs = plt.subplots(1, 2, sharex=True, sharey=True, tight_layout=True)
    fig, axs = plt.subplots(1, 3, figsize=(10, 10), tight_layout=True)
    for i,metric in enumerate(metric_cols):
        df2 = df[grid_params+[metric]].astype(float)
        piv = df2.fillna(-1).pivot_table(
            index=grid_params[0],
            columns=grid_params[1],
            values=metric,
            aggfunc='mean',
            fill_value=0
        )
        axs[i].xaxis.tick_top()
        axs[i].set_title(metric_titles[i])
        sns.heatmap(piv, cmap=cmap_list[i], ax=axs[i], linewidths=1)

    #get multiagent final bid heatmaps
    game_id = '67835b1b-3e53-4263-b4c6-479d06894375'
    folder = "/Users/med/Desktop/MMI/MMI706/pyBuyAI-master/parameter_grid_search/results"
    player0_serialised = os.path.join(folder,'player_0_'+game_id+'.npy')
    #player0_path = os.path.join(folder,'player_0_'+game_id+'.hdf')
    player1_serialised = os.path.join(folder,'player_1_'+game_id+'.npy')
    #player1_path = os.path.join(folder, 'player_1_' + game_id + '.hdf')

    player0 = agent.Player()
    player0.load_serialised_agent(player0_serialised)

    player1 = agent.Player()
    player1.load_serialised_agent(player1_serialised)
    path_dataframes = [player0.path_df,player1.path_df]

    max_bid=int(round(player0.path_df['bid'].max()))

    update_mode = player0.q_update_mode
    n_bids = int(100)
    final_bids_df = env.get_last_x_bids_array(path_dataframes,n_bids)
    title = '{}, last {} games'.format(update_mode, n_bids)
    fig,axs = plot_final_bids_heatmap(final_bids_df,max_bid,title)
    #fig.savefig(player.get_serialised_file_name() + '_final_{}bids_heatmap.png'.format(str(n_bids)))
    #plt.show()

    #debug path values
    #look at final paths:
    player0.path_df.tail(10)

    #look at qmatrix and r matrix
    State = env.define_state()
