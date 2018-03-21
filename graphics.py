import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import environment as env
import agent

def rewards_graphics(player_list, episodes, bid_periods, price_levels, num_players):
    """
    Plot a graph showing how each players reward rate varied throughout the process
    """
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1, ])
    for p in player_list:
        ax.plot(p.rewards_vector, label='Player {p}'.format(p=p.player_id))
    ax.set_xlabel('Thousand Episodes')
    ax.set_ylabel('Mean Reward per Episode')
    ax.set_title('Mean Rewards per Episode')
    ax.legend()
    fig.savefig(env.get_environment_level_file_name(episodes, bid_periods, price_levels, num_players)+'.png')
    return fig, ax

def path_graphics(path_df,alpha=0.5,sub_plots=5,trial_intervals=None):

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
    fig, axs = plt.subplots(1)
    df2 = df.pivot(param1,param2,dependent_var)
    sns.heatmap(df2,ax=axs)
    return (fig,axs)

def plot_final_bids_heatmap(bids_df):
    bids_df['freq'] = 1
    piv = bids_df.fillna(-1).pivot_table(index=bids_df.columns[0], columns=bids_df.columns[1], values='freq', aggfunc='sum')
    fig, axs = plt.subplots(1)
    sns.heatmap(piv, ax=axs)
    return (fig,axs)

def plot_rewards_per_episode(df):
    df2 = df[df['bidding_round'] == max(df['bidding_round'])]

    fig, axs = plt.subplots(1)
    plt.plot(df2['episode'], df2['reward'])
    #plt.show()

    return (fig,axs)

if __name__ == '__main__':
    file_name = r'./parameter_grid_search/results/grid_search_results_4ae7ca5f-5f8f-4301-b871-7c438d881617.hdf'
    df = pd.read_csv(file_name,sep='#')
    ept = 0.6
    df2 = df[(df['bid_periods'] == 5) & (df['epsilon_threshold'] == ept)]

    fig,axs=plot_grid_search_heatmap('epsilon_decay_1','epsilon_decay_2','Period Converged',df2)
    fig.suptitle('Epsilon Threshold: ept')

    fig, axs = plt.subplots(1, 3, figsize=(15, 15), sharex=True, sharey=True,
                            tight_layout=True)
    cbar_ax = fig.add_axes([.91, .3, .03, .4])

    for i,th in enumerate([0.2,0.4,0.6]):
        df2 = df[(df['bid_periods'] == 5) & (df['epsilon_threshold'] == th)]
        df3 = df2.pivot('epsilon_decay_1','epsilon_decay_2','Period Converged')
        im = sns.heatmap(df3,ax=axs[i],cbar=i == 0,cbar_ax=None if i else cbar_ax)
        axs[i].set_title('Epsilon threshold: {0}'.format(th))

    fig.suptitle('Episodes until convergence for epsilon decay configurations')

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)

    fig = plt.figure()
    plt.plot(df2['episode'], df2['reward'])
    plt.show()

    print('lala, end')