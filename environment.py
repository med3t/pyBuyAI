import ast
import os
import sys
import pandas as pd
import collections
import itertools
import logging
import numpy as np
from math import ceil
import uuid

def get_possible_states(num_price_levels, num_players):
    """
    Function generates a list of all possible states S, where each state is a named tuple of state attributes
    :param num_price_levels: number of discrete prices
    :param num_players: number of players in auction
    :return: list of possible states S
    """
    prices = (np.nan,) + tuple(range(num_price_levels))
    bid_combinations = list(itertools.product(*[prices for p in range(num_players)]))

    State = define_state()
    S = [State(current_winner=np.nan, current_bids=(b)) for b in bid_combinations]
    S = [set_winner(s) for s in S]
    logging.info('Declared {0} possible states, examples: \n{1}'.format(len(S), S[0:3]))
    return S

def get_initial_state(S,initial_state_random):
    if initial_state_random:
        s = np.random.choice(len(S))
        logging.info('Randomly initialize state to {0}'.format({s:S[s]}))
    else:
        s = 0
        logging.info('Set initial state to {0}'.format({s:S[s]}))

    return s

def set_winner(state):
    """
    Function reads in a State named tuple and returns a new State tuple with the field 'current_winner' overwritten
    using the values in the current_bids attribute
    :param state:
    :return:
    """
    state_values = state._asdict()
    state_values['current_winner'] = get_winner_for_state(state)
    State = define_state()
    return State(**state_values)

def get_winner_for_state(state):
    """
    Function returns a value for the current winner, given the current_bids in the input state
    :param state: input State
    :return: output State with current_winner attribute overwritten
    """
    return get_winner(state.current_bids)

def get_winner(bids_tuple):
    if all([np.isnan(bid) for bid in bids_tuple]):
        return np.nan
    elif len(bids_tuple) == 1:
        return 0
    elif len(set(bids_tuple)) == 1:
        return np.nan
    else:
        for i, b in enumerate(bids_tuple):
            if b == np.nanmax(bids_tuple):
                return i

def define_state():
    return collections.namedtuple('State', 'current_winner current_bids')

def check_and_create_directory(dir):
    if not os.path.isdir(dir):
        try:
            os.mkdir(dir)
            logging.info('Created directory {0}'.format(dir))
        except Exception as ex:
            logging.error('Unable to create directory {0} \n Error:{1}'.format(dir,ex))
            return False

def calc_rewards_vector(path_df, reward_vector_interval:int):
    rewards_vector = []
    for k in range(0,int(ceil(path_df['episode'].max()/reward_vector_interval))):
        start_idx = int(k*reward_vector_interval)
        end_idx = int((k+1)*reward_vector_interval)
        rewards_vector.append(sum(path_df.iloc[start_idx:end_idx]['reward'])/reward_vector_interval)
    return rewards_vector

def get_last_x_bids_array(path_dataframes:list,n_games:int):
    """
    Function reads several path_df dataframes (intended to be 1 per player for a single game)
    and returns the final bids of the last n_games in an n_games*n_players array
    :param path_dataframes: path dataframes of players
    :return: results dataframe
    """
    all_bids_df = pd.DataFrame()
    for path_df in path_dataframes:
        player_id = 'player_'+str(path_df['player_id'].min())
        max_bid_rounds = path_df['bidding_round'].max()
        first_episode_to_include = path_df['episode'].max() - n_games
        first_episode_to_include if first_episode_to_include > 0 else 1

        rounds_df = path_df[
            (path_df['episode'] >= first_episode_to_include) & (path_df['bidding_round'] == max_bid_rounds)
        ].sort_values(['episode'],ascending=True).reset_index()

        #bids_df = pd.DataFrame(columns=[player_id],data=rounds_df['bid'],index=rounds_df.index)
        bids_df = pd.DataFrame(rounds_df['bid'])
        bids_df = bids_df.rename(columns={'bid':player_id})
        all_bids_df = pd.concat([all_bids_df,bids_df],axis=1)

    return all_bids_df

def get_results_summary(path_dataframes:list,reward_vector_interval=1000):
    """
    Function reads several path_df dataframes (intended to be 1 per player for a single game)
    and returns a results summary in results_df format
    :param path_dataframes: path dataframes of players
    :return: results dataframe
    """
    results_df = pd.DataFrame(columns=
                              ['Player ID', 'Total Episodes', 'Player Converged', 'Period Converged','Avg Reward','Avg Reward Vector',
                               'alpha','gamma','epsilon_decay_1','epsilon_decay_2','epsilon_threshold','final_epsilon']
                              )

    for path_df in path_dataframes:

        players = path_df['player_id'].drop_duplicates()
        if len(players) > 1:
            logging.warning('get_results_summary: multiple players in the same path_df! Results will be wrong!')
        player_id = players[0]

        total_episodes = path_df['episode'].max()
        convergence_status = path_df['q_converged'].tail(1).values[0]
        if convergence_status:
            period_converged = path_df[path_df['q_converged']]['episode'].min()
        else:
            period_converged = np.nan

        reward_per_episode = path_df.groupby(['episode'])['reward'].sum()
        avg_reward = round(sum(reward_per_episode)/total_episodes,2)

        reward_vector = calc_rewards_vector(path_df,reward_vector_interval)

        row_df = pd.DataFrame(columns=results_df.columns,index=[0])
        row_df['Player ID'] = player_id
        row_df['Total Episodes'] = total_episodes
        row_df['Player Converged'] = convergence_status
        row_df['Period Converged'] = period_converged
        row_df['Avg Reward'] = avg_reward
        row_df['Avg Reward Vector'] = '_'.join([str(round(x,4)) for x in reward_vector])

        for col in ['alpha','gamma','epsilon_decay_1','epsilon_decay_2','epsilon_threshold']:
            row_df[col] = path_df.tail(1)[col].values

        row_df['final_epsilon'] = path_df.tail(1)['epsilon'].values

        results_df = results_df.append(row_df)

        return results_df

def interpret_args(sys_args):
    if len(sys_args) == 1:
        return {}

    arg_list = sys.argv[1].split(',')
    arg_dict = {x.split(':')[0]: x.split(':')[1] for x in arg_list}
    print('Arg dict: {}'.format(arg_dict))
    for k in arg_dict:
        try:
            arg_dict[k] = ast.literal_eval(arg_dict[k])
        except Exception as ex:
            logging.error('interpret_args: unable to do literal eval for argument: {0} \n Error: {1}'.format(arg_dict[k], ex))

    return arg_dict

def get_game_id():
    file_name = str(uuid.uuid4())
    return file_name