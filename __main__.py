import logging
import agent
import environment as env
import datetime as dt

def main():
    """
    Get params
    Instantiate env
    Instantiate players
    Run auction
    """
    # Game parameters
    episodes = 10000
    initial_state_random = True

    # Environment parameters
    bid_periods = 3
    price_levels = 10
    num_players = 1

    # Player parameters
    alpha = 0.8
    gamma = 0.5
    epsilon = 1
    epsilon_decay_1 = 0.99995
    epsilon_decay_2 = 0.999
    epsilon_threshold = 0.6
    agent_valuation = price_levels * 0.7

    S = env.get_possible_states(price_levels,num_players)

    # Initialise the players
    player_list = []
    for p in range(num_players):
        new_player = agent.Player(p, alpha, gamma, epsilon, epsilon_decay_1, epsilon_decay_2, epsilon_threshold, agent_valuation, S)
        new_player.set_r(S,bid_periods,agent_valuation)
        new_player.set_q()
        player_list = player_list + [new_player]

    for i in range(episodes):
        logging.info('Begin episode {0} of {1}'.format(i, episodes - 1))
        s = env.get_initial_state(S, initial_state_random)
        path = []
        for t in range(bid_periods):
            is_final_period = False if t < bid_periods - 1 else True
            logging.info('Begin bidding period {0}, final period: {1}, state: {2}'.format(t, is_final_period, S[s]))
            for p in player_list:
                a = p.select_action(t,s)
                path = path + [a]
                p.update_q(t, s, a, is_final_period)
                s = a
                p.update_epsilon()

        path = [(ac, S[ac]) for ac in path]
        logging.info('Auction complete, path taken: {0}'.format(path))
    logging.info('All episodes complete')


if __name__ == '__main__':
    logging.basicConfig(filename='bidding.log'.format(dt.datetime.strftime(dt.datetime.now(), '%Y%m%d-%H%M%S')),
                        format='%(asctime)s - %(levelname)s - %(message)s', level=logging.DEBUG)
    logging.info('Process start')
    main()
    logging.info('Process end')