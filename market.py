import numpy as np
import random
import logging
logger = logging.getLogger('log')

class Market(object):
    """
    A market represents the external state of the marketplace experienced by each market player
    It can be considered "the collection of all other market participants"

    Attributes:
        bid_price: The 'market' bid price
        ask_price: The 'market' ask price
    """

    def __init__(self, bid_price=None, ask_price=None, price_upper_bound=100, price_lower_bound=0):
        """
        Return a market object
        :param bid_price: bid price to initialise on - None will give random init
        :param ask_price: ask price to initialise on - None will give random init
        :param price_upper_bound:
        :param price_lower_bound:
        """
        self.bid_price = bid_price
        self.ask_price = ask_price
        self.price_upper_bound = price_upper_bound
        self.price_lower_bound = price_lower_bound

    def set_prices(self):
        self.set_bid_price()
        self.set_ask_price()

    def set_bid_price(self):
        self.bid_price = random.uniform(self.price_lower_bound,self.price_upper_bound)
        return self.bid_price

    def set_ask_price(self):
        self.ask_price = self.bid_price
        return self.ask_price


class MarketRandomWalk(Market):
    """
    Market with random walk
    """
    def __init__(self,walk_dist,walk_dist_params):
        super(MarketRandomWalk, self).__init__()
        self.walk_dist = walk_dist
        self.walk_dist_params = walk_dist_params

    def set_bid_price(self):
        """ If blank, initialise using Market price setting method: otherwise random walk from previous value"""
        if self.bid_price is None:
            self.bid_price = super(MarketRandomWalk,self).set_bid_price()
        else:
            price_change = self.walk_dist(*self.walk_dist_params)
            self.bid_price = self.bid_price + price_change
        return self.bid_price

    def set_ask_price(self):
        """ If blank, initialise using Market price setting method: otherwise random walk from previous value"""
        if self.ask_price is None:
            self.ask_price = super(MarketRandomWalk,self).set_ask_price()
        else:
            price_change = self.walk_dist(*self.walk_dist_params)
            self.ask_price = self.ask_price + price_change
        return self.ask_price

