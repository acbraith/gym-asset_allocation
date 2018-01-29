import gym
from gym import error, spaces, utils
from gym.utils import seeding

import numpy as np
import pandas as pd
import quandl
import errno
import os
import pickle
import datetime

class AssetAllocationEnv(gym.Env):
    # # # # # # # # # # # # # # # # # # # #
    # Overridden methods
    # # # # # # # # # # # # # # # # # # # #
    def __init__(self):
        # prices is a T x n array
        # first indexed by time, then by asset
        self.cols = ['Open','High','Low','Close','Volume']
        self._pull_price_data()
        n = len(self.asset_price_dfs)

        self.observation_space = spaces.Box(0, np.finfo('d').max, shape=(n,5))
        self.action_space = spaces.Box(0, 1, shape=(n))

        self._seed()
        self._reset()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        self.t += 1
        self._take_action(action)

        observation = self._get_observation()
        reward = self._get_reward()
        episode_over = self._get_over()
        info = {}

        return observation, reward, episode_over, info

    def _reset(self):

        # self.price is a dict, keys 'Open','High','Low','Close','Volume'
        self.price = self._get_price_data()

        self.T = self.price['Open'].shape[0] # final time
        self.n = self.price['Open'].shape[1] # number of assets

        # These are all for the START of day t
        self.allocation = np.zeros((self.T, self.n)) # fund allocation
        self.holdings = np.zeros((self.T, self.n)) # asset holdings
        self.cash = np.zeros((self.T))
        self.cash[0] = 100

        self.value = np.zeros((self.T)) # total value
        self.value[0] = np.dot(self.holdings[0], self.price['Open'][0]) + self.cash[0]

        self.t = 0 # current time


    def _render(self, mode='human', close=False):
        # TODO
        pass

    # # # # # # # # # # # # # # # # # # # #
    # Helper methods
    # # # # # # # # # # # # # # # # # # # #
    def _pull_price_data(self):
        # cache data for 1 day (because API calls limited to daily cap)
        directory = 'data/'
        filename = 'price_data.pkl'
        try:
            with open(directory+filename, 'rb') as f:
                timestamp, self.asset_price_dfs = pickle.load(f)
                if datetime.datetime.now() - timestamp > datetime.timedelta(days=1):
                    raise FileNotFoundError(
                        errno.ENOENT, os.strerror(errno.ENOENT), directory+filename)
        except FileNotFoundError:
            if not os.path.exists(directory):
                os.makedirs(directory)
            assets = ['TSE/9994', 'TSE/3443']
            self.asset_price_dfs = []
            for asset in assets:
                self.asset_price_dfs += [quandl.get(asset)]

            with open(directory+filename, 'wb') as f:
                pickle.dump((datetime.datetime.now(), self.asset_price_dfs), f)

    def _get_price_data(self, days=252):
        self.np_random.shuffle(self.asset_price_dfs)

        # get slices of length days from each dataframe
        asset_df_slices = []
        for asset_df in self.asset_price_dfs:
            if len(asset_df) > days:
                i_start = self.np_random.randint(0, len(asset_df)-days)
                asset_df_slice = asset_df.iloc[i_start:i_start+days]
                asset_df_slices += [asset_df_slice]

        # stack these together in to prices
        prices = {}
        for col in self.cols:
            prices[col] = np.column_stack([df[col] for df in asset_df_slices])

        return prices

    def _take_action(self, action):
        # Normalise action
        if action.sum() > 1:
            action /= action.sum()

        # update current allocation, value and holdings
        self.allocation[self.t] = action

        # keep holdings[t-1] from t-1 to t
        self.value[self.t] = np.dot(self.holdings[self.t-1], self.price['Open'][self.t]) + \
            self.cash[self.t-1]

        # calculate our new holdings
        self.holdings[self.t] = self.value[self.t] * self.allocation[self.t] / \
            self.price['Open'][self.t]
        self.cash[self.t] = (1-action.sum()) * self.value[self.t]

    def _get_observation(self):
        return np.array([self.price[col][self.t] for col in self.cols]).transpose()

    def _get_reward(self):
        return (self.value[self.t] - self.value[self.t-1]) / self.value[self.t-1]

    def _get_over(self):
        return self.t >= self.T
