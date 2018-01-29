import gym
from gym import error, spaces, utils
from gym.utils import seeding

import numpy as np
import pandas as pd
import quandl

class AssetAllocationEnv(gym.Env):
    # # # # # # # # # # # # # # # # # # # # 
    # Overridden methods
    # # # # # # # # # # # # # # # # # # # # 
    def __init__(self):
        # prices is a T x n array
        # first indexed by time, then by asset
        self.price = self._get_price_data()

        self.T = self.price.shape[0] # final time
        self.n = self.price.shape[1] # number of assets

        self._reset()

    def _step(self, action):
        self.t += 1
        self._take_action(action)

        observation = self._get_observation()
        reward = self._get_reward()
        episode_over = self._get_over()
        info = {}

        return observation, reward, episode_over, info

    def _reset(self):
        self.allocation = np.zeros((self.T,self.n)) # fund allocation at time t
        self.holdings = np.zeros((self.T,self.n)) # actual holdings at time t
        self.cash = np.zeros((self.T))
        self.cash[0] = 100

        self.value = np.zeros((self.T)) # total value at time t
        self.value[0] = np.dot(self.holdings[0], self.price[0]) + self.cash[0]

        self.t = 0 # current time


    def _render(self, mode='human', close=False):
        # TODO
        pass

    # # # # # # # # # # # # # # # # # # # # 
    # Helper methods
    # # # # # # # # # # # # # # # # # # # # 
    def _get_price_data(self):
        # TODO pull data from internet
        days = 20
        assets = ['TSE/9994','TSE/3484']
        asset_price_lists = []
        for asset in assets:
            asset_df = quandl.get(asset)
            # get 1 year from asset_df
            i_start = np.random.randint(0,len(asset_df)-days)
            asset_price_lists += [asset_df.Close[i_start:i_start+days]]
        asset_prices = np.column_stack(asset_price_lists)
        return asset_prices

    def _take_action(self, action):
        if action.sum() > 1:
            action /= action.sum()
        # update current allocation, value and holdings
        self.allocation[self.t] = action

        # keep holdings[t-1] from t-1 to t
        self.value[self.t] = np.dot(self.holdings[self.t-1], self.price[self.t]) + self.cash[self.t-1]
        # TODO add transaction costs; need to think about how to deal with these
        # simple approach is to cut portfolio value by x%
        self.holdings[self.t] = np.floor(self.value[self.t] * self.allocation[self.t] / self.price[self.t])
        self.cash[self.t] = self.value[self.t] - np.dot(self.holdings[self.t], self.price[self.t])

    def _get_observation(self):
        return self.price[self.t]
    
    def _get_reward(self):
        return (self.value[self.t] - self.value[self.t-1]) / self.value[self.t-1]

    def _get_over(self):
        return self.t >= self.T
'''
data = np.round(np.random.random(size=(10,3))*5) + 10
aa = AssetAllocation(data)

a1 = np.array([.3,.4,.2])
a2 = np.array([.4,.2,.1])
a3 = np.array([.4,.2,.1])

print(data)

print("time = ",aa.t)
print("\tPrices    =",aa.price[aa.t])
print("\tValue     =",aa.value[aa.t])
print("\tAllocation=",aa.allocation[aa.t])
print("\tHoldings  =",aa.holdings[aa.t])
print("\tCash      =",aa.cash[aa.t])
aa.step(a1)
print("time = ",aa.t)
print("\tPrices    =",aa.price[aa.t])
print("\tValue     =",aa.value[aa.t])
print("\tAllocation=",aa.allocation[aa.t])
print("\tHoldings  =",aa.holdings[aa.t])
print("\tCash      =",aa.cash[aa.t])
aa.step(a2)
print("time = ",aa.t)
print("\tPrices    =",aa.price[aa.t])
print("\tValue     =",aa.value[aa.t])
print("\tAllocation=",aa.allocation[aa.t])
print("\tHoldings  =",aa.holdings[aa.t])
print("\tCash      =",aa.cash[aa.t])
aa.step(a3)
print("time = ",aa.t)
print("\tPrices    =",aa.price[aa.t])
print("\tValue     =",aa.value[aa.t])
print("\tAllocation=",aa.allocation[aa.t])
print("\tHoldings  =",aa.holdings[aa.t])
print("\tCash      =",aa.cash[aa.t])'''