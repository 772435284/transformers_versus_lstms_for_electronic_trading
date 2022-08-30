'''
@Author: Yitao Qiu
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm


class backtestor():
    def __init__(self,mid_price,test_pred, volume=1,captial =1, trade_delay=5, trade_fee=0.00000):
        self.side = 0
        self.win = 0
        self.loss = 0
        self.under_fee = 0
        self.open_price = -9999
        self.volume = volume
        self.trade_delay = trade_delay
        self.trade_fee = trade_fee
        # Culmulative return
        self.cul_return = 1
        self.captial = 1 * self.volume
        self.his = [self.captial]
        self.captial_his = [self.captial]
        self.mid_price = mid_price
        self.test_pred = test_pred
        self.order_his = []
        



    def deal(self, price, side):
        # Calculate the profit
        fee = self.trade_fee * self.volume
        net_profit = -(price / self.open_price-1) * self.volume * side
        return_without_fee = 1 + net_profit
        profit = (return_without_fee - fee)
        multi_return = self.captial_his[-1] * profit
        self.captial_his.append(multi_return)
        self.his.append(profit)
        net_profit = profit - self.captial
        if net_profit > 0:
            self.win += 1
            if net_profit < fee:
                self.under_fee +=1
        else:
            self.loss += 1
        # Close the order
        self.side = 0
        self.order_his.append(self.side)


    def buy_signal(self, mid_price):
        if self.side == 0:
            # If there is no order, place an order
            self.open_price = mid_price
            self.side = 1
            self.order_his.append(self.side)
            self.captial_his.append(self.captial_his[-1])
        elif self.side == -1:
            # If there is an sell order, close it
            self.deal(mid_price,1)
        else:
            # If there is already a buy order, hold the order
            self.order_his.append(self.side)
            self.captial_his.append(self.captial_his[-1])
            

    def sell_signal(self, mid_price):
        if self.side == 0:
            # If there is no order,place an order
            self.open_price = mid_price
            self.side = -1
            self.order_his.append(self.side)
            self.captial_his.append(self.captial_his[-1])
        elif self.side == 1:
            # If there is an buy order, close it
            self.deal(mid_price,-1)
        else:
            # If there is already a sell order, hold the order
            self.order_his.append(self.side)
            self.captial_his.append(self.captial_his[-1])
    
    def close_signal(self, mid_price):
        if self.side == 1:
            self.deal(mid_price,-1)
        elif self.side == -1:
            self.deal(mid_price,1)
        else:
            self.order_his.append(self.side)
        

    def start_backtest(self):
        for i in tqdm(range(len(self.mid_price)-self.trade_delay)):
            if self.test_pred[i] == 2:
                # The price movement is going up, open buy order
                #print("buy")
                self.buy_signal(self.mid_price[i+self.trade_delay])
            elif self.test_pred[i] == 0:
                # The price movement is stationary, hold the order
                self.sell_signal(self.mid_price[i+self.trade_delay])
                
                #self.close_signal((mid_price[i+self.trade_delay]))
                #print("hold")
            elif self.test_pred[i] == 1:
                # The price movement is going down, open sell order
                #print("sell")
                #self.order_his.append(self.side)
                #self.close_signal((mid_price[i+self.trade_delay]))
                self.order_his.append(self.side)
                self.captial_his.append(self.captial_his[-1])