import gym
import numpy as np
from gym import spaces
import enum
import random
import torch


class Actions(enum.Enum):
    hold = 0
    buy = 1
    sell = 2


class StockTradingEnv_1D(gym.Env):
    def __init__(self,
                 df,
                 initial_amount,
                 state_space,  # 30+2
                 tickers,
                 max_step
                 ):

        self.df = df
        self.initial_amount = initial_amount
        self.max_step = max_step
        self.buy_days = 0

        self.state_space = state_space
        self.action_space = spaces.Discrete(n=len(Actions))

        self.terminal = False

        self.day = 9
        self.tickers = tickers
        self.tic_idx = 0
        self.tic = self.tickers[self.tic_idx]
        self.end_total_asset = 0

        self.data = self.df.loc[(self.df.index >= self.day - 9) & (self.df.index <= self.day)
                                & (self.df['tic'] == self.tic), :]
        self.shares = 0
        self.amount = self.initial_amount

        self.state = np.zeros(self.state_space)

        self.reward = 0
        self.trades = 0
        self.episode = 0

        self.buy_price = 0
        self.sell_price = 0
        self.steps = 0
        self.baseline = 0
        self.daily_cum_rewards = 0

        self.balance_rate = 0
        self.close_rate = 0

    def reset(self):
        # initiate state
        self.reward = 0
        self.trades = 0
        self.daily_cum_rewards = 0
        self.steps = 0
        self.shares = 0
        self.buy_price = 0
        self.sell_price = 0
        self.terminal = False

        self.amount = self.initial_amount
        self.episode += 1

        self.day = random.randint(9, len(self.df[self.df["tic"] == self.tic]) - 90)  # 任选时间段
        # self.day = 9
        self.data = self.df.loc[(self.df.index >= self.day - 9) & (self.df.index <= self.day)
                                & (self.df['tic'] == self.tic), :]
        self.baseline = self.data.iloc[:9, 2].mean()

        self.state = self._initiate_state()

        return self.state

    def _initiate_state(self):
        state = np.zeros(self.state_space)
        for j in range(10):
            state[3*j] = self.data.iloc[j, 4]
            state[3*j + 1] = self.data.iloc[j, 5]
            state[3*j + 2] = self.data.iloc[j, 6]

        if self.shares != 0:
            state[-2] = 1
            state[-1] = (self.data.iloc[9, 2] / self.buy_price) - 1.0

        # state[-1] = self.steps

        return state.astype(np.float32)

    def _update_state(self):
        state = self._initiate_state()

        return state

    def action_mask(self):
        end_cash = self.amount
        end_market_value = self.shares * self.data.iloc[9, 2]
        end_total_asset = end_cash + end_market_value
        self.daily_cum_rewards = end_total_asset - self.initial_amount
        self.close_rate = (self.data.iloc[9, 2] - self.baseline) / self.baseline
        self.balance_rate = self.daily_cum_rewards / self.initial_amount

        if self.shares != 0:
            # if self.steps >= self.max_step - 1 or self.close_rate <= -0.03:
            #     mask = torch.tensor([0, 0, 1])
            # else:
            mask = torch.tensor([1, 0, 1])

        else:
            # if self.steps >= self.max_step - 1 or self.close_rate <= -0.03:
            #     mask = torch.tensor([1, 0, 0])
            # else:
            mask = torch.tensor(([1, 1, 0]))

        return mask

    def step(self, action):

        self.reward = 0
        if action == 1:
            self.buy_price = self.data.iloc[9, 2]
            buy_num_shares = (self.amount // self.buy_price) // 100 * 100  # all in, 100的整数倍
            self.shares += buy_num_shares
            buy_amount = self.buy_price * buy_num_shares
            self.amount -= buy_amount
            self.trades += 1
            self.buy_days += 1

            self.reward -= 0.001 * buy_amount

        elif action == 2:
            self.sell_price = self.data.iloc[9, 2]
            sell_num_shares = self.shares  # all out
            sell_amount = self.sell_price * sell_num_shares
            self.amount += sell_amount
            self.shares -= sell_num_shares
            self.trades += 1

            self.reward += sell_num_shares * (self.sell_price - self.buy_price)
            self.reward -= 0.001 * sell_amount


        # update next state
        self.day += 1
        self.steps += 1

        if self.day >= len(self.df[self.df['tic'] == self.tic]):
            self.terminal = True
        else:
            self.data = self.df.loc[(self.df.index >= self.day - 9) & (self.df.index <= self.day)
                                    & (self.df['tic'] == self.tic), :]
            self.state = self._update_state()

        return self.state, self.reward, self.trades, self.terminal, self.daily_cum_rewards, self.steps

class StockTradingEnv(gym.Env):
    def __init__(self,
                 df,
                 initial_amount,
                 state_space,  # [8, 90]
                 tickers,
                 max_step
                 ):

        self.df = df
        self.initial_amount = initial_amount

        self.state_space = state_space
        self.action_space = spaces.Discrete(n=len(Actions))
        self.observation_space = spaces.Box(
            low=-1, high=1, shape=self.state_space, dtype=np.float32
        )

        self.terminal = False

        self.day = 9
        self.tickers = tickers
        self.tic_idx = 0
        self.tic = self.tickers[self.tic_idx]
        self.max_step = max_step
        self.end_total_asset = 0

        self.data = self.df.loc[(self.df.index >= self.day - 9) & (self.df.index <= self.day)
                                & (self.df['tic'] == self.tic), :]
        self.shares = 0
        self.amount = self.initial_amount

        self.state = np.zeros(self.state_space)

        self.reward = 0
        self.trades = 0
        self.episode = 0
        self.real_ep_reward = 0

        self.buy_price = 0
        self.sell_price = 0
        self.steps = 0
        self.baseline = 0
        self.daily_cum_rewards = 0

        self.balance_rate = 0
        self.close_rate = 0

    def reset(self):
        # initiate state
        self.reward = 0
        self.trades = 0
        self.real_ep_reward = 0
        self.daily_cum_rewards = 0
        self.steps = 0
        self.shares = 0
        self.buy_price = 0
        self.sell_price = 0
        self.terminal = False

        self.amount = self.initial_amount
        self.episode += 1

        self.tic_idx = random.randint(0, len(self.tickers) - 1)
        self.tic = self.tickers[self.tic_idx]  # 任选tic
        while len(self.df[self.df["tic"] == self.tic]) < 90:  # 27+90+59
            self.tic_idx = random.randint(0, len(self.tickers) - 1)
            self.tic = self.tickers[self.tic_idx]

        self.day = random.randint(9, len(self.df[self.df["tic"] == self.tic]) - 90)  # 任选时间段
        self.data = self.df.loc[(self.df.index >= self.day - 9) & (self.df.index <= self.day)
                                & (self.df['tic'] == self.tic), :]
        # self.baseline = self.data.iloc[79:89, 2].mean()

        cnt = 0
        data = self.df.loc[(self.df.index >= self.day - 9) &
                           (self.df.index <= self.day + self.max_step + 30 - 1) &
                           (self.df['tic'] == self.tic)
        , :]

        date_diff = data['date'].diff().dt.days

        while (date_diff > 30).any() or self.data.iloc[9, 2] < self.baseline * 0.97:  # 停牌日期过长或第一天跌破3%，重新选择片段
            cnt += 1
            self.day = random.randint(9, len(self.df[self.df["tic"] == self.tic]) - 90)  # 任选时间段
            self.data = self.df.loc[(self.df.index >= self.day - 9) & (self.df.index <= self.day)
                                    & (self.df['tic'] == self.tic), :]
            # self.baseline = self.data.iloc[79:89, 2].mean()
            data = self.df.loc[(self.df.index >= self.day - 9) &
                               (self.df.index <= self.day + self.max_step + 30 - 1) &
                               (self.df['tic'] == self.tic)
            , :]
            date_diff = data['date'].diff().dt.days

            if cnt > 5:
                self.tic_idx = random.randint(0, len(self.tickers) - 1)
                self.tic = self.tickers[self.tic_idx]
                self.day = random.randint(9, len(self.df[self.df["tic"] == self.tic]) - 90)  # 任选时间段
                self.data = self.df.loc[(self.df.index >= self.day - 9) & (self.df.index <= self.day)
                                        & (self.df['tic'] == self.tic), :]
                # self.baseline = self.data.iloc[79:89, 2].mean()
                data = self.df.loc[(self.df.index >= self.day - 9) &
                                   (self.df.index <= self.day + self.max_step + 30 - 1) &
                                   (self.df['tic'] == self.tic)
                , :]
                date_diff = data['date'].diff().dt.days
                cnt = 0

        self.state = self._initiate_state()

        return self.state

    def _initiate_state(self):
        state = np.zeros(self.state_space)

        for j in range(10):
            state[:3, j] = list(self.data.iloc[j, 4: 7])

        if self.shares != 0:
            state[3, :] = 1
            state[4, :] = (self.data.iloc[9, 2] / self.buy_price) - 1.0

        # if self.shares > 0:
        #     flag = 1
        # else:
        #     flag = 0
        # state[8, :] = flag  # 是否持股
        # state[9, :] = (self.data.iloc[89, 2] - self.baseline) / self.baseline  # 当日价格较baseline涨跌幅
        # state[10, :] = self.daily_cum_rewards / self.initial_amount  # 累积利润百分比

        return state.astype(np.float32)

    def _update_state(self):
        state = self._initiate_state()

        return state

    def action_mask(self):
        end_cash = self.amount
        end_market_value = self.shares * self.data.iloc[9, 2]
        end_total_asset = end_cash + end_market_value
        self.daily_cum_rewards = end_total_asset - self.initial_amount
        # self.close_rate = (self.data.iloc[9, 2] - self.baseline) / self.baseline
        # self.balance_rate = self.daily_cum_rewards / self.initial_amount

        if self.shares != 0:
            mask = torch.tensor([1, 0, 1])
        else:
            mask = torch.tensor(([1, 1, 0]))

        return mask

    def step(self, action):

        if action == 1:
            self.buy_price = self.data.iloc[9, 2]
            buy_num_shares = (self.amount // self.buy_price) // 100 * 100  # all in, 100的整数倍
            self.shares += buy_num_shares
            buy_amount = self.buy_price * buy_num_shares
            self.amount -= buy_amount
            self.trades += 1

            self.reward = 0

        elif action == 2 and self.shares != 0:
            self.sell_price = self.data.iloc[9, 2]
            sell_num_shares = self.shares  # all out
            sell_amount = self.sell_price * sell_num_shares
            self.amount += sell_amount
            self.shares -= sell_num_shares
            self.trades += 1

            self.reward = sell_num_shares * (self.sell_price - self.buy_price)

        else:
            self.reward = 0

        # update next state
        self.day += 1
        self.steps += 1

        if (self.steps >= self.max_step and self.shares == 0) \
                or self.day >= len(self.df[self.df['tic'] == self.tic]):
            self.terminal = True
        else:
            self.data = self.df.loc[(self.df.index >= self.day - 9) & (self.df.index <= self.day)
                                    & (self.df['tic'] == self.tic), :]
            self.state = self._update_state()

        return self.state, self.reward, self.trades, self.terminal, self.daily_cum_rewards, self.steps


class StockTradingEnv_test(gym.Env):
    def __init__(self,
                 df,
                 initial_amount,
                 state_space,  # [30, 8, 90]
                 tickers,
                 cycle_days,
                 delay_days
                 ):

        self.df = df
        self.initial_amount = initial_amount
        self.day = 89 + 27
        self.state_space = state_space
        self.action_space = spaces.Discrete(n=len(Actions))
        self.observation_space = spaces.Box(
            low=-1, high=1, shape=self.state_space, dtype=np.float32
        )

        self.terminal = False

        self.tickers = tickers
        self.tic_cnt = 0
        self.tic = tickers[self.tic_cnt]
        self.cycle_days = cycle_days
        self.delay_days = delay_days
        self.cnt = 0
        self.end_total_asset = 0

        # initalize state
        self.data = self.df.loc[(self.df.index >= self.day - 89) & (self.df.index <= self.day + self.cycle_days - 1)
                                & (self.df['tic'] == self.tic), :]
        self.shares = 0
        self.amount = self.initial_amount
        self.state = self._initiate_state()

        # initialize reward
        self.reward = []
        self.trades = 0
        self.episode = 0

        self.real_ep_reward = 0

    def reset(self, tic):
        # initiate state
        self.tic = tic
        self.shares = 0

        self.day = random.randint(89 + 27, len(self.df[self.df["tic"] == self.tic]) - 30 - 9)
        self.data = self.df.loc[(self.df.index >= self.day - 89) & (self.df.index <= self.day + self.cycle_days - 1)
                                & (self.df['tic'] == self.tic), :]

        data = self.df.loc[(self.df.index >= self.day - 89 - 27) &
                           (self.df.index <= self.day + self.delay_days + self.cycle_days - 1) &
                           (self.df['tic'] == self.tic)
        , :]
        date_diff = data['date'].diff().dt.days

        while (date_diff >= 15).any():  # 停牌日期过长，重新选择片段
            self.day = random.randint(89 + 27, len(self.df[self.df["tic"] == self.tic]) - 30 - 9)
            self.data = self.df.loc[(self.df.index >= self.day - 89) & (self.df.index <= self.day + self.cycle_days - 1)
                                    & (self.df['tic'] == self.tic), :]
            data = self.df.loc[(self.df.index >= self.day - 89 - 27) &
                               (self.df.index <= self.day + self.delay_days + self.cycle_days - 1) &
                               (self.df['tic'] == self.tic)
            , :]
            date_diff = data['date'].diff().dt.days

        self.state = self._initiate_state()
        self.trades = 0
        self.terminal = False
        self.amount = self.initial_amount
        self.episode += 1
        self.real_ep_reward = 0

        return self.state

    def _initiate_state(self):
        state = np.zeros(self.state_space)
        for i in range(self.cycle_days):
            for j in range(90):
                state[i, :, j] = list(self.data.iloc[i + j, 3: 11])

        return state.astype(np.float32)

    def step(self, actions):

        lst = []

        for i in range(self.cycle_days):
            begin_cash = self.amount
            begin_shares = self.shares
            date = self.data.iloc[89 + i, 0].date()
            if actions[i] == 1:
                buy_num_shares = self.amount // self.data.iloc[89 + i, 2]  # all in
                if buy_num_shares > 0:
                    real_action = 1
                    buy_amount = self.data.iloc[89 + i, 2] * buy_num_shares
                    self.amount -= buy_amount

                    self.shares += buy_num_shares
                    self.trades += 1
                else:
                    real_action = 0

            elif actions[i] == 2:
                if self.shares > 0:
                    real_action = 2
                    sell_num_shares = self.shares  # all out
                    sell_amount = self.data.iloc[89 + i, 2] * sell_num_shares
                    self.amount += sell_amount
                    self.shares -= sell_num_shares
                    self.trades += 1
                else:
                    real_action = 0

            else:
                real_action = 0

            end_cash = self.amount
            end_shares = self.shares

            lst.append([self.tic, date, begin_cash, begin_shares, actions[i], real_action, end_cash, end_shares])

        # calculate information after trading
        end_market_value = self.shares * self.df.loc[
                                         (self.df.index == (self.day + self.cycle_days + self.delay_days - 1)) & (
                                                     self.df['tic'] == self.tic), :][0, 2]
        self.end_total_asset = self.amount + end_market_value
        self.real_ep_reward = round(self.end_total_asset - self.initial_amount)

        return self.state, self.trades, self.real_ep_reward, lst


class StockTradingEnv_Test_Onetic(gym.Env):
    def __init__(self,
                 df,
                 initial_amount,
                 state_space,
                 tickers,
                 max_step
                 ):

        self.df = df
        self.initial_amount = initial_amount

        self.state_space = state_space
        self.action_space = spaces.Discrete(n=len(Actions))

        self.terminal = False

        self.day = 9
        self.tickers = tickers
        self.tic_idx = 0
        self.tic = self.tickers[self.tic_idx]
        self.max_step = max_step
        self.end_total_asset = 0

        self.data = self.df.loc[(self.df.index >= self.day - 9) & (self.df.index <= self.day)
                                & (self.df['tic'] == self.tic), :]
        self.shares = 0
        self.amount = self.initial_amount

        self.state = np.zeros(self.state_space)

        self.reward = 0
        self.trades = 0
        self.episode = 0
        self.real_ep_reward = 0

        self.buy_price = 0
        self.sell_price = 0
        self.steps = 0
        self.baseline = 0
        self.daily_cum_rewards = 0
        self.balance_rate = 0
        self.close_rate = 0


    def reset(self):
        # initiate states
        self.reward = 0
        self.trades = 0
        self.shares = 0
        self.steps = 0
        self.terminal = False

        self.initial_amount = self.daily_cum_rewards + self.initial_amount
        self.amount = self.initial_amount
        self.episode += 1

        self.data = self.df.loc[(self.df.index >= self.day - 9) & (self.df.index <= self.day)
                                & (self.df['tic'] == self.tic), :]
        self.baseline = self.data.iloc[:9, 2].mean()

        self.state = self._initiate_state()

        return self.state

    def _initiate_state(self):
        state = np.zeros(self.state_space)
        for j in range(10):
            state[3*j] = self.data.iloc[j, 4]
            state[3*j + 1] = self.data.iloc[j, 5]
            state[3*j + 2] = self.data.iloc[j, 6]

        if self.shares != 0:
            state[-2] = 1
            state[-1] = (self.data.iloc[9, 2] / self.buy_price) - 1.0

        # state[-1] = self.steps

        return state.astype(np.float32)

    def _update_state(self):
        state = self._initiate_state()

        return state

    def action_mask(self):
        end_cash = self.amount
        end_market_value = self.shares * self.data.iloc[9, 2]
        end_total_asset = end_cash + end_market_value
        self.daily_cum_rewards = end_total_asset - self.initial_amount
        self.close_rate = (self.data.iloc[9, 2] - self.baseline) / self.baseline
        self.balance_rate = self.daily_cum_rewards / self.initial_amount

        if self.shares != 0:
            # if self.steps >= self.max_step - 1 or self.close_rate <= -0.03:
            #     mask = torch.tensor([0, 0, 1])
            # else:
            mask = torch.tensor([1, 0, 1])

        else:
            # if self.steps >= self.max_step - 1 or self.close_rate <= -0.03:
            #     mask = torch.tensor([1, 0, 0])
            # else:
            mask = torch.tensor(([1, 1, 0]))

        return mask

    def step(self, action):

        if action == 1:
            self.buy_price = self.data.iloc[9, 2]
            buy_num_shares = (self.amount // self.buy_price) // 100 * 100  # all in, 100的整数倍
            self.shares += buy_num_shares
            buy_amount = self.buy_price * buy_num_shares
            self.amount -= buy_amount
            self.trades += 1

        elif action == 2:
            self.sell_price = self.data.iloc[9, 2]
            sell_num_shares = self.shares  # all out
            sell_amount = self.sell_price * sell_num_shares
            self.amount += sell_amount
            self.shares -= sell_num_shares
            self.trades += 1

        # update next state
        self.day += 1
        self.steps += 1

        if self.day >= len(self.df[self.df['tic'] == self.tic]):
            self.terminal = True
        else:
            self.data = self.df.loc[(self.df.index >= self.day - 9) & (self.df.index <= self.day)
                                    & (self.df['tic'] == self.tic), :]
            self.state = self._update_state()

        return self.state, self.reward, self.trades, self.terminal, self.daily_cum_rewards+self.initial_amount, self.steps