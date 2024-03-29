import pandas as pd
from Environment import StockTradingEnv_Test_Onetic
from PPO import PPO
import numpy as np
import matplotlib.pyplot as plt
import mplfinance as mpf
import torch


plt.rcParams['font.family'] = 'SimHei'
tic = '601225.SH'
checkpoint_path = f"log/ppo_{tic}/models/best_model.pth"
print("loading network from : " + checkpoint_path)
device = 'cpu'
val_file = f"./dataset/2006-2024/test_data_{tic}.csv"
val_data = pd.read_csv(val_file)
tic_counts = val_data.groupby('tic').size()

val_tickers = list(val_data['tic'].unique())
selected_tickers = val_tickers
# selected_tickers = random.sample(val_tickers, 50)
max_step = 60
state_space = 32
val_data["date"] = pd.to_datetime(val_data["date"], format="%Y-%m-%d")
val_data = val_data[val_data["tic"].isin(selected_tickers)]
# val_data = val_data[val_data['date'] <= '2022-01-01']
val_data['index'] = val_data.groupby('tic').cumcount()
val_data.set_index(['index'], inplace=True)
val_data.index.names = [None]

action_dim = 3
lr_actor = 0.0001
lr_critic = 0.0001
gamma = 0.99
K_epochs = 10
eps_clip = 0.2


env_kwargs = {
    "initial_amount": 1000000,
    "state_space": state_space,
    "tickers": val_tickers,
    "max_step": max_step
}
env = StockTradingEnv_Test_Onetic(df=val_data, **env_kwargs)
ppo_agent = PPO(state_space, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip)
ppo_agent.load(checkpoint_path)
print("--------------------------------------------------------------------------------------------")
balance_list = []
rewards = []

data = pd.read_csv(f'./dataset/{tic}.csv')
data["date"] = pd.to_datetime(data["date"], format="%Y-%m-%d")
data = data[data['date'] >= '2022-01-01']
data['index'] = data.groupby('tic').cumcount()
data.set_index(['index'], inplace=True)
data.index.names = [None]
actions_0 = []
actions_1 = []
actions_2 = []

for tic in selected_tickers:
    # if len(val_data[val_data["tic"] == tic]) < 155:
    #     print(f"{tic} doesn't have enough length!")
    #     continue
    step = 0
    print(f"ticker: {tic}")
    start_idx = val_data[val_data['date'] >= '2022-01-01'].index[0]
    test_length = len(val_data.loc[(val_data["tic"] == tic) & (val_data['date'] >= '2022-01-01'), :])

    while step < test_length - 9:
        state = env.reset()
        done = False
        # if not state:
        #     print("reset 10 times and still don't find a suitable period.")
        #     break

        while not done:
            step += 1
            mask = env.action_mask()
            action = ppo_agent.select_action(state, mask, device, False)
            state, reward, trades, done, balance, steps = env.step(action)
            balance_list.append(balance)
            if action == 0:
                actions_0.append(data.iloc[step - 1 + 9, 5])
                actions_1.append(np.NaN)
                actions_2.append(np.NaN)
            elif action == 1:
                actions_0.append(np.NaN)
                actions_1.append(data.iloc[step - 1 + 9, 5])
                actions_2.append(np.NaN)
            else:
                actions_0.append(np.NaN)
                actions_1.append(np.NaN)
                actions_2.append(data.iloc[step - 1 + 9, 5])
            # for i in range(cycle_days):
            #     if lst[i][4] != 0:
            #         result_df.loc[len(result_df)] = lst[i]
        # if steps < max_step and step < test_length - 9:
        #     t = max_step - steps + 1
        #     for i in range(1, t+1):
        #         step += 1
        #         actions_0.append(data.iloc[step - 1 + 9 + t, 5])
        #         actions_1.append(np.NaN)
        #         actions_2.append(np.NaN)
        #         balance_list.append(balance)


        # new_row = ['Reward: {}, trades: {}'.format(round(real_reward, 2), trades), None, None, None, None, None,
        #            None, None]
        # result_df.loc[len(result_df)] = new_row
        # print('Reward: {}, trades: {}'.format(round(real_reward, 2), trades))

    print("============================================================================================")

# calculate profit rate
print((balance_list[-1]-balance_list[0])/balance_list[0])

# calculate MDD
max_drawdown = 0
for i in range(len(balance_list)):
    for j in range(i+1, len(balance_list)):
        drawdown = (balance_list[i] - balance_list[j]) / balance_list[i]
        if drawdown > max_drawdown:
            max_drawdown = drawdown

print(max_drawdown)

balance_list1 = []
step = 0
start_idx = val_data[val_data['date'] >= '2022-01-01'].index[0]
test_length = len(val_data.loc[(val_data["tic"] == tic) & (val_data['date'] >= '2022-01-01'), :])
env1 = StockTradingEnv_Test_Onetic(df=val_data, **env_kwargs)

while step < test_length - 9:
    state = env1.reset()
    done = False

    while not done:
        step += 1
        if step == 1:
            action = 1
        else:
            action = 0
        mask = env1.action_mask()
        state, reward, trades, done, balance, steps = env1.step(action)
        balance_list1.append(balance)

print("============================================================================================")

print((balance_list1[-1]-balance_list1[0])/balance_list1[0])

length = len(actions_0)
data = data.drop(data.index[:9])[:length]
data['actions_0'] = actions_0
data['actions_1'] = actions_1
data['actions_2'] = actions_2
data.set_index(['date'], inplace=True)
data.to_csv('K_line_buy_and_sell.csv', index=False)


# plt.plot(balance_list, label='Agent')
# plt.plot(balance_list1, label='持有至结束')
# plt.xlabel("交易天数")
# plt.ylabel("资产")
# plt.title("陕西煤业")
# plt.legend()
# # plt.show()
# plt.savefig(f'{tic}.png')

my_color = mpf.make_marketcolors(up='red', down='green', edge='inherit', volume='inherit')
my_style = mpf.make_mpf_style(marketcolors=my_color)

add_plot = [mpf.make_addplot(data['actions_1'][0:100], scatter=True, markersize=20, marker='^', color='r'),
            mpf.make_addplot(data['actions_2'][0:100], scatter=True, markersize=20, marker='v', color='g')]
mpf.plot(data[0:100], type='candle', ylabel='price', style=my_style, addplot=add_plot)
# result_df.to_csv('./results/test_result.csv', index=False)
