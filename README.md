# 环境配置

- python3.10 
- 终端输入：`pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121`
- pip install 所需库：gym，tushare，stockstats， mplfinance， tensorboardX等

# Procedures

## 1. Data preprocessing
`preprocess.py`文件
- `get_data`函数获取`Tushare`所有股票从2006年至2024年的每日前复权数据(OHLCV)。调用`get_tushare.py`文件中`download_data`方法
- `get_onetic`函数将一只股票的历史数据分为训练集和测试集，计算最高价、最低价和收盘价相对于开盘价的上升或下降比例。
- 数据存入`dataset`目录中

## 2. Environment
`Environment.py`文件 
- 注：该文件内写了多个环境，本次实验只使用StockTradingEnv_1D和StockTradingEnv_Test_Onetic环境。
- 任意一支股票从数据集最初以初始资金`100w`开始交易，直至数据集结束。不设最大游戏天数。

### 2.1 state
`class StockTradingEnv_1D`
- `32`维的一维矩阵。使用前`10`天的最高价、最低价和收盘价相对于开盘价的上升或下降比例（3*10）、当前是否持有股票、当前收盘价相对于本次买入价格的增长率。

        for j in range(10):
            state[3*j] = self.data.iloc[j, 4]
            state[3*j + 1] = self.data.iloc[j, 5]
            state[3*j + 2] = self.data.iloc[j, 6]

        if self.shares != 0:
            state[-2] = 1
            state[-1] = (self.data.iloc[9, 2] / self.buy_price) - 1.0

### 2.2 action
- `buy, sell, hold`。决定`buy`则用当前所有现金买（整百），决定`sell`则将当前持有的所有股票卖出。
-     `action_mask`方法：
      if 当前持有股票为0:
          只能选择动作hold和sell
      else:
          只能选择动作hold和buy

### 2.2 reward
- 只有当卖出股票时才计算奖励。
- reward = sell_num_shares * (sell_price - buy_price)，
每次买卖扣除`0.001`的交易费。

## 3. Agent
`PPO.py`
- 使用PPO算法，三层全连接神经网络。

-     self.actor = nn.Sequential(
            nn.Linear(shape, 512),
            nn.Tanh(),
            nn.Linear(512, 512),
            nn.Tanh(),
            nn.Linear(512, action_dim),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(shape, 512),
            nn.Tanh(),
            nn.Linear(512, 512),
            nn.Tanh(),
            nn.Linear(512, 1)
        )

## 4. train and test
1. split dataset：2006-2021年训练集，2022至今为测试集。
2. train：`train_ppo.py` 指定一支股票，从数据集任意时间节点开始，至整个数据集结束，执行一个episode的交易。
每收集`4`个episode的训练数据更新一次参数。采用多进程并行收集数据，具体实现在`sampler_asyn.py`文件。
3. validation：每更新`1`次参数做一次validation，在训练过程中来衡量模型的泛化能力，不参与训练，只做评估，便于观察训练效果。
使用`StockTradingEnv_Test_Onetic`环境，对所选股票从2022年开始交易至今，计算所获利润。
4. test：`test.py` agent训练完后，使用环境`StockTradingEnv_Test_Onetic`测试。画出资金曲线和在K线图之上的买卖点。输出`K_line_buy_and_sell.csv`文件，记录每日OHLC和买卖点的买入价格和卖出价格。

## 5. tensorboard
- 可使用tensorboard观察loss和reward变化曲线
- 在终端输入命令：`tensorboard --logdir=./log/ppo_601919.SH`