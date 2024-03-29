import tushare as ts
import pandas as pd
from get_tushare import TushareProcessor
from sklearn.preprocessing import MinMaxScaler
import os
import random

INDICATORS = [
    "macd",
    "close_9_ema",
    "close_27_ema",
]

start_date = '2006-01-01'
stop_date = '2024-03-01'


dp = TushareProcessor('8eb0393d1ee06075ce5092170d0dc91f8932ac04e4ab960e9a72e1ef')
scaler = MinMaxScaler()


def ticker_code():
    pro = ts.pro_api('8eb0393d1ee06075ce5092170d0dc91f8932ac04e4ab960e9a72e1ef')
    data = pro.query('stock_basic', exchange='', list_status='L', fields='ts_code')

    ticker_list = data["ts_code"]
    ticker_list.to_csv("ticker.csv", index=False)


def normalize(group):
    columns_to_normalize = group.columns.difference(['date', 'tic', 'close0'])
    group[columns_to_normalize] = scaler.fit_transform(group[columns_to_normalize])
    return group


def get_data():
    ticker_list = pd.read_csv("ticker.csv")["ts_code"]
    dp.download_data(ticker_list, start_date, stop_date, '1d')

    # data = pd.DataFrame()
    # cnt = 0
    # for tic in os.listdir("./dataset"):
    #     if tic[-6:-4] in ['SH', 'SZ', 'BJ']:
    #         cnt += 1
    #         t = pd.read_csv(f"./dataset/{tic}")
    #         data = pd.concat([data, t])
    #         print(f"{cnt} tickers done!")
    #
    # data.columns = [
    #     "date",
    #     "tic",
    #     "open",
    #     "high",
    #     "low",
    #     "close",
    #     "volume"
    # ]
    #
    # print("read done.")
    # data = dp.add_technical_indicator(data, INDICATORS)
    # print("add ti done.")
    # data = dp.delete_first_30(data)
    # print("delete 30 done.")
    # data = data.sort_values(by=["tic", "date"]).reset_index(drop=True)
    # print("sort done.")
    #
    # data.insert(2, 'close0', data["close"])
    # print("close0 done.")
    #
    # data = data.groupby('tic').apply(normalize)
    # print("standarized done.")
    #
    # num_to_train = len(ticker_list) // 10 * 9
    # ticker_list = ticker_list.to_list()
    # random.shuffle(ticker_list)
    # train_tickers = ticker_list[:num_to_train]
    # test_tickers = ticker_list[num_to_train:]
    #
    # train_data = data[data['tic'].isin(train_tickers)]
    # test_data = data[data['tic'].isin(test_tickers)]
    # print("isin done.")
    # train_data.to_csv("./dataset/2006-2024/train_data.csv", index=False)
    # print("train data done!")
    # test_data.to_csv("./dataset/2006-2024/test_data.csv", index=False)
    # print("test data done!")


def get_one_tic(tic_name):

    data = pd.DataFrame()
    for tic in os.listdir("./dataset"):
        if tic[:-4] == tic_name:
            t = pd.read_csv(f"./dataset/{tic}")
            data = pd.concat([data, t])

    data.columns = [
        "date",
        "tic",
        "open",
        "high",
        "low",
        "close",
        "volume"
    ]

    data.drop(columns=['volume'], inplace=True)

    # data = dp.add_technical_indicator(data, INDICATORS)
    # print("add ti done.")
    # data = dp.delete_first_30(data)
    # print("delete 30 done.")
    data = data.sort_values(by=["tic", "date"]).reset_index(drop=True)

    data.insert(2, 'close0', data["close"])
    data['high'] = (data['high'] - data['open'])/data['open']
    data['low'] = (data['low'] - data['open']) / data['open']
    data['close'] = (data['close'] - data['open']) / data['open']
    # scaler = MinMaxScaler()
    # columns_to_normalize = data.columns.difference(['date', 'tic', 'close0'])
    # # 对选择的列进行最大最小标准化
    # data[columns_to_normalize] = scaler.fit_transform(data[columns_to_normalize])
    # print("standarized done.")

    train_data = data[data['date'] < '2022-01-01']
    test_data = data[data['date'] >= '2022-01-01']
    train_data.to_csv(f"./dataset/2006-2024/train_data_{tic_name}.csv", index=False)
    print("train data done!")
    test_data.to_csv(f"./dataset/2006-2024/test_data_{tic_name}.csv", index=False)
    print("test data done!")


if __name__ == "__main__":
    # ticker_code()
    get_one_tic('600585.SH')