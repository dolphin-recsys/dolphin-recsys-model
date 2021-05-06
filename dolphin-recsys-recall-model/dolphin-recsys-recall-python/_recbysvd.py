import os
import random
import numpy as np
import pandas as pd
from typing import *
from IPython.display import display, HTML, Markdown
from timeit import default_timer
from datetime import timedelta
from surprise import SVD
from surprise import Dataset, Reader
from surprise.model_selection import cross_validate, train_test_split
from surprise import accuracy

from sklearn.model_selection import train_test_split as sk_split
import warnings
warnings.filterwarnings('ignore')
my_seed = 1337
random.seed(my_seed)
np.random.seed(my_seed)

#一些必要的函数
class Timer(object):
    """Timer class.

    `Original code <https://github.com/miguelgfierro/pybase/blob/2298172a13fb4a243754acbc6029a4a2dcf72c20/log_base/timer.py>`_.

    Examples:
        # >>> import time
        # >>> t = Timer()
        # >>> t.start()
        # >>> time.sleep(1)
        # >>> t.stop()
        # >>> t.interval < 1
        # True
        # >>> with Timer() as t:
        # ...   time.sleep(1)
        # >>> t.interval < 1
        # True
        # >>> "Time elapsed {}".format(t) #doctest: +ELLIPSIS
        # 'Time elapsed 1...'
    """

    def __init__(self):
        self._timer = default_timer
        self._interval = 0
        self.running = False

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()

    def __str__(self):
        return "{:0.4f}".format(self.interval)

    def start(self):
        """Start the timer."""
        self.init = self._timer()
        self.running = True

    def stop(self):
        """Stop the timer. Calculate the interval in seconds."""
        self.end = self._timer()
        try:
            self._interval = self.end - self.init
            self.running = False
        except AttributeError:
            raise ValueError(
                "Timer has not been initialized: use start() or the contextual form with Timer() as t:"
            )

    @property
    def interval(self):
        """Get time interval in seconds.

        Returns:
            float: Seconds.
        """
        if self.running:
            raise ValueError("Timer has not been stopped, please use stop().")
        else:
            return self._interval

def compute_ranking_predictions(
        algo,
        data,
        usercol='用户ID',
        itemcol='电影名',
        predcol='评分',
        remove_seen=False,
):
    """Computes predictions of an algorithm from Surprise on all users and items in data. It can be used for computing
    ranking metrics like NDCG.

    Args:
        algo (surprise.prediction_algorithms.algo_base.AlgoBase): an algorithm from Surprise
        data (pd.DataFrame): the data from which to get the users and items
        usercol (str): name of the user column
        itemcol (str): name of the item column
        remove_seen (bool): flag to remove (user, item) pairs seen in the training data

    Returns:
        pd.DataFrame: dataframe with usercol, itemcol, predcol
    """
    preds_lst = []
    users = data[usercol].unique()
    items = data[itemcol].unique()
    for user in users:
        for item in items:
            preds_lst.append([user, item, algo.predict(user, item).est])
    # all_predictions 所有用户对每个电影的 用户iD 电影iD 预测评分
    all_predictions = pd.DataFrame(data=preds_lst, columns=[usercol, itemcol, predcol])

    if remove_seen:
        # tempdf 存储的是用户看过的电影
        tempdf = pd.concat(
            [
                data[[usercol, itemcol]],
                pd.DataFrame(
                    data=np.ones(data.shape[0]), columns=["dummycol"], index=data.index
                ),
            ],
            axis=1,
        )
        # 看过的电影和所有电影merge
        merged = pd.merge(tempdf, all_predictions, on=[usercol, itemcol], how="outer")
        # 在结果集中去掉用户看过的电影
        return merged[merged["dummycol"].isnull()].drop("dummycol", axis=1)
    else:
        return all_predictions


def Svd_predict(
        algo,
        data,
        usercol='用户ID',
        itemcol='电影名',
        predcol='评分',
):
    """Computes predictions of an algorithm from Surprise on the data. Can be used for computing rating metrics like RMSE.

    Args:
        algo (surprise.prediction_algorithms.algo_base.AlgoBase): an algorithm from Surprise
        data (pd.DataFrame): the data on which to predict
        usercol (str): name of the user column
        itemcol (str): name of the item column

    Returns:
        pd.DataFrame: dataframe with usercol, itemcol, predcol
    """
    predictions = [
        algo.predict(getattr(row, usercol), getattr(row, itemcol))
        for row in data.itertuples()
    ]
    predictions = pd.DataFrame(predictions)
    predictions = predictions.rename(
        index=str, columns={"uid": usercol, "iid": itemcol, "est": predcol}
    )
    return predictions.drop(["details", "r_ui"], axis="columns")

#Step one加载全量数据
def load_user_watched_movies() :
    user_data = pd.read_csv(data_dir + '/user.csv')
    print('user_data\n',user_data.shape)
    user_data['评论时间'] = pd.to_datetime(user_data['评论时间'])
    return user_data

#step two SVD建模
#step three 从总数据集中采样一部分样本参加训练
def sampledata_train(u_data):
    train_all  =  u_data
    print('in sampledata_train\n',train_all.shape)
    train_all = train_all.sample(frac=0.1, replace=True, random_state=1)
    print('in sampledata_train\n',train_all.shape)
    print(train_all.shape)
    train_all_set = Dataset.load_from_df(train_all, reader=Reader(rating_scale=(2, 10))).build_full_trainset()

    svd = SVD(random_state=0, n_factors=200, n_epochs=15, verbose=True)

    with Timer() as train_time:
        svd.fit(train_all_set)

    print("Took {} seconds for training.".format(train_time.interval))
    return svd,train_all

#step four 根据用户观看过的电影的平均数13个。给每个用户推荐其最有可能观看的10个电影
def get_all_predictions(datas):
    svd, train_all = sampledata_train(datas)
    print('in get_all_predictions train_all shape\n', train_all.shape)
    with Timer() as test_time:
        all_predictions = compute_ranking_predictions(svd, train_all, usercol='用户ID', itemcol='电影名', remove_seen=True)

    print("Took {} seconds for prediction.".format(test_time.interval))

    # 按评分简单排序
    all_predictions = all_predictions.sort_values(by=['用户ID', '评分'], ascending=False)
    # 召回
    # 对每个用户取排序后前20的电影名
    print('----开始为用户推荐其最有可能看的11个电影-----\n')
    user_count = 0
    for user in all_predictions.用户ID.unique():
        usertemp_df = all_predictions[all_predictions['用户ID'] == user][:11]
        if user_count == 0:
            dfret = usertemp_df.copy(deep=True)
            user_count += 1
        else:
            dfret = pd.concat([dfret, usertemp_df], axis=0, ignore_index=True)
    print('dfret shape', dfret.shape)

    # 召回结果保存
    dfret.to_csv("./cache/recalldatas0429.csv", index=False)

def get_all():
    user_rating_data = load_user_watched_movies()
    print(user_rating_data.head())
    u_data = user_rating_data[['用户ID', '电影名', '评分']]
    get_all_predictions(u_data)

if __name__ == "__main__":
    data_dir = 'D:/python/Jupyter_Last_project/dataset/'
    get_all()