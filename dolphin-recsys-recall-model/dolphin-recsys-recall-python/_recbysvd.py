import os
import random
import pickle
import warnings
import numpy as np
import pandas as pd
from surprise import SVD
from threading import Timer
from datetime import datetime
from timeit import default_timer
from surprise import Dataset, Reader
warnings.filterwarnings('ignore')
from recallprocess import config_get
my_seed = 1337
random.seed(my_seed)
np.random.seed(my_seed)

class Tsave:
    def __init__(self):
        self.path_lis = []
    def save_df(self,df,path,count=0):
        temp_path = path + datetime.now().strftime("%Y-%m-%d-%H") + str(count) + '.pkl'
        if isinstance(df ,pd.DataFrame):
            if len(set(self.path_lis)) > 2:
                tlis = list(set(self.path_lis))
                df1 = pickle.load( open(tlis[-1], 'rb'))
                df2 = pickle.load( open(tlis[-2], 'rb'))
                if df1.shape == df2.shape:
                    print('保存完毕')
                    return 1
        if os.path.exists(temp_path):
            pass
        else:
            pickle.dump(df, open(temp_path, 'wb'))
            self.path_lis.append(temp_path)

def find_user_item_index(tlis,users,items):
    print('in fid func')
    for uindex,user in enumerate(users):
        if user == tlis[0]:
            print('user index is ',uindex)
            print(user)
            break
    for iindex,item  in enumerate(items):
        if item == tlis[1]:
            print('item index is ',iindex)
            print(item)
            break
    return uindex,iindex

#一些必要的函数
class Timera(object):
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
    """
    if the path exists the func can load the list from pickle
    and change the start index of user and item . in this way
    it don't have to calculate from zero index of user and zero
    index of item in the double for loop. so it can save time.

    Computes predictions of an algorithm from Surprise on all users
    and items in data. It can be used for computing
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
    print('len(users)\n',len(users))
    print('len(items)\n',len(items))
    pred_path = config_get('rcache_path','recal_middle_pkl')
    user_start_index = 0
    item_start_index = 0
    if os.path.exists(pred_path):
        print('begin load list')
        preds_lst = pickle.load(open(pred_path, 'rb'))
        print('len preds_lst\n',len(preds_lst))
        print(preds_lst[-1])
        user_start_index,item_start_index = find_user_item_index(preds_lst[-1],users,items)
    print('开始用户index', user_start_index)
    print('开始电影index', item_start_index)
    # for user in  users[user_start_index:]:
    #     for item in items[item_start_index:]:
    #         print('开始用户',user)
    #         print('开始电影',item)
    #         preds_lst.append([user, item, algo.predict(user, item).est])
    #         tsave = Tsave()
    #         path = config_get('data_source','cache_list')
    #         tsave.save_df(preds_lst, path)
    ##all_predictions 所有用户对每个电影的 用户iD 电影iD 预测评分
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
    user_data = pd.read_csv(config_get('data_source', 'user_source_data'))
    user_data['评论时间'] = pd.to_datetime(user_data['评论时间'])
    return user_data

#step two SVD建模
#step three 从总数据集中采样一部分样本参加训练
def sampledata_train(u_data):
    train_all = u_data
    train_all_set = Dataset.load_from_df(train_all, reader=Reader(rating_scale=(2, 10))).build_full_trainset()
    svd = SVD(random_state=0, n_factors=200, n_epochs=15, verbose=True)
    with Timera() as train_time:
        svd.fit(train_all_set)
    print("Took {} seconds for training.".format(train_time.interval))
    return svd,train_all

#step four 根据用户观看过的电影的平均数13个。给每个用户推荐其最有可能观看的10个电影
def get_all_predictions(datas):
    '''
    there is no need to save the middle datas
    because there is only one for loop in this func
    '''
    svd, train_all = sampledata_train(datas)
    print('in get_all_predictions train_all shape\n', train_all.shape)
    with Timera() as test_time:
        all_predictions = compute_ranking_predictions(svd, train_all, usercol='用户ID', itemcol='电影名', remove_seen=True)
    print('loaded data formed dataframe shape',all_predictions.shape)
    print("Took {} seconds for prediction.".format(test_time.interval))
    # 按评分简单排序
    all_predictions = all_predictions.sort_values(by=['用户ID', '评分'], ascending=False)
    # 召回
    # 对每个用户取排序后前10的电影名
    print('开始为用户推荐其最有可能看的10个电影\n')
    user_count = 0
    save_count = 0
    cti = 0
    total_count = all_predictions.用户ID.nunique()
    tsave = Tsave()
    print('total_count ',total_count)
    for user in all_predictions.用户ID.unique():
        save_count += 1
        usertemp_df = all_predictions[all_predictions['用户ID'] == user][:10]
        if user_count == 0:
            dfret = usertemp_df.copy(deep=True)
            user_count += 1
        else:
            dfret = pd.concat([dfret, usertemp_df], axis=0, ignore_index=True)
            path = config_get('data_source','cache_dataframe')
            # if (save_count == int(total_count / 10)):
            #     tsave.save_df(dfret, path,count=cti)
            #     save_count = 0
            #     cti += 1
    print('dfret shape', dfret.shape)
    pickle.dump(dfret, open(path + '_all.pkl', 'wb'))#用于校验
    dfret.to_csv(config_get('data_source','svd_recall_datas'), index=False)

def get_all():
    user_rating_data = load_user_watched_movies()
    print(user_rating_data.head())
    u_data = user_rating_data[['用户ID', '电影名', '评分']]
    get_all_predictions(u_data)

if __name__ == "__main__":
    get_all()
    #df  = pickle.load( open('C:/Users/Administrator/Desktop/laoliu0504/cache/2021-05-09-10-349.pkl', 'rb'))
    #print(df.shape)