# -*- coding: utf-8 -*-
import os
import redis
import pickle
import pandas as pd
import findspark
findspark.init()
import pyspark.sql as sql
from pyspark.sql.types import *
from configparser import ConfigParser
from collections import defaultdict
from pyspark.sql.functions import *
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark import SparkContext, SparkConf

def same_process(x):
    listx = list(x)
    listx = list(set(listx))
    return '|'.join(listx)

def processdict(x,dictin):
    litemp = []
    for i in x:
        if dictin[i] not in litemp:
            litemp.append(dictin[i])
    return [tuple(litemp)]

def get_all_directors_and_actors(dfin):
    all_acotr_list = dfin.主演.unique()
    all_director_list = dfin.导演.unique()
    actors = []
    directors = []
    for i in all_acotr_list:
        if '|' in i:
            temp_actor = i.split('|')
            for j in temp_actor:
                actors.append(j)
        else:
            actors.append(i)

    for i in all_director_list:
        if '|' in i:
            temp_director = i.split('|')
            for j in temp_director:
                directors.append(j)
        else:
            directors.append(i)
    return list(set(actors)), list(set(directors))

def get_actor_dict_and_director_dict(dfin):
    actors_list ,directors_list = get_all_directors_and_actors(dfin)
    list_actors_index = [i for i in range(len(actors_list))]
    dict_actortoindex = dict(zip(actors_list,list_actors_index))
    dict_indextoacotr = dict(zip(list_actors_index,actors_list))
    list_director_index = [i for i in range(len(directors_list))]
    dict_directortoindex = dict(zip(directors_list,list_director_index))
    dict_indextodirecotr = dict(zip(list_director_index,directors_list))
    return dict_actortoindex,dict_indextoacotr,dict_directortoindex,dict_indextodirecotr

def config_get(name1,name2=None):
    ini_file = "../config.ini"
    cfg = ConfigParser()
    # 读取文件内容
    cfg.read(ini_file, encoding='utf-8')
    if name2:
        return cfg.get(name1,name2)
    else:
        return dict(cfg.items(name1))

def recall_process():
    pool = redis.ConnectionPool(**config_get('redis'))
    r = redis.Redis(connection_pool=pool)
    allmovies = pd.read_csv(config_get('data_source', 'movie_source_data'))
    allmovies = allmovies.rename(columns={'评分': "豆瓣网评分"})
    allusers = pd.read_csv(config_get('data_source', 'user_source_data'))
    all_movies_list = allmovies.电影名.unique()
    index_list = [i for i in range(len(all_movies_list))]
    dict_movietoindex = dict(zip(all_movies_list, index_list))
    dict_indextomovie = dict(zip(index_list, all_movies_list))
    recalls = pd.read_csv(config_get('data_source', 'svd_recall_datas'))
    recalls['movieid'] = recalls['电影名'].map(lambda x: dict_movietoindex[x])
    alldatas = pd.merge(allusers, allmovies, on=['电影名', '类型'], how='inner')
    dict_actortoindexpath = '../cache/dict_actortoindex.pkl'
    dict_indextoacotrpath = '../cache/dict_indextoacotr.pkl'
    dict_directortoindexpath = '../cache/dict_directortoindex.pkl'
    dict_indextodirecotrpath = '../cache/dict_indextodirecotr.pkl'
    dict_path = '../cache/dict.pkl'
    if os.path.exists(dict_actortoindexpath):
        print('exists')
        dict_actortoindex = pickle.load(open(dict_actortoindexpath, 'rb'))
        dict_indextoacotr = pickle.load(open(dict_indextoacotrpath, 'rb'))
        dict_directortoindex = pickle.load(open(dict_directortoindexpath, 'rb'))
        dict_indextodirecotr = pickle.load(open(dict_indextodirecotrpath, 'rb'))
    else:
        dict_actortoindex, dict_indextoacotr, \
        dict_directortoindex, dict_indextodirecotr = get_actor_dict_and_director_dict(alldatas)
        pickle.dump(dict_actortoindex, open(dict_actortoindexpath, 'wb'))
        pickle.dump(dict_indextoacotr, open(dict_indextoacotrpath, 'wb'))
        pickle.dump(dict_directortoindex, open(dict_directortoindexpath, 'wb'))
        pickle.dump(dict_indextodirecotr, open(dict_indextodirecotrpath, 'wb'))
    del alldatas['类型']
    del alldatas['主演']
    del alldatas['地区']
    del alldatas['导演']
    del alldatas['特色']
    alldatas['label'] = alldatas.评分.map(lambda x: 1 if x >= 6 else 0)
    if os.path.exists(dict_path):
        dict_userid_to_others = pickle.load(open(dict_path, 'rb'))
    else:
        dict_userid_to_others = dict()
        for idex, row in alldatas.iterrows():
            if dict_userid_to_others.get(row['用户ID'], 0) == 0:
                dict_userid_to_others[row['用户ID']] = {'评分': row['评分'], '用户名': row['用户名'],
                                                      '评论时间': row['评论时间'], '电影名': row['电影名'],
                                                      '豆瓣网评分': row['豆瓣网评分'], 'movieid': row['movieid'],
                                                      'label': row['label']}
        pickle.dump(dict_userid_to_others, open(dict_path, 'wb'))
    recalls['用户名'] = recalls['用户ID'].map(lambda x: dict_userid_to_others[x].get('用户名'))
    recalls['评论时间'] = recalls['用户ID'].map(lambda x: dict_userid_to_others[x].get('评论时间'))
    recalls['豆瓣网评分'] = recalls['用户ID'].map(lambda x: dict_userid_to_others[x].get('豆瓣网评分'))
    recalls = recalls.rename(columns={'用户ID': "userid"})
    recalls = recalls.dropna(how='any')
    recalls['userPositiveType1'] = recalls['userid'].map(lambda x: r.hgetall("uf:" + str(x))['userPositiveType1'] \
        if len(r.hgetall("uf:" + str(x))) >= 1 else None)
    recalls['userPositiveType2'] = recalls['userid'].map(lambda x :r.hgetall("uf:" + str(x))['userPositiveType2'] \
                                    if len(r.hgetall("uf:" + str(x))) >=1 else None)
    recalls['userPositiveType3'] = recalls['userid'].map(lambda x :r.hgetall("uf:" + str(x))['userPositiveType3'] \
                                    if len(r.hgetall("uf:" + str(x))) >=1 else None)
    recalls['userPositiveType4'] = recalls['userid'].map(lambda x :r.hgetall("uf:" + str(x))['userPositiveType4'] \
                                    if len(r.hgetall("uf:" + str(x))) >=1 else None)
    recalls['userPositiveType5'] = recalls['userid'].map(lambda x :r.hgetall("uf:" + str(x))['userPositiveType5'] \
                                    if len(r.hgetall("uf:" + str(x))) >=1 else None)
    recalls['userRatedMovie1'] = recalls['userid'].map(lambda x :r.hgetall("uf:" + str(x))['userRatedMovie1'] \
                                    if len(r.hgetall("uf:" + str(x))) >=1 else None)
    recalls['userPositivefeature1'] = recalls['userid'].map(lambda x :r.hgetall("uf:" + str(x))['userPositivefeature1'] \
                                    if len(r.hgetall("uf:" + str(x))) >=1 else None)
    recalls['userPositivefeature2'] = recalls['userid'].map(lambda x :r.hgetall("uf:" + str(x))['userPositivefeature2'] \
                                    if len(r.hgetall("uf:" + str(x))) >=1 else None)
    recalls['userPositivefeature3'] = recalls['userid'].map(lambda x :r.hgetall("uf:" + str(x))['userPositivefeature3'] \
                                    if len(r.hgetall("uf:" + str(x))) >=1 else None)
    recalls['userPositivefeature4'] = recalls['userid'].map(lambda x :r.hgetall("uf:" + str(x))['userPositivefeature4'] \
                                    if len(r.hgetall("uf:" + str(x))) >=1 else None)
    recalls['userPositivefeature5'] = recalls['userid'].map(lambda x :r.hgetall("uf:" + str(x))['userPositivefeature5'] \
                                    if len(r.hgetall("uf:" + str(x))) >=1 else None)
    recalls['userPositivearea1'] = recalls['userid'].map(lambda x :r.hgetall("uf:" + str(x))['userPositivearea1'] \
                                    if len(r.hgetall("uf:" + str(x))) >=1 else None)
    recalls['userPositivearea2'] = recalls['userid'].map(lambda x :r.hgetall("uf:" + str(x))['userPositivearea2'] \
                                    if len(r.hgetall("uf:" + str(x))) >=1 else None)
    recalls['userPositivearea3'] = recalls['userid'].map(lambda x :r.hgetall("uf:" + str(x))['userPositivearea3'] \
                                    if len(r.hgetall("uf:" + str(x))) >=1 else None)
    recalls['userPositivearea4'] = recalls['userid'].map(lambda x :r.hgetall("uf:" + str(x))['userPositivearea4'] \
                                    if len(r.hgetall("uf:" + str(x))) >=1 else None)
    recalls['userPositivearea5'] = recalls['userid'].map(lambda x :r.hgetall("uf:" + str(x))['userPositivearea5'] \
                                    if len(r.hgetall("uf:" + str(x))) >=1 else None)
    recalls['userRatingCount'] = recalls['userid'].map(lambda x :r.hgetall("uf:" + str(x))['userRatingCount'] \
                                    if len(r.hgetall("uf:" + str(x))) >=1 else None)
    recalls['userRatingStddev'] = recalls['userid'].map(lambda x :r.hgetall("uf:" + str(x))['userRatingStddev'] \
                                    if len(r.hgetall("uf:" + str(x))) >=1 else None)
    recalls['userAvgRating'] = recalls['userid'].map(lambda x :r.hgetall("uf:" + str(x))['userAvgRating'] \
                                    if len(r.hgetall("uf:" + str(x))) >=1 else None)
    recalls['movieactor1'] = recalls['movieid'].map(lambda x :r.hgetall("mf:" + str(x))['movieactor1'] \
                                        if len(r.hgetall("mf:" + str(x))) >=1 else None) \
                                        .map(lambda x: dict_actortoindex[x] if len(x) >= 1 else None)
    recalls['movieactor2'] = recalls['movieid'].map(lambda x :r.hgetall("mf:" + str(x))['movieactor2'] \
                                        if len(r.hgetall("mf:" + str(x))) >=1 else None)\
                                    .map(lambda x: dict_actortoindex[x] if  len(x)>=1    else None)

    recalls['movieactor3'] = recalls['movieid'].map(lambda x :r.hgetall("mf:" + str(x))['movieactor3'] \
                                        if len(r.hgetall("mf:" + str(x))) >=1 else None) \
                                    .map(lambda x: dict_actortoindex[x] if len(x) >= 1 else None)
    recalls['movieactor4'] = recalls['movieid'].map(lambda x :r.hgetall("mf:" + str(x))['movieactor4'] \
                                        if len(r.hgetall("mf:" + str(x))) >=1 else None) \
                                    .map(lambda x: dict_actortoindex[x] if len(x) >= 1 else None)
    recalls['moviedirecotr1'] = recalls['movieid'].map(lambda x :r.hgetall("mf:" + str(x))['moviedirecotr1'] \
                                        if len(r.hgetall("mf:" + str(x))) >=1 else None)\
                                         .map(lambda x:dict_directortoindex[x] if len(x) >= 1 else None)
    recalls['movieRatingCount'] = recalls['movieid'].map(lambda x :r.hgetall("mf:" + str(x))['movieRatingCount'] \
                                    if len(r.hgetall("mf:" + str(x))) >=1 else None)
    recalls['movieAvgRating'] = recalls['movieid'].map(lambda x :r.hgetall("mf:" + str(x))['movieAvgRating'] \
                                    if len(r.hgetall("mf:" + str(x))) >=1 else None)
    recalls['movieRatingStddev'] = recalls['movieid'].map(lambda x :r.hgetall("mf:" + str(x))['movieRatingStddev'] \
                                    if len(r.hgetall("mf:" + str(x))) >=1 else None)
    recalls.to_csv(config_get('data_source','process_svdr_datas'),index=False)

if __name__ == '__main__':
    recall_process()