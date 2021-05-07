import tensorflow as tf
import os
import pickle
import pandas as pd
import redis
training_samples_file_path = "D:/python/SparrowRecSys/target/classes/webroot/test0427/trainingSamplestttt.csv"
test_samples_file_path =  "D:/python/SparrowRecSys/target/classes/webroot/test0427/testSamplesttt.csv"
all_samples_file_path = "D:/python/SparrowRecSys/target/classes/webroot/test0427/alldatastttt.csv"
dict_actortoindexpath = './cache/dict_actortoindex.pkl'
dict_indextoacotrpath = './cache/dict_indextoacotr.pkl'
dict_directortoindexpath = './cache/dict_directortoindex.pkl'
dict_indextodirecotrpath = './cache/dict_indextodirecotr.pkl'

# load sample as tf dataset
def get_dataset(file_path):
    dataset = tf.data.experimental.make_csv_dataset(
        file_path,
        batch_size=12,
        label_name='label',
        na_value="0",
        num_epochs=1,
        ignore_errors=True)
    return dataset

def get_predataset(file_path):
    dataset = tf.data.experimental.make_csv_dataset(
        file_path,
        batch_size=12,
        label_name=None,
        na_value="0",
        num_epochs=1,
        ignore_errors=True)
    return dataset

flag = 'predict'
if flag == 'debug':
    if os.path.exists(dict_actortoindexpath):
        pass
    dict_actortoindex = pickle.load(open(dict_actortoindexpath, 'rb'))
    dict_indextoacotr = pickle.load(open(dict_indextoacotrpath, 'rb'))
    dict_directortoindex = pickle.load(open(dict_directortoindexpath, 'rb'))
    dict_indextodirecotr = pickle.load(open(dict_indextodirecotrpath, 'rb'))
    pool = redis.ConnectionPool(host='127.0.0.1', port=6379,decode_responses=True, encoding='UTF-8')
    r = redis.Redis(connection_pool=pool)
    # traindatas = pd.read_csv(training_samples_file_path)
    # traindatas['movieactor1'] = traindatas['movieid'].map(lambda x :r.hgetall("mf:" + str(x))['movieactor1'] \
    #                                 if len(r.hgetall("mf:" + str(x))) >=1 else None) \
    #                                 .map(lambda x: dict_actortoindex[x] if len(x) >= 1 else None)
    # traindatas['movieactor2'] = traindatas['movieid'].map(lambda x :r.hgetall("mf:" + str(x))['movieactor2'] \
    #                                 if len(r.hgetall("mf:" + str(x))) >=1 else None)\
    #                             .map(lambda x: dict_actortoindex[x] if  len(x)>=1    else None)
    #
    # traindatas['movieactor3'] = traindatas['movieid'].map(lambda x :r.hgetall("mf:" + str(x))['movieactor3'] \
    #                                 if len(r.hgetall("mf:" + str(x))) >=1 else None) \
    #                             .map(lambda x: dict_actortoindex[x] if len(x) >= 1 else None)
    # traindatas['movieactor4'] = traindatas['movieid'].map(lambda x :r.hgetall("mf:" + str(x))['movieactor4'] \
    #                                 if len(r.hgetall("mf:" + str(x))) >=1 else None) \
    #                             .map(lambda x: dict_actortoindex[x] if len(x) >= 1 else None)
    # traindatas['moviedirecotr1'] = traindatas['movieid'].map(lambda x :r.hgetall("mf:" + str(x))['moviedirecotr1'] \
    #                                 if len(r.hgetall("mf:" + str(x))) >=1 else None)\
    #                                  .map(lambda x:dict_directortoindex[x] if len(x) >= 1 else None)
    #
    # #traindatas.to_csv("D:/python/SparrowRecSys/target/classes/webroot/test0427/trainingSamplestttt.csv",index=False)
    #
    # testdatas = pd.read_csv(test_samples_file_path)
    # testdatas['movieactor1'] = testdatas['movieid'].map(lambda x :r.hgetall("mf:" + str(x))['movieactor1'] \
    #                                 if len(r.hgetall("mf:" + str(x))) >=1 else None) \
    #                                 .map(lambda x: dict_actortoindex[x] if len(x) >= 1 else None)
    # testdatas['movieactor2'] = testdatas['movieid'].map(lambda x :r.hgetall("mf:" + str(x))['movieactor2'] \
    #                                 if len(r.hgetall("mf:" + str(x))) >=1 else None)\
    #                             .map(lambda x: dict_actortoindex[x] if  len(x)>=1    else None)
    #
    # testdatas['movieactor3'] = testdatas['movieid'].map(lambda x :r.hgetall("mf:" + str(x))['movieactor3'] \
    #                                 if len(r.hgetall("mf:" + str(x))) >=1 else None) \
    #                             .map(lambda x: dict_actortoindex[x] if len(x) >= 1 else None)
    # testdatas['movieactor4'] = testdatas['movieid'].map(lambda x :r.hgetall("mf:" + str(x))['movieactor4'] \
    #                                 if len(r.hgetall("mf:" + str(x))) >=1 else None) \
    #                             .map(lambda x: dict_actortoindex[x] if len(x) >= 1 else None)
    # testdatas['moviedirecotr1'] = testdatas['movieid'].map(lambda x :r.hgetall("mf:" + str(x))['moviedirecotr1'] \
    #                                 if len(r.hgetall("mf:" + str(x))) >=1 else None)\
    #                                  .map(lambda x:dict_directortoindex[x] if len(x) >= 1 else None)
    #
    # #testdatas.to_csv("D:/python/SparrowRecSys/target/classes/webroot/test0427/testSamplesttt.csv",index=False)
    alldatas = pd.read_csv("D:/python/SparrowRecSys/target/classes/webroot/test0427/allSamples.csv")
    alldatas['movieactor1'] = alldatas['movieid'].map(lambda x :r.hgetall("mf:" + str(x))['movieactor1'] \
                                    if len(r.hgetall("mf:" + str(x))) >=1 else None) \
                                    .map(lambda x: dict_actortoindex[x] if len(x) >= 1 else None)
    alldatas['movieactor2'] = alldatas['movieid'].map(lambda x :r.hgetall("mf:" + str(x))['movieactor2'] \
                                    if len(r.hgetall("mf:" + str(x))) >=1 else None)\
                                .map(lambda x: dict_actortoindex[x] if  len(x)>=1    else None)

    alldatas['movieactor3'] = alldatas['movieid'].map(lambda x :r.hgetall("mf:" + str(x))['movieactor3'] \
                                    if len(r.hgetall("mf:" + str(x))) >=1 else None) \
                                .map(lambda x: dict_actortoindex[x] if len(x) >= 1 else None)
    alldatas['movieactor4'] = alldatas['movieid'].map(lambda x :r.hgetall("mf:" + str(x))['movieactor4'] \
                                    if len(r.hgetall("mf:" + str(x))) >=1 else None) \
                                .map(lambda x: dict_actortoindex[x] if len(x) >= 1 else None)
    alldatas['moviedirecotr1'] = alldatas['movieid'].map(lambda x :r.hgetall("mf:" + str(x))['moviedirecotr1'] \
                                    if len(r.hgetall("mf:" + str(x))) >=1 else None)\
                                     .map(lambda x:dict_directortoindex[x] if len(x) >= 1 else None)

    alldatas.to_csv("D:/python/SparrowRecSys/target/classes/webroot/test0427/alldatastttt.csv",index=False)

elif  flag == 'train':
    # split as test dataset and training dataset
    train_dataset = get_dataset(training_samples_file_path)
    test_dataset = get_dataset(test_samples_file_path)
    all_dataset = get_dataset(all_samples_file_path)
    feature_vocab = ['经典', '青春', '搞笑', '文艺', '魔幻', '女性', '励志', '黑帮', '感人']
    type_vocab = ['剧情', '爱情', '喜剧', '动作', '犯罪', '惊悚', '奇幻', '悬疑',
                  '冒险', '科幻', '音乐', '传记', '恐怖', '历史', '战争', '歌舞 ',
                  '武侠', '西部', '灾难']
    area_vocab = ['美国', '中国大陆', '日本', '香港', '法国', '英国', '韩国', '德国',
                  '台湾', '加拿大', '意大利', '印度', '西班牙', '泰国', '澳大利亚', '俄罗斯',
                  '爱尔兰', '瑞典', '丹麦', '巴西', '伊朗']

    GENRE_FEATURES = {
        'userPositiveType1': type_vocab,
        'userPositiveType2': type_vocab,
        'userPositiveType3': type_vocab,
        'userPositiveType4': type_vocab,
        'userPositiveType5': type_vocab,
        'userPositivefeature1': feature_vocab,
        'userPositivefeature2': feature_vocab,
        'userPositivefeature3': feature_vocab,
        'userPositivefeature4': feature_vocab,
        'userPositivefeature5': feature_vocab,
        'userPositivearea1': area_vocab,
        'userPositivearea2': area_vocab,
        'userPositivearea3': area_vocab,
        'userPositivearea4': area_vocab,
        'userPositivearea5': area_vocab
    }

    # all categorical features
    categorical_columns = []
    for feature, vocab in GENRE_FEATURES.items():
        cat_col = tf.feature_column.categorical_column_with_vocabulary_list(
            key=feature, vocabulary_list=vocab)
        emb_col = tf.feature_column.embedding_column(cat_col, 10)
        categorical_columns.append(emb_col)
    # movie id embedding feature

    movie_col = tf.feature_column.categorical_column_with_identity(key='movieid', num_buckets=90000)
    movie_emb_col = tf.feature_column.embedding_column(movie_col, 10)
    categorical_columns.append(movie_emb_col)

    # user id embedding feature
    user_col = tf.feature_column.categorical_column_with_identity(key='userid', num_buckets=90000)
    user_emb_col = tf.feature_column.embedding_column(user_col, 10)
    categorical_columns.append(user_emb_col)

    # the process of actor and director
    for column in ['movieactor1', 'movieactor2', 'movieactor3', 'movieactor4', \
                   'moviedirecotr1']:
        movie_actor1 = tf.feature_column.categorical_column_with_identity(key=column, num_buckets=90000)
        movie_emb_col1 = tf.feature_column.embedding_column(movie_actor1, 10)
        categorical_columns.append(movie_emb_col1)

    # all numerical features
    numerical_columns = [
        tf.feature_column.numeric_column('movieRatingCount'),
        tf.feature_column.numeric_column('movieAvgRating'),
        tf.feature_column.numeric_column('movieRatingStddev'),
        tf.feature_column.numeric_column('userRatingCount'),
        tf.feature_column.numeric_column('userAvgRating'),
        tf.feature_column.numeric_column('userRatingStddev')]

    # cross feature between current movie and user historical movie
    rated_movie = tf.feature_column.categorical_column_with_identity(key='userRatedMovie1', num_buckets=90000)

    crossed_feature = tf.feature_column.indicator_column(
        tf.feature_column.crossed_column([movie_col, rated_movie], 90000))

    # define input for keras model
    inputs = {
        'movieAvgRating': tf.keras.layers.Input(name='movieAvgRating', shape=(), dtype='float32'),
        'movieRatingStddev': tf.keras.layers.Input(name='movieRatingStddev', shape=(), dtype='float32'),
        'movieRatingCount': tf.keras.layers.Input(name='movieRatingCount', shape=(), dtype='int32'),
        'userAvgRating': tf.keras.layers.Input(name='userAvgRating', shape=(), dtype='float32'),
        'userRatingStddev': tf.keras.layers.Input(name='userRatingStddev', shape=(), dtype='float32'),
        'userRatingCount': tf.keras.layers.Input(name='userRatingCount', shape=(), dtype='int32'),
        'movieid': tf.keras.layers.Input(name='movieid', shape=(), dtype='int32'),
        'userid': tf.keras.layers.Input(name='userid', shape=(), dtype='int32'),
        'movieactor1': tf.keras.layers.Input(name='movieactor1', shape=(), dtype='int32'),
        'movieactor2': tf.keras.layers.Input(name='movieactor2', shape=(), dtype='int32'),
        'movieactor3': tf.keras.layers.Input(name='movieactor3', shape=(), dtype='int32'),
        'movieactor4': tf.keras.layers.Input(name='movieactor4', shape=(), dtype='int32'),
        'moviedirecotr1': tf.keras.layers.Input(name='moviedirecotr1', shape=(), dtype='int32'),
        'userRatedMovie1': tf.keras.layers.Input(name='userRatedMovie1', shape=(), dtype='int32'),
        'userPositiveType1': tf.keras.layers.Input(name='userPositiveType1', shape=(), dtype='string'),
        'userPositiveType2': tf.keras.layers.Input(name='userPositiveType2', shape=(), dtype='string'),
        'userPositiveType3': tf.keras.layers.Input(name='userPositiveType3', shape=(), dtype='string'),
        'userPositiveType4': tf.keras.layers.Input(name='userPositiveType4', shape=(), dtype='string'),
        'userPositiveType5': tf.keras.layers.Input(name='userPositiveType5', shape=(), dtype='string'),
        'userPositivefeature1': tf.keras.layers.Input(name='userPositivefeature1', shape=(), dtype='string'),
        'userPositivefeature2': tf.keras.layers.Input(name='userPositivefeature2', shape=(), dtype='string'),
        'userPositivefeature3': tf.keras.layers.Input(name='userPositivefeature3', shape=(), dtype='string'),
        'userPositivefeature4': tf.keras.layers.Input(name='userPositivefeature4', shape=(), dtype='string'),
        'userPositivefeature5': tf.keras.layers.Input(name='userPositivefeature5', shape=(), dtype='string'),
        'userPositivearea1': tf.keras.layers.Input(name='userPositivearea1', shape=(), dtype='string'),
        'userPositivearea2': tf.keras.layers.Input(name='userPositivearea2', shape=(), dtype='string'),
        'userPositivearea3': tf.keras.layers.Input(name='userPositivearea3', shape=(), dtype='string'),
        'userPositivearea4': tf.keras.layers.Input(name='userPositivearea4', shape=(), dtype='string'),
        'userPositivearea5': tf.keras.layers.Input(name='userPositivearea5', shape=(), dtype='string'),
    }

    # wide and deep model architecture
    # deep part for all input features
    deep = tf.keras.layers.DenseFeatures(numerical_columns + categorical_columns)(inputs)
    deep = tf.keras.layers.Dense(128, activation='relu')(deep)
    deep = tf.keras.layers.Dense(128, activation='relu')(deep)
    # wide part for cross feature
    wide = tf.keras.layers.DenseFeatures(crossed_feature)(inputs)
    both = tf.keras.layers.concatenate([deep, wide])
    output_layer = tf.keras.layers.Dense(1, activation='sigmoid')(both)
    model = tf.keras.Model(inputs, output_layer)

    # compile the model, set loss function, optimizer and evaluation metrics
    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy', tf.keras.metrics.AUC(curve='ROC'), tf.keras.metrics.AUC(curve='PR')])

    # train the model
    model.fit(all_dataset, epochs=5)
    tf.keras.models.save_model(
        model,
        "D:/python/SparrowRecSys/target/classes/webroot/ourmodeldata/wide&deep/",
        overwrite=True,
        include_optimizer=True,
        save_format=None,
        signatures=None,
        options=None
    )

    # evaluate the model
    test_loss, test_accuracy, test_roc_auc, test_pr_auc = model.evaluate(test_dataset)
    print('\n\nTest Loss {}, Test Accuracy {}, Test ROC AUC {}, Test PR AUC {}'.format(test_loss, test_accuracy,
                                                                                       test_roc_auc, test_pr_auc))

    # print some predict results
    predictions = model.predict(test_dataset)
    for prediction, goodRating in zip(predictions[:12], list(test_dataset)[0][1][:12]):
        print("Predicted good rating: {:.2%}".format(prediction[0]),
              " | Actual rating label: ",
              ("Good Rating" if bool(goodRating) else "Bad Rating"))
elif  flag == 'predict':
    recalldatapath = "D:/python/Project/2_recsys_model/cache/recalldatas0506.csv"
    recalldatas = pd.read_csv(recalldatapath)
    predict_dataset = get_predataset(recalldatapath)
    model = tf.keras.models.load_model("D:/python/SparrowRecSys/target/classes/webroot/ourmodeldata/wideanddeep/1/")
    predictions = model.predict(predict_dataset)
    assert recalldatas.shape[0] == predictions.shape[0]
    recalldatas['predictions'] =  predictions
    recalldatas['label'] = recalldatas['predictions'].map(lambda x : 1 if x >= 0.7 else 0)
    print(recalldatas.head())
    recalldatas.to_csv("D:/python/Project/2_recsys_model/cache/preresofrecdata0506.csv")