import tensorflow as tf
import pandas as pd
allSamples = pd.read_csv("D:/python/SparrowRecSys/target/classes/webroot/test0427/allSamples.csv")


dftraningSamples = pd.read_csv("D:/python/SparrowRecSys/target/classes/webroot/test0427/trainingSamples.csv")

dftraningSamples = dftraningSamples.rename(columns = {'用户ID': "userid"})
# dftraningSamples = dftraningSamples.dropna(how='any')
dftraningSamples.to_csv("D:/python/SparrowRecSys/target/classes/webroot/test0427/trainingSamples.csv",index=False)
dftestSamples = pd.read_csv("D:/python/SparrowRecSys/target/classes/webroot/test0427/testSamples.csv")
dftestSamples = dftestSamples.rename(columns = {'用户ID': "userid"})
# dftestSamples = dftestSamples.dropna(how='any')
dftestSamples.to_csv("D:/python/SparrowRecSys/target/classes/webroot/test0427/testSamples.csv",index=False)

training_samples_file_path = "D:/python/SparrowRecSys/target/classes/webroot/test0427/trainingSamples.csv"
test_samples_file_path =  "D:/python/SparrowRecSys/target/classes/webroot/test0427/testSamples.csv"

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




# split as test dataset and training dataset
train_dataset = get_dataset(training_samples_file_path)
test_dataset = get_dataset(test_samples_file_path)

#电影名必须转换成电影ID



# movie id embedding feature
#categorical_column_with_identity：把numerical data转乘one hot encoding
movie_col = tf.feature_column.categorical_column_with_identity(key='movieid', num_buckets=80000)
movie_emb_col = tf.feature_column.embedding_column(movie_col, 10)

# user id embedding feature
user_col = tf.feature_column.categorical_column_with_identity(key='userid', num_buckets=80000)
user_emb_col = tf.feature_column.embedding_column(user_col, 10)

# define input for keras model
inputs = {
    'movieid': tf.keras.layers.Input(name='movieid', shape=(), dtype='int32'),
    'userid': tf.keras.layers.Input(name='userid', shape=(), dtype='int32'),
}


# neural cf model arch two. only embedding in each tower, then MLP as the interaction layers
def neural_cf_model_1(feature_inputs, item_feature_columns, user_feature_columns, hidden_units):
    item_tower = tf.keras.layers.DenseFeatures(item_feature_columns)(feature_inputs)
    user_tower = tf.keras.layers.DenseFeatures(user_feature_columns)(feature_inputs)
    interact_layer = tf.keras.layers.concatenate([item_tower, user_tower])
    for num_nodes in hidden_units:
        interact_layer = tf.keras.layers.Dense(num_nodes, activation='relu')(interact_layer)
    output_layer = tf.keras.layers.Dense(1, activation='sigmoid')(interact_layer)
    neural_cf_model = tf.keras.Model(feature_inputs, output_layer)
    return neural_cf_model


# neural cf model arch one. embedding+MLP in each tower, then dot product layer as the output
def neural_cf_model_2(feature_inputs, item_feature_columns, user_feature_columns, hidden_units):
    item_tower = tf.keras.layers.DenseFeatures(item_feature_columns)(feature_inputs)
    for num_nodes in hidden_units:
        item_tower = tf.keras.layers.Dense(num_nodes, activation='relu')(item_tower)

    user_tower = tf.keras.layers.DenseFeatures(user_feature_columns)(feature_inputs)
    for num_nodes in hidden_units:
        user_tower = tf.keras.layers.Dense(num_nodes, activation='relu')(user_tower)

    output = tf.keras.layers.Dot(axes=1)([item_tower, user_tower])
    output = tf.keras.layers.Dense(1, activation='sigmoid')(output)

    neural_cf_model = tf.keras.Model(feature_inputs, output)
    return neural_cf_model


# neural cf model architecture
model = neural_cf_model_1(inputs, [movie_emb_col], [user_emb_col], [10, 10])

# compile the model, set loss function, optimizer and evaluation metrics
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy', tf.keras.metrics.AUC(curve='ROC'), tf.keras.metrics.AUC(curve='PR')])

# train the model
model.fit(train_dataset, epochs=1)

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
print('before save')
#model.save('D:/python/SparrowRecSys/target/classes/webroot/ourmodeldata/neuralcf/002/model0429.h5')
tf.keras.models.save_model(
    model,
    "D:/python/SparrowRecSys/target/classes/webroot/ourmodeldata/neuralcf/002/",
    overwrite=True,
    include_optimizer=True,
    save_format=None,
    signatures=None,
    options=None
)