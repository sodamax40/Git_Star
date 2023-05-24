from keras.models import model_from_json
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, KFold
from keras.wrappers.scikit_learn import KerasRegressor
from keras.layers import Dense, Dropout, BatchNormalization
from keras.models import Sequential
from sklearn.ensemble import RandomForestRegressor
#from catboost import CatBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
import numpy as np
import pandas as pd
import joblib
import pickle

# Visualisation
#import matplotlib as mpl
#import matplotlib.pyplot as plt
#import matplotlib.pylab as pylab
#import seaborn as sns

# Configure visualisations
#color = sns.color_palette()
pd.options.mode.chained_assignment = None
pd.options.display.max_columns = 999
# mpl.style.use( 'ggplot' )
# sns.set_style('whitegrid')
#pylab.rcParams['figure.figsize'] = 10, 8
seed = 7

# importing libraries


np.random.seed(seed)

data = pd.read_csv('repositories.csv').iloc[:, 1:]

X = data.drop(['star_count'], axis=1)
y = data.star_count

s = StandardScaler()
X = s.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42)

training_scores = []
test_scores = []
models = []


models.append("KNN")

neigh = KNeighborsRegressor(n_neighbors=7)
neigh.fit(X_train, y_train)
training_score = neigh.score(X_train, y_train)
test_score = neigh.score(X_test, y_test)
training_scores.append(training_score)
test_scores.append(test_score)
print("training set performance: ", training_score)
print("test set performance: ", test_score)

models.append("gradient boost")

reg = GradientBoostingRegressor(verbose=1, n_estimators=200)
reg.fit(X_train, y_train)

training_score = reg.score(X_train, y_train)
test_score = reg.score(X_test, y_test)

training_scores.append(training_score)
test_scores.append(test_score)
print("training set performance: ", training_score)
print("test set performance: ", test_score)

models.append("random forest")

model_forest = RandomForestRegressor(
    n_jobs=-1, n_estimators=10, verbose=1, random_state=seed)
model_forest.fit(X_train, y_train)

result = model_forest.predict(X[7].reshape(1, -1))
print(result)

training_score = model_forest.score(X_train, y_train)
test_score = model_forest.score(X_test, y_test)

print("training score: ", training_score)
print("test score: ", test_score)

training_scores.append(training_score)
test_scores.append(test_score)

models.append("neural network")


def baseline_model():
    model = Sequential()
    model.add(Dense(100, input_dim=12, activation='relu',
              kernel_initializer='glorot_normal'))
    model.add(Dropout(0.2))
    model.add(Dense(80, activation='relu', kernel_initializer='glorot_normal'))
    model.add(Dropout(0.2))
    model.add(Dense(60, activation='relu', kernel_initializer='glorot_normal'))
    model.add(Dropout(0.2))
    model.add(Dense(40, activation='relu', kernel_initializer='glorot_normal'))
    model.add(Dropout(0.2))
    model.add(Dense(20, activation='relu', kernel_initializer='glorot_normal'))
    model.add(Dropout(0.2))
    model.add(Dense(10, activation='relu', kernel_initializer='glorot_normal'))
    model.add(Dropout(0.2))
    model.add(Dense(1, kernel_initializer='glorot_normal'))

    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


estimator = KerasRegressor(build_fn=baseline_model,
                           nb_epoch=200, epochs=10, batch_size=32, verbose=True)
# kfold = KFold(n_splits=10, random_state=seed)
# results = cross_val_score(estimator, X_train.values, y_train.values, cv=kfold)
# print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))

estimator.fit(X_train, y_train)

train_pred = estimator.predict(X_train)
test_pred = estimator.predict(X_test)

training_score = r2_score(y_train, train_pred)
test_score = r2_score(y_test, test_pred)

print("training score: ", training_score)
print("test score: ", test_score)

training_scores.append(training_score)
test_scores.append(test_score)

# models.append("cat boost")

# model_cat = CatBoostRegressor(
#     iterations=440, depth=8, learning_rate=0.1, loss_function='RMSE', use_best_model=True)
# model_cat.fit(X_train[:90503], y_train[:90503], eval_set=(
#     X_train[90503:], y_train[90503:]), plot=True)

# y_train_pred = model_cat.predict(X_train)
# y_pred = model_cat.predict(X_test)

# train_score = r2_score(y_train, y_train_pred)
# test_score = r2_score(y_test, y_pred)

# training_scores.append(training_score)
# test_scores.append(test_score)
# print("Training score - " + str(train_score))
# print("Test score - " + str(test_score))

print(models)
print(training_scores)
print(test_scores)

# serialize model to JSON
model_json = estimator.model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
estimator.model.save_weights("model.h5")
print("Saved model to disk")


# save rf model
RFC_pickle = pickle.dumps(model_forest)
joblib.dump(model_forest, "rfc_model.m")

# save gradien boost model
GBR_pickle = pickle.dumps(reg)
joblib.dump(reg, "gdbt_model.m")


# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

# make class predictions with the model
predictions = np.round(loaded_model.predict(X)).flatten().astype(np.int32)
# summarize the first 5 cases
for i in range(5):
    print('%s => %d (expected %d)' % (X[i].tolist(), predictions[i], y[i]))
