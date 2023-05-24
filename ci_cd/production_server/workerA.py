from celery import Celery
import pandas as pd
#import numpy as np
#import pickle
#from numpy import loadtxt
import numpy as np
#from tensorflow.keras.models import model_from_json
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import joblib

# model_json_file = './model.json'
# model_weights_file = './model.h5'
# data_file = './pima-indians-diabetes.csv'

# the data file name!
data_file = './repositories.csv'
# the model name we load!
model = './rfc_model.m'


def load_data():
    data = pd.read_csv(data_file).iloc[:, 1:]
    X = data.drop(['star_count'], axis=1)
    y = data.star_count
    s = StandardScaler()
    X = s.fit_transform(X)
    return X, y


def load_model():
    # # load json and create model
    # json_file = open(model_json_file, 'r')
    # loaded_model_json = json_file.read()
    # json_file.close()
    # loaded_model = model_from_json(loaded_model_json)
    # # load weights into new model
    # loaded_model.load_weights(model_weights_file)
    # #print("Loaded model from disk")
    loaded_model = joblib.load(model)
    return loaded_model


# Celery configuration
CELERY_BROKER_URL = 'amqp://rabbitmq:rabbitmq@rabbit:5672/'
CELERY_RESULT_BACKEND = 'rpc://'
# Initialize Celery
celery = Celery('workerA', broker=CELERY_BROKER_URL,
                backend=CELERY_RESULT_BACKEND)


@celery.task()
def add_nums(a, b):
    return a + b


@celery.task
def get_predictions():
    # results = {}
    # X, y = load_data()
    # loaded_model = load_model()
    # predictions = np.round(loaded_model.predict(X)).flatten().astype(np.int32)
    # results['y'] = y.tolist()
    # results['predicted'] = predictions.tolist()
    #print ('results[y]:', results['y'])
    # for i in range(len(results['y'])):
    #print('%s => %d (expected %d)' % (X[i].tolist(), predictions[i], y[i]))
    # results['predicted'].append(predictions[i].tolist()[0])
    #print ('results:', results)
    results = {}
    X, y = load_data()
    loaded_model = load_model()
    predictions = loaded_model.predict(X)
    results['predicted'] = []
    results['y'] = []
    for i in range(5):
        results['y'].append(y[i].tolist())
        results['predicted'].append(predictions[i].tolist())
    return results


@celery.task
def get_accuracy():
    # X, y = load_data()
    # loaded_model = load_model()
    # loaded_model.compile(loss='binary_crossentropy',
    #                      optimizer='rmsprop', metrics=['accuracy'])

    # score = loaded_model.evaluate(X, y, verbose=0)
    X, y = load_data()
    loaded_model = load_model()
    predictions = loaded_model.predict(X)
    test_score = r2_score(y, predictions)
    #print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
    return test_score
