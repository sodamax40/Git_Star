#from workerA import get_accuracy, get_predictions
from sklearn.preprocessing import StandardScaler
import joblib
from sklearn.metrics import r2_score
import pandas as pd

from flask import (
    Flask,
    request,
    jsonify,
    Markup,
    render_template
)

#app = Flask(__name__, template_folder='./templates',static_folder='./static')
app = Flask(__name__)


# the data file name!
data_file = 'repositories.csv'
# the model name we load!
model = 'rfc_model.m'

# df = pd.read_csv('repositories.csv', na_values=['jquery-database'])
# print(df)


def load_data():
    data = pd.read_csv(data_file).iloc[:, 1:]
    X = data.drop(['star_count'], axis=1)
    y = data.star_count
    s = StandardScaler()
    X = X.astype(float)
    X = s.fit_transform(X)
    return X, y


def load_model():
    loaded_model = joblib.load(model)
    return loaded_model


@app.route("/")
def index():
    return '<h1>DataEngineering II Project: GitStar predictor.</h1>'


@app.route("/R2", methods=['POST', 'GET'])
def accuracy():
    if request.method == 'POST':
        # r = get_accuracy.delay()
        # a = r.get()
        X, y = load_data()
        loaded_model = load_model()
        predictions = loaded_model.predict(X)
        test_score = r2_score(y, predictions)
        a = test_score
        return '<h1>The R2 score is {}</h1>'.format(a)

    return '''<form method="POST"><input type="submit"></form>'''


@app.route("/predictions", methods=['POST', 'GET'])
def predictions():
    if request.method == 'POST':
        # results = get_predictions.delay()
        # predictions = results.get()
        # results = get_accuracy.delay()
        # accuracy = results.get()
        # final_results = predictions
        results = {}
        X, y = load_data()
        loaded_model = load_model()
        predictions = loaded_model.predict(X)

        test_score = r2_score(y, predictions)
        a = test_score
        accuracy = a
        results['predicted'] = []
        results['y'] = []
        for i in range(5):
            results['y'].append(y[i].tolist())
            results['predicted'].append(predictions[i].tolist())
        final_results = results
        return render_template('result.html', accuracy=accuracy, final_results=final_results)

    return '''<form method="POST"><input type="submit"></form>'''


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5100, debug=True)
