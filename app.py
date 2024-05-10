from flask import Flask, request, jsonify, render_template

from helpers.etl import ETL
from helpers.predict_and_forecast import PredictAndForecast
from models.model_manager import get_or_build_model
from prediction import prediction

app = Flask(__name__)


@app.route('/')
def index():  # put application's code here
    return render_template("index.html")


@app.route('/predict')
def predict():
    ticker = request.args.get('ticker')
    etl_instance = ETL(ticker)  # ETL setup

    lstm_model = get_or_build_model('lstm', ticker, etl_instance)
    transformer_model = get_or_build_model('transformer', ticker, etl_instance)

    lstm_preds = PredictAndForecast(lstm_model, etl_instance.X_train, etl_instance.X_test).predictions.tolist()
    transformer_preds = PredictAndForecast(transformer_model, etl_instance.X_train, etl_instance.X_test).predictions.tolist()

    return jsonify({
        'dates': etl_instance.df.index.strftime('%Y-%m-%d').tolist(),
        'data': etl_instance.df.tolist(),
        'lstm_preds': lstm_preds,
        'transformer_preds': transformer_preds
    })


# @app.route('/predict')
# def predict():
#     ticker = request.args.get('ticker')
#
#     etl_instance, baseline_preds, transformer_preds = prediction(ticker)
#
#     # Assuming 'baseline_preds' and 'transformer_preds' are instances of PredictAndForecast
#     baseline_list = baseline_preds.predictions.tolist()
#     transformer_list = transformer_preds.predictions.tolist()
#
#     return jsonify({
#         'dates': etl_instance.df.index.strftime('%Y-%m-%d').tolist(),  # Format dates for JSON serialization
#         'data': etl_instance.df.tolist(),  # Convert DataFrame or Series to list
#         'baseline_preds': baseline_list,
#         'transformer_preds': transformer_list
#     })


if __name__ == '__main__':
    app.run()
