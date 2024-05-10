from flask import Flask, request, jsonify, render_template

from helpers.etl import ETL
from helpers.predict_and_forecast import PredictAndForecast
from prediction import prediction

app = Flask(__name__)


@app.route('/')
def index():  # put application's code here
    return render_template("index.html")


@app.route('/predict')
def predict():
    ticker = request.args.get('ticker')
    etl_instance, baseline_preds, transformer_preds = prediction(ticker)

    # Correctly calculating the start index for prediction dates
    prediction_start_idx = len(etl_instance.train) - etl_instance.n_input + 1

    # Debug print to check what dates are being calculated
    print("Start index for predictions:", prediction_start_idx)
    print("Actual dates being used:", etl_instance.df.index[prediction_start_idx:prediction_start_idx + len(baseline_preds.predictions)].strftime('%Y-%m-%d').tolist())

    # Fetch dates for predictions
    prediction_dates = etl_instance.df.index[
                       prediction_start_idx:prediction_start_idx + len(baseline_preds.predictions)].strftime(
        '%Y-%m-%d').tolist()

    return jsonify({
        'dates': prediction_dates,
        'data': etl_instance.df[prediction_start_idx:prediction_start_idx + len(baseline_preds.predictions)].tolist(),
        'baseline_preds': baseline_preds.predictions.flatten().tolist(),
        'transformer_preds': transformer_preds.predictions.flatten().tolist()
    })




if __name__ == '__main__':
    app.run()
