import time
from helpers.etl import ETL
from helpers.evaluate import Evaluate
from helpers.predict_and_forecast import PredictAndForecast
from helpers.visualize import plot_results
from models.lstm import build_lstm
from models.transformer import build_transfromer, fit_transformer
import sys

if sys.version_info[0] >= 3:
    sys.stdout = sys.stdout.detach()
    sys.stdin = sys.stdin.detach()
    sys.stdout = open(sys.stdout.fileno(), 'w', encoding='utf-8', closefd=False)
    sys.stdin = open(sys.stdin.fileno(), 'r', encoding='utf-8', closefd=False)


def prediction(ticker='AAPL'):
    # 1. Get Data
    data = ETL(ticker)

    # 2. Implementing an LSTM baseline
    baseline = build_lstm(data)
    baseline_model = baseline[0]
    history = baseline[1]
    baseline_model.summary()

    # 3. Implementing a Transformer
    transformer = build_transfromer(head_size=128, num_heads=4, ff_dim=2, num_trans_blocks=4, mlp_units=[256],
                                    mlp_dropout=0.10, dropout=0.10, attention_axes=1)
    transformer.summary()
    hist = fit_transformer(transformer, data)

    # Inference on our Models
    start = time.time()
    baseline_preds = PredictAndForecast(baseline_model, data.train, data.test)
    print(time.time() - start)

    start = time.time()
    transformer_preds = PredictAndForecast(transformer, data.train, data.test)
    print(time.time() - start)

    # 4. Evaluating the Set
    baseline_evals = Evaluate(data.test, baseline_preds.predictions)
    transformer_evals = Evaluate(data.test, transformer_preds.predictions)
    baseline_evals.mape, transformer_evals.mape
    baseline_evals.var_ratio, transformer_evals.var_ratio

    # Visualize the Sets
    # plot = plot_results(data.test, baseline_preds.predictions, transformer_preds.predictions, data.df, title_suffix='LSTM', xlabel='AAPL stock Price')
    # trans_figure = plot_results(data.test, transformer_preds.predictions, data.df, title_suffix='Transformer', xlabel='AAPL stock Price')

    return data, baseline_preds, transformer_preds