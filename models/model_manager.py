import os

from tensorflow.keras.models import load_model

from models.lstm import build_lstm
from models.transformer import build_transfromer


def save_model(model, model_type, ticker):
    model.save(f'models/trained_models/{ticker}_{model_type}.keras')


def get_or_build_model(model_type, ticker, etl):
    model_path = f'models/trained_models/{ticker}_{model_type}.keras'
    if os.path.exists(model_path):
        print(f'Loading {model_type} model for {ticker}...')
        return load_model(model_path)
    else:
        print(f'Building new {model_type} model for {ticker}...')
        if model_type == 'lstm':
            model, _ = build_lstm(etl)
        elif model_type == 'transformer':
            model = build_transfromer(head_size=128, num_heads=4, ff_dim=2, num_trans_blocks=4, mlp_units=[256],
                                      mlp_dropout=0.10, dropout=0.10, attention_axes=1)
        save_model(model, model_type, ticker)
        return model
