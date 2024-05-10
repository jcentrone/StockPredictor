import tensorflow as tf
from keras import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.callbacks import EarlyStopping


def build_lstm(etl, epochs=25, batch_size=32) -> tuple[tf.keras.Model, tf.keras.callbacks.History]:
    """
    Builds, compiles, and fits our LSTM baseline model.
    """
    n_timesteps, n_features, n_outputs = 5, 1, 5
    callbacks = [EarlyStopping(patience=10, restore_best_weights=True)]
    model = Sequential()
    model.add(LSTM(200, activation='relu', input_shape=(n_timesteps, n_features)))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(n_outputs))
    print('compiling baseline model...')
    model.compile(optimizer='adam', loss='mse', metrics=['mae', 'mape'])
    print('fitting model...')
    history = model.fit(etl.X_train, etl.y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_data=(etl.X_test, etl.y_test),
                        verbose=1,
                        callbacks=callbacks)
    return model, history

