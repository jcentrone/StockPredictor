import matplotlib.pyplot as plt
from matplotlib.figure import Figure


def plot_results(test, lstm_preds, trans_preds, df, title_suffix=None, xlabel=None):
    """
    Returns a matplotlib figure containing the plot of actual values, LSTM predictions, and Transformer predictions.
    """
    fig = Figure(figsize=(20, 6))
    ax = fig.subplots()
    x = df[-(test.shape[0] * test.shape[1]):].index
    test = test.reshape((test.shape[0] * test.shape[1], 1))
    lstm_preds = lstm_preds.reshape((test.shape[0] * test.shape[1], 1))
    trans_preds = trans_preds.reshape((test.shape[0] * test.shape[1], 1))
    ax.plot(x, test, label='Actual', color='red')
    ax.plot(x, lstm_preds, label='LSTM Predictions', color='green')
    ax.plot(x, trans_preds, label='Transformer Predictions', color='blue')
    ax.set_title('Predictions vs. Actual' + (f', {title_suffix}' if title_suffix else ''))
    ax.set_xlabel('Date')
    ax.set_ylabel(xlabel or 'Price')
    ax.legend()
    return fig


# def plot_results(test, preds, df, image_path=None, title_suffix=None, xlabel=None):
#     """
#     Plots training data in blue, actual values in red, and predictions in green,
#     over time.
#     """
#     fig, ax = plt.subplots(figsize=(20, 6))
#     # x = df.Close[-498:].index
#     plot_test = test[1:]
#     plot_preds = preds[1:]
#     x = df[-(plot_test.shape[0] * plot_test.shape[1]):].index
#     plot_test = plot_test.reshape((plot_test.shape[0] * plot_test.shape[1], 1))
#     plot_preds = plot_preds.reshape((plot_test.shape[0] * plot_test.shape[1], 1))
#     ax.plot(x, plot_test, label='actual')
#     ax.plot(x, plot_preds, label='preds')
#     if title_suffix == None:
#         ax.set_title('Predictions vs. Actual')
#     else:
#         ax.set_title(f'Predictions vs. Actual, {title_suffix}')
#     ax.set_xlabel('Date')
#     ax.set_ylabel(xlabel)
#     ax.legend()
#     if image_path != None:
#         imagedir = '/content/drive/MyDrive/Colab Notebooks/images'
#         plt.savefig(f'{imagedir}/{image_path}.png')
#     plt.show()
