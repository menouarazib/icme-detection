import warnings
from datetime import datetime

import click
import matplotlib.pyplot as plt
import pandas as pd
import scipy.constants as constants
from matplotlib import patches
from scipy.ndimage import median_filter
from sklearn.exceptions import DataConversionWarning
from sklearn.externals import joblib

import numpy as np
import logging
from keras.models import load_model

_path_to_models = "./data/models/"
_path_to_scalers = "./data/scalers/"

_step_stride = 6
_max_width = 50


def get_module_logger(mod_name):
    """
    get logger of module
    :param mod_name: the name of module
    :return: logging.getLogger(mod_name)
    """

    logger_ = logging.getLogger(mod_name)
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
    handler.setFormatter(formatter)
    logger_.addHandler(handler)

    file_formatter = logging.Formatter("[%(asctime)s] [%(filename)s:%(lineno)d] %(levelname)s - %(message)s")
    file_handler = logging.FileHandler("logging.log", mode='w')
    file_handler.setFormatter(file_formatter)

    logger_.addHandler(handler)
    logger_.addHandler(file_handler)
    logger_.setLevel(logging.INFO)
    return logger_


logger = get_module_logger(__name__)


def add_extra_features(data):
    """
    In addition to this 30 input variables, we computed 3 additional features that will also serve as input variables

    :param data: dataset is made of 30 primary input variables
    :return: the new dataset with 33 primary input variables
    """
    try:
        # the plasma parameter β
        data['Beta'] = 1e6 * data['Vth'] * data['Vth'] * constants.m_p * data['Np'] * 1e6 * constants.mu_0 / (
                1e-18 * data['B'] * data['B'])
        # the dynamic pressure Pd yn = NpV^2
        data['Pm'] = 1e-18 * data['B'] * data['B'] / (2 * constants.mu_0)
        # the normalized magnetic fluctuations σB
        data['RmsBob'] = np.sqrt(data['Bx_rms'] ** 2 + data['By_rms'] ** 2 + data['Bz_rms'] ** 2) / data['B']
    except KeyError:
        logger.error('Error computing Beta,B,Vth or Np, might not be loaded in dataframe')


def delimit_data(data):
    """
    Delimit data by a  starting and  ending time

    :param data: dataset (features)
    :return: the delimited dataset
    """
    start_time_gautier_ = datetime(1997, 10, 1)
    end_time_gautier_ = datetime(2016, 1, 1)

    data = data[data.index < end_time_gautier_]
    data = data[data.index > start_time_gautier_]
    return data


def resampling_data(data):
    """
    Due to instrumental constraints, holes are present within the whole dataset, the great majority of these holes
    have a duration between 2 and 10 minutes. On the other hand, the crossings of ICMEs with their sheath typically
    have durations of several hours. We therefore resample the data to a 10 minutes resolution, thereby eliminating
    the greatest majority of the holes while still remaining accurate in the determination of start and end times of
    labeled events

    :param data: dataset
    :return: resampled dataset
    """
    return data.resample('10T').mean().dropna()


def sliding_windows(data, window, step=1):
    """
    create a new numpy array with dimension of (((n-m)//step) +1, window, m), where each rwo is a (window,
    m) array. Two successive windows have same rows except the first (step) ones. This is method is used to windowing
    the data. This method is used to split the data set into sliding windows of various sizes (from 1 to 100 Hr)

    https://numpy.org/doc/stable/reference/generated/numpy.lib.stride_tricks.as_strided.html

    :param data: data as a (n,m) numpy array
    :param window: the width of a window
    :param step: step between two windows
    :return: (((n-m)//step) +1, window, m) numpy array

    @see page 61 pdf : https://tel.archives-ouvertes.fr/tel-03198435/document
    """
    new_shape = int((data.shape[0] - window) // step + 1), window, data.shape[1]
    new_strides = (data.strides[0] * step,) + data.strides
    return np.lib.stride_tricks.as_strided(data, shape=new_shape, strides=new_strides)


def predict(data, width):
    """
    Make predictions from a given model (CNN) id and for a given data features

    :param data: data to predict from
    :param width: the id of the CNN to load and it represents also the id of the sliding window
    :return: the similarity parameter for each window size, the size is equal to id window (width) * _step_stride
    """
    window = int(_step_stride * width)
    scale = joblib.load(_path_to_models + 'scaler' + str(width) + '.pkl')
    model = load_model(_path_to_models + 'model' + str(width) + '.h5')
    x_scaled = scale.transform(data)
    x_to_predict = sliding_windows(x_scaled, window=window)
    y_predicted = model.predict(x_to_predict, verbose=0)
    y_to_series = pd.Series(index=data.index[int(window / 2) - 1:-int(window / 2)],
                            data=np.array([y_predicted[i][0] for i in range(0, len(y_predicted))]))
    return y_to_series


def predict_all(data):
    """
     The raw predictions of the similarities for each of 100 CNNs for a given data

    :param data:data to predict from
    :return:  raw predictions obtained from each of our CNN.
    """

    warnings.filterwarnings(action='ignore', category=DataConversionWarning)

    nb_items = (len(data) // _step_stride) + 1
    windows = np.arange(1, min(nb_items, _max_width + 1))
    predictions = pd.DataFrame(index=data.index)
    with click.progressbar(windows, fill_char='█', label='Predictions from CNNs', info_sep='\n') as bar:
        for width in bar:
            predictions[str(width)] = predict(data, width)
    return pd.DataFrame(index=predictions.index, data=median_filter(predictions.values, (1, 5)))


def plot_similarity_prediction(prediction, path_save, plot):
    """
    Plot the similarity prediction

    :param path_save: the path where to save the results
    :param plot: plot the results or not
    :param prediction: raw prediction made by the 100 CNNs
    :return:
    """
    fig, ax = plt.subplots()

    t_data = prediction.index
    similarities_tmp = prediction.values.T
    xx, yy = np.meshgrid(t_data, np.arange(0, len(prediction.columns)))
    im1 = ax.pcolormesh(xx, yy, similarities_tmp, vmin=0, vmax=1, cmap='jet')
    ax.set_ylabel('Window size (hr)')

    ax.text(1.0, 0.9, 'Prediction', transform=ax.transAxes, verticalalignment='top', rotation=-90)
    c_bar = fig.colorbar(im1, orientation='horizontal')
    c_bar.set_label(label='similarity')
    c_bar.ax.tick_params()
    plt.savefig(path_save)
    if plot:
        plt.show()


def plot_events_icme(integral, list_icmes, path_save, plot):
    """
    Plot the icme events

    :param integral: data
    :param list_icmes: list of icme events
    :param path_save: the path where to save the results
    :param plot: plot the results or not
    :return:
    """
    ax = integral.plot(color='green')
    plt.ylim(ymin=0)
    ip = 12
    for ev in list_icmes:
        rect = patches.Rectangle((ev.begin, 0), ev.duration(),
                                 ip, linewidth=1,
                                 edgecolor='green',
                                 facecolor='none')
        ax.add_patch(rect)

    plt.savefig(path_save)
    if plot:
        plt.show()
