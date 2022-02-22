import errno
from datetime import datetime, timedelta
from matplotlib import pyplot as plt, patches
import numpy as np
import pandas as pd

from utils.utils import delimit_data, resampling_data, \
    add_extra_features, plot_similarity_prediction, predict_all, get_module_logger, plot_events_icme
from utils.gautierfunctions.process import turn_peaks_to_clouds

import logging.config
import sys
import os

_delta = 0
_date_format = '%Y-%m-%dT%H:%M:%S'
_directory_data = 'data/datasets/'
_plot = True
_debug = False


def validate_time_format(date_string):
    """
    Converts string date to datetime object according to this format: '%Y-%m-%dT%H:%M:%S'

    :param date_string: given string date
    :return: datetime object
    """
    try:
        return datetime.strptime(date_string, _date_format)
    except ValueError:
        raise ValueError("This is the incorrect date string format. It should be: ", _date_format, " for this input: ",
                         date_string)


def main():
    """
    Gets a predicted ICME events for a given start and stop times,
    then saves them into csv file named 'icme_events.csv' and into
    image named 'icme_events.png'


    if you are in the debug mode (_debug == True) then you need to specify the start/stop times manually (see lines
    55, 56) if you are in the production mode (_debug == False) then you need to pass start/stop times as arguments (see
    readme.md)

    if you need to plot the results you need to assign True to this flag _plot

    :return:
    """
    logger.info("Checking arguments... ")
    nb_arguments = len(sys.argv)
    destination_folder_path = ""
    if not _debug and nb_arguments < 3:
        message = "X X : Expected at least 3 arguments. You give only: ", nb_arguments, *sys.argv
        logger.error(message)
        sys.exit(errno.EINVAL)
    elif not _debug and nb_arguments ==4:
        message = "Destination folder is given: ", sys.argv[3]
        logger.info(message)
        destination_folder_path = sys.argv[3]
    elif not _debug and nb_arguments ==3:
        message = "Destination folder is missing, the output files will be saved in the current folder"
        logger.warn(message)

    start = "2012-7-2T20:30:00"
    stop = "2012-7-20T23:59:00"

    if not _debug:
        message = "Checking the formats of start time and stop time: ", start, stop
        logger.info(message)
        try:
            start = validate_time_format(sys.argv[1])
            stop = validate_time_format(sys.argv[2])
        except ValueError as error:
            logger.error('Caught this error: ' + repr(error))
            sys.exit(errno.EINVAL)
    else:
        start = datetime.strptime(start, _date_format)
        stop = datetime.strptime(stop, _date_format)

    logger.info("Arguments are valid")

    if start >= stop:
        message = "X X : The start time must be less than the stop time: ", start, stop
        logger.error(message)
        sys.exit(errno.EINVAL)

    logger.info("Preparing data... ")
    raw_data = pd.read_parquet(_directory_data + 'datasetWithSpectro.parquet', engine='pyarrow')
    add_extra_features(raw_data)
    data = delimit_data(raw_data)
    data = resampling_data(data)

    prediction_start_time = start - timedelta(minutes=_delta)
    prediction_stop_time = stop + timedelta(minutes=_delta)

    data_prediction = data[data.index < prediction_stop_time]
    data_prediction = data_prediction[data_prediction.index > prediction_start_time]

    logger.info("Run predictions... ")
    prediction = predict_all(data_prediction)
    logger.info("Post processing... ")
    integral = prediction.sum(axis=1)
    list_icmes = turn_peaks_to_clouds(integral, 12)

    path_save = os.path.join(destination_folder_path, "similarity_output.png")
    plot_similarity_prediction(prediction, path_save, plot=_plot)

    path_save = os.path.join(destination_folder_path, "events_output.png")
    plot_events_icme(integral, list_icmes,  path_save, _plot)

    logger.info("Writing results... ")
    events_catalog = convert_events_to_np_array(list_icmes)
    path_save = os.path.join(destination_folder_path, "events_tt_output.csv")
    np.savetxt(path_save, events_catalog, delimiter=" ", fmt='%s')
    message = "List of events %s ", *events_catalog
    logger.info(message)
    logger.info("End")


def convert_events_to_np_array(list_events):
    """
    Converts ICME event to numpy array
    :param list_events: list of events
    :return: numpy array (catalog)
    """
    table = []
    for event in list_events:
        table.append([datetime.strftime(event.begin, _date_format), datetime.strftime(event.end, _date_format)])
    return np.array(table)


if __name__ == '__main__':
    logger = get_module_logger(__name__)
    # run predictions
    main()
    logging.shutdown()
