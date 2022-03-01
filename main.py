import errno
from datetime import timedelta

import numpy as np

from utils.utils import delimit_data, resampling_data, \
    add_extra_features, plot_similarity_prediction, predict_all, get_module_logger, plot_events_icme, \
    validate_time_format, get_or_download_data_set, convert_events_to_np_array
from utils.gautierfunctions.process import turn_peaks_to_clouds

import logging.config
import sys
import os

PLOT = False
"""
Plot results or not
"""

DEBUG = False
"""Run in debug mode without passing arguments to the module and using directly START_TIME_DEBUG and STOP_TIME_DEBUG 
defined just below """

START_TIME_DEBUG = "2012-7-2T20:30:00"
"""
The start time to use in order to debug, feel free to modify it
"""

STOP_TIME_DEBUG = "2012-7-20T23:59:00"
"""
The stop time to use in order to debug, feel free to modify it
"""

DELTA_TIME = 0
"""
Add a time to the start/stop times, by default is zero
"""


def main():
    """
    Returns the predicted ICME events for a given start and stop times,
    then saves them into csv file named 'icme_events.csv' and into
    image named 'icme_events.png'


    if you are in the debug mode (DEBUG == True) then you need to specify the start/stop times manually (see
    START_TIME_DEBUG and STOP_TIME_DEBUG), if you are in the production mode (DEBUG == False) then you need to pass
    start/stop times as arguments (see readme.md)

    if you need to plot the results you need to assign True to PLOT

    :return: the predicted ICME events
    """
    
    arguments = sys.argv
    nb_arguments = len(arguments)
    destination_folder_path = ""

    logger.info("Checking arguments... ")
    if not DEBUG:
        if nb_arguments < 4:
            message = "Expected at least 4 arguments. You give only: ", nb_arguments, *arguments
            logger.error(message)
            sys.exit(errno.EINVAL)
        else:
            message = "Given arguments: ", *arguments
            logger.info(message)
            destination_folder_path = arguments[1]
            if nb_arguments == 4:
                start = arguments[2]
                stop = arguments[3]
            else:
                start = arguments[3]
                stop = arguments[4]
            message = "Checking the formats of start time and stop time..."
            logger.info(message)
            try:
                start = validate_time_format(start)
                stop = validate_time_format(stop)
            except ValueError as error:
                logger.error('Caught this error: ' + repr(error))
                sys.exit(errno.EINVAL)
    else:
        start = validate_time_format(START_TIME_DEBUG)
        stop = validate_time_format(STOP_TIME_DEBUG)

    logger.info("Arguments are valid")

    if start >= stop:
        message = "The start time must be less than the stop time: ", start, stop
        logger.error(message)
        sys.exit(errno.EINVAL)

    logger.info("Preparing data... ")
    raw_data = get_or_download_data_set()

    add_extra_features(raw_data)
    data = delimit_data(raw_data)
    data = resampling_data(data)

    prediction_start_time = start - timedelta(minutes=DELTA_TIME)
    prediction_stop_time = stop + timedelta(minutes=DELTA_TIME)

    data_prediction = data[data.index < prediction_stop_time]
    data_prediction = data_prediction[data_prediction.index > prediction_start_time]

    logger.info("Run predictions... ")
    prediction = predict_all(data_prediction)
    logger.info("Post processing... ")
    integral = prediction.sum(axis=1)
    list_icmes = turn_peaks_to_clouds(integral, 12)

    path_save = os.path.join(destination_folder_path, "similarity_output.png")
    plot_similarity_prediction(prediction, path_save, plot=PLOT)

    path_save = os.path.join(destination_folder_path, "events_output.png")
    plot_events_icme(integral, list_icmes, path_save, PLOT)

    logger.info("Writing results... ")
    events_catalog = convert_events_to_np_array(list_icmes)
    path_save = os.path.join(destination_folder_path, "events_tt_output.csv")
    np.savetxt(path_save, events_catalog, delimiter=" ", fmt='%s')
    message = "List of events: ", *events_catalog
    logger.info(message)
    logger.info("End")


if __name__ == '__main__':
    logger = get_module_logger(__name__)
    # run predictions
    main()
    logging.shutdown()
