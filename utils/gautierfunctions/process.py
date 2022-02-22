# All these functions are developed by Gautier Nguyen, please refers to:
# https://github.com/gautiernguyen/Automatic-detection-of-ICMEs-at-1-AU-a-deep-learning-approach

import datetime

import numpy as np
from joblib import Parallel, delayed

import pandas as pds

from .evt import Event

import numpy.random as random
from scipy.signal import find_peaks, peak_widths
from lmfit import models


def make_event_list(y, label, delta=2):
    """
    Consider y as a pandas series, returns a list of Events corresponding to
    the requested label (int), works for both smoothed and expected series
    Delta corresponds to the series frequency (in our basic case with random
    index, we consider this value to be equal to 2)
    """
    list_of_pos_label = y[y == label]

    if len(list_of_pos_label) == 0:
        return []
    delta_between_pos_label = list_of_pos_label.index[1:] - list_of_pos_label.index[:-1]
    delta_between_pos_label.insert(0, datetime.timedelta(0))
    end_of_events = np.where(delta_between_pos_label > datetime.timedelta(minutes=delta))[0]
    index_begin = 0
    event_list = []
    for i in end_of_events:
        end = i
        event_list.append(Event(list_of_pos_label.index[index_begin], list_of_pos_label.index[end]))
        index_begin = i + 1
    event_list.append(Event(list_of_pos_label.index[index_begin], list_of_pos_label.index[-1]))
    return event_list


def turn_peaks_to_clouds(series, threshold, freq=10,
                         duration_creepies=2.5, n_jobs=2):
    """
    Transforms the output series of a pipeline into a complete list of events
    """
    events = []
    prediction = pds.Series(index=pds.date_range(series.index[0],
                                                 series.index[-1],
                                                 freq=(str(freq) + 'T')),
                            data=np.nan)

    prediction[series.index[series > threshold]] = 1
    prediction[series.index[series < threshold]] = 0

    prediction = prediction.interpolate()

    intervals = make_event_list(prediction, 1, freq)
    intervals = remove_creepy(intervals, duration_creepies)

    results = Parallel(n_jobs=n_jobs)(delayed(_turn_intervals_to_event)(event, series) for event in intervals)

    for fls in results:
        events.extend(fls)
    return events


def remove_creepy(event_list, threshold=2):
    """
    For a given list, remove the element whose duration is under the threshold
    """
    return [x for x in event_list if x.duration() > datetime.timedelta(hours=threshold)]


def _generate_model(spec):
    composite_model = None
    params = None
    x = spec['time']
    y = spec['y']
    x_min = np.min(x)
    x_max = np.max(x)
    x_range = x_max - x_min
    y_max = np.max(y)
    for i, basis_func in enumerate(spec['model']):
        prefix = f'm{i}_'
        model = getattr(models, basis_func['type'])(prefix=prefix)
        if basis_func['type'] in ['GaussianModel', 'LorentzianModel',
                                  'VoigtModel']:  # for now VoigtModel has gamma constrained to sigma
            model.set_param_hint('sigma', min=1e-6, max=x_range)
            model.set_param_hint('center', min=x_min, max=x_max)
            model.set_param_hint('height', min=1e-6, max=1.1 * y_max)
            model.set_param_hint('amplitude', min=1e-6)
            default_params = {
                prefix + 'center': x_min + x_range * random.random(),
                prefix + 'height': y_max * random.random(),
                prefix + 'sigma': x_range * random.random()
            }
        else:
            raise NotImplemented(f'model {basis_func["type"]} not implemented yet')
        if 'help' in basis_func:  # allow override of settings in parameter
            for param, options in basis_func['help'].items():
                model.set_param_hint(param, **options)
        model_params = model.make_params(**default_params, **basis_func.get('params', {}))
        if params is None:
            params = model_params
        else:
            params.update(model_params)
        if composite_model is None:
            composite_model = model
        else:
            composite_model = composite_model + model
    return composite_model, params


def _turn_intervals_to_event(event, series):
    """
    Find events in a temporal interval that contain one or several events
    """
    spec = {
        'time': np.arange(0, len(series[event.begin:event.end])),
        'y': series[event.begin:event.end].values,
        'model': [
            {'type': 'GaussianModel'},
            {'type': 'GaussianModel'},
            {'type': 'GaussianModel'},
            {'type': 'GaussianModel'},
            {'type': 'GaussianModel'}
        ]
    }

    model, params = _generate_model(spec)
    output = model.fit(spec['y'], params, x=spec['time'])
    fitted_integral = output.best_fit
    pos = find_peaks(fitted_integral)[0]
    width = peak_widths(fitted_integral, pos)

    ref_index = series[event.begin:event.end].index
    clouds = [Event(ref_index[int(width[2][x])], ref_index[int(width[3][x])]) for x in
              np.arange(0, len(width[0]))]
    return clouds
