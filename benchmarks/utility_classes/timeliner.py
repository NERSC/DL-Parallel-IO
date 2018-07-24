# Code taken from: https://github.com/ikhlestov/tensorflow_profiling/blob/master/03_merged_timeline_example.py
# This code is for merging the json output from TensorFlow's timeline module for multiple session runs

import json


class TimeLiner:

    def __init__(self):
        # Initialize members
        self._timeline_dict = None

    def update_timeline(self, chrome_trace):
        # convert chrome trace to python dict
        chrome_trace_dict = json.loads(chrome_trace)
        # for first run store full trace
        if self._timeline_dict is None:
            self._timeline_dict = chrome_trace_dict
        # for other - update only time consumption, not definitions
        else:
            for event in chrome_trace_dict['traceEvents']:
                # events time consumption started with 'ts' prefix
                if 'ts' in event:
                    self._timeline_dict['traceEvents'].append(event)

    def save(self, f_name):
        with open(f_name, 'w') as f:
            json.dump(self._timeline_dict, f)
