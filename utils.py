import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from os import listdir
from os.path import isfile, join


def load_datadic_from_tfrecords(path, _Datasets, env_name: str, feature_map: dict):
    ds_info = _Datasets[env_name]

    tfrecord_files = [join(path, f) for f in listdir(path) if isfile(join(path, f))]
    raw_dataset = tf.data.TFRecordDataset(tfrecord_files)

    def _parse_function(example_proto):
        # Parse the input `tf.train.Example` proto using the dictionary above.
        return tf.io.parse_single_example(example_proto, feature_map)

    parsed_dataset = raw_dataset.map(_parse_function)

    datadic = dict()
    for key in feature_map.keys():
        datadic[key] = list()

    for parsed_record in parsed_dataset:
        datadic['init_pos'].append(parsed_record['init_pos'].numpy())
        datadic['init_hd'].append(parsed_record['init_hd'].numpy())
        datadic['ego_vel'].append(parsed_record['ego_vel'].numpy())
        datadic['target_pos'].append(parsed_record['target_pos'].numpy())
        datadic['target_hd'].append(parsed_record['target_hd'].numpy())

