import collections
from utils import load_datadic_from_tfrecords
import tensorflow as tf

# Getting data from file

## initial config
path = 'data/tf-records/'
DatasetInfo = collections.namedtuple(
    'DatasetInfo', ['basepath', 'size', 'sequence_length', 'coord_range'])

_DATASETS = dict(
    square_room=DatasetInfo(
        basepath='square_room_100steps_2.2m_1000000',
        size=100,
        sequence_length=100,
        coord_range=((-1.1, 1.1), (-1.1, 1.1))), )

ds_info = _DATASETS['square_room']

feature_map = {
  'init_pos':
      tf.io.FixedLenFeature(shape=[2], dtype=tf.float32),
  'init_hd':
      tf.io.FixedLenFeature(shape=[1], dtype=tf.float32),
  'ego_vel':
      tf.io.FixedLenFeature(
          shape=[ds_info.sequence_length, 3],
          dtype=tf.float32),
  'target_pos':
      tf.io.FixedLenFeature(
          shape=[ds_info.sequence_length, 2],
          dtype=tf.float32),
  'target_hd':
      tf.io.FixedLenFeature(
          shape=[ds_info.sequence_length, 1],
          dtype=tf.float32),
}

##loading
data_dic = load_datadic_from_tfrecords(path, _DATASETS, 'square_room', feature_map)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print('started training')
