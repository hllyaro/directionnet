import os
import collections
import random
import tensorflow.compat.v1 as tf
import util
import argparse


# new
from absl import flags
from absl import app

import os
import sys
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

FLAGS = flags.FLAGS

flags.DEFINE_string('data_path', '', 'Directory where the data is stored')
flags.DEFINE_integer('epochs', 10, 'Number of training epochs')
flags.DEFINE_integer('batch_size', 2, 'Batch size')
flags.DEFINE_boolean('training', True, 'Mode')
flags.DEFINE_boolean('load_estimated_rot', False, 'TODO: investigate')

def data_loader(
        data_path,
        epochs,
        batch_size,
        training=True,
        load_estimated_rot=False):


    """Load stereo image datasets.

    Args:
      data_path: (string)
      epochs: (int) the number of training epochs.
      batch_size: (int) batch size.
      training: (bool) set it True when training to enable illumination randomization
        for input images.
      load_estimated_rot: (bool) set it True when training DirectionNet-T to load
        estimated rotation from DirectionNet-R saved as 'rotation_pred' on disk.

    Returns:
      Tensorflow Dataset
    """

    def load_data(path):
        "Load files saved as pickle."
        # print("=======================")
        # print(path)
        # print(type(path))

        # print("=======================")

        # sys.exit()
        img_id, rotation = tf.numpy_function(util.read_pickle,
                                      [path + '/rotation_gt.pickle'], [tf.string, tf.float32])
        _, translation = tf.numpy_function(util.read_pickle,
                                    [path + '/epipoles_gt.pickle'], [tf.string, tf.float32])
        _, fov = tf.numpy_function(util.read_pickle,
                            [path + '/fov.pickle'], [tf.string, tf.float32])

        if load_estimated_rot:
            _, rotation_pred = tf.numpy_function(util.read_pickle,
                                          [path + '/rotation_pred.pickle'], [tf.string, tf.float32])
        else:
            rotation_pred = tf.zeros_like(rotation)

        img_path = path + '/' + img_id
        return tf.data.Dataset.from_tensor_slices(
            (img_id, img_path, rotation, translation, fov, rotation_pred))

    def load_images(img_id, img_path, rotation, translation, fov, rotation_pred):
        """Load images and decode text lines."""

        def load_single_image(img_path):
            image = tf.image.decode_png(tf.read_file(img_path))
            image = tf.image.convert_image_dtype(image, tf.float32)
            image.set_shape([512, 512, 3])
            image = tf.squeeze(
                tf.image.resize_area(tf.expand_dims(image, 0), [256, 256]))
            return image

        input_pair = collections.namedtuple(
            'data_input',
            [
                'id',
                'src_image',
                'trt_image',
                'rotation',
                'translation',
                'fov',
                'rotation_pred'
            ])

        src_image = load_single_image(img_path + '.src.perspective.png')
        trt_image = load_single_image(img_path + '.trt.perspective.png')

        random_gamma = random.uniform(0.7, 1.2)
        if training:
            src_image = tf.image.adjust_gamma(src_image, random_gamma)
            trt_image = tf.image.adjust_gamma(trt_image, random_gamma)
            
            
        rotation = tf.reshape(rotation, [3, 3])
        rotation.set_shape([3, 3])

        # random_gamma = random.uniform(0.7, 1.2)
        # if training:
        # 	src_image = tf.image.adjust_gamma(src_image, random_gamma)
        # 	trt_image = tf.image.adjust_gamma(trt_image, random_gamma)

        # rotation = tf.reshape(rotation, [3, 3])
        # rotation.set_shape([3, 3])

        # translation = tf.reshape(
        #     tf.stack([tf.decode_csv(translation, [0.0] * 3)], 0), [3])

        
        translation = tf.reshape(translation, [3])
        translation.set_shape([3])



        # fov = tf.reshape(tf.stack([tf.decode_csv(fov, [0.0])], 0), [1])
        fov = tf.reshape(fov, [1])
        fov.set_shape([1])

        if load_estimated_rot:

            rotation_pred = tf.reshape(rotation_pred, [3, 3])
            # rotation_pred = tf.reshape(
            #     tf.stack([tf.decode_csv(rotation_pred, [0.0] * 9)], 0), [3, 3])
            rotation_pred.set_shape([3, 3])

        return input_pair(img_id, src_image, trt_image, rotation, translation, fov, rotation_pred)

    ds = tf.data.Dataset.list_files(os.path.join(data_path, '*'))
    # print("ds")
    # print(ds)
    ds = ds.flat_map(load_data)
    ds = ds.map(load_images, num_parallel_calls=50).apply(
        tf.data.experimental.ignore_errors()).repeat(epochs)
    ds = ds.batch(batch_size, drop_remainder=True).prefetch(10)

    return ds


def main(argv=None):

    dataloader = data_loader(
        FLAGS.data_path,
        FLAGS.epochs,
        FLAGS.batch_size,
        FLAGS.training,
        FLAGS.load_estimated_rot
    )

    for batch in dataloader.take(1):  # You can adjust how many batches to fetch
        img_id, src_image, trt_image, rotation, translation, fov, rotation_pred = batch
        print(f"Batch details:")
        print(f"Image ID: {img_id}")
        print(f"Source Image shape: {src_image.shape}")
        print(f"Target Image shape: {trt_image.shape}")
        print(f"Rotation: {rotation}")
        print(f"Translation: {translation}")
        print(f"FOV: {fov}")
        print(f"Rotation prediction: {rotation_pred}")


    print(f"Dataloader executed succesfully!")

if __name__ == '__main__':

    app.run(main)
