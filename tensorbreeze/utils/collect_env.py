import torch
import torchvision
import tensorflow
from tensorflow.python.framework import test_util as tf_test_util


tf_info_fmt = """
TensorFlow version: {tf_version}
Is Google CUDA enabled: {is_google_cuda_enabled}
Is MKL enabled: {is_mkl_enabled}
""".strip()


def get_pretty_tf_info():
    tf_version = tensorflow.__version__
    is_google_cuda_enabled = tf_test_util.IsGoogleCudaEnabled()
    is_mkl_enabled = tf_test_util.IsMklEnabled()

    tf_info_str = tf_info_fmt.format(
        tf_version=tf_version,
        is_google_cuda_enabled=is_google_cuda_enabled,
        is_mkl_enabled=is_mkl_enabled
    )

    return tf_info_str


def get_pretty_torchvision_info():
    torchvision_version = torchvision.__version__
    return 'Torchvision version: {}'.format(torchvision_version)


def get_pretty_env_info():
    tf_info_str = get_pretty_tf_info()
    torchvision_info_str = get_pretty_torchvision_info()
    env_str = torch.utils.collect_env.get_pretty_env_info()
    return '\n'.join([tf_info_str, torchvision_info_str, env_str])
