import os
import sys

# Append TensorBreeze root directory to sys
this_dir = os.path.dirname(os.path.realpath(__file__))
root_dir = os.path.abspath(os.path.join(this_dir, os.pardir))
sys.path.append(root_dir)

# import pytest
import numpy as np
import torch
import tensorflow as tf

from torchvision import models as torch_models
from tensorbreeze.utils.context import Session
from tensorbreeze.resnet import add_resnet_ops, load_pretrained_weights


def test_build():
    x = tf.placeholder('float32', shape=(2, 3, 224, 224))
    y, resnet_config = add_resnet_ops(x)

    x = tf.placeholder('float32', shape=(2, 3, 224, 96))
    y, resnet_config = add_resnet_ops(x, OUTPUT_ACTIVATION=None)


def test_output():
    dummy_input = np.random.rand(2, 3, 224, 224).astype('float32')

    with Session(allow_growth=True) as sess:
        x = tf.placeholder('float32', shape=(2, 3, 224, 224))
        y, resnet_config = add_resnet_ops(x)
        load_pretrained_weights('resnet50')
        tb_output = sess.run(y, {x: dummy_input})

    torch_resnet50 = torch_models.resnet50(pretrained=True).eval().cuda()
    torch_output = torch_resnet50(torch.Tensor(dummy_input).cuda())
    torch_output = torch.nn.functional.softmax(torch_output, dim=1)
    torch_output = torch_output.data.cpu().numpy()

    output_diff = np.mean(np.abs(tb_output - torch_output))
    assert output_diff < 1e-7, 'Difference between torch and tb output too large!'
