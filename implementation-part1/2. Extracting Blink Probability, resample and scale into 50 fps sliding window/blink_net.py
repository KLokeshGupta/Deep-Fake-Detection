import tensorflow as tf
from tensorflow.keras.layers import Layer
import numpy as np
import yaml
import os
from easydict import EasyDict as edict
from deep_base import ops as net_ops
from deep_base import vgg16 as base

pwd = os.path.dirname(__file__)

class ReshapeLayer(Layer):
    def __init__(self, target_shape, **kwargs):
        super(ReshapeLayer, self).__init__(**kwargs)
        self.target_shape = target_shape

    def call(self, inputs):
        return tf.reshape(inputs, self.target_shape)

class BlinkCNN(tf.keras.Model):
    """
    CNN for eye blinking detection
    """
    def __init__(self, is_train):
        super(BlinkCNN, self).__init__()

        cfg_file = os.path.join(pwd, 'blink_lrcn.yml')
        with open(cfg_file, 'r') as f:
            cfg = edict(yaml.load(f, Loader=yaml.FullLoader))

        self.cfg = cfg
        self.img_size = cfg.IMG_SIZE
        self.num_classes = cfg.NUM_CLASS
        self.is_train = is_train

        self.layers = {}
        self.params = {}

    def build(self):
        # Input
        self.input = tf.keras.Input(shape=(self.img_size[0], self.img_size[1], self.img_size[2]), dtype=tf.float32)
        self.layers = base.get_prob(self.input, self.params, self.num_classes, self.is_train)
        self.prob = self.layers.prob
        self.gt = tf.keras.Input(shape=(None,), dtype=tf.int32)
        self.var_list = self.trainable_variables

    def call(self, inputs):
        # Forward pass (model output)
        return self.prob

    def loss(self):
        self.net_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.gt, logits=self.layers.fc8)
        self.net_loss = tf.reduce_mean(self.net_loss)
        tf.losses.add_loss(self.net_loss)

        # L2 weight regularize
        self.L2_loss = tf.reduce_mean([self.cfg.TRAIN.BETA * tf.nn.l2_loss(v)
                                       for v in self.trainable_variables if 'weights' in v.name])
        tf.losses.add_loss(self.L2_loss)
        self.total_loss = tf.losses.get_total_loss()

class BlinkLRCN(object):
    """
    LRCN for eye blinking detection
    """

    def __init__(self, is_train):
        # Load config and set up parameters
        cfg_file = os.path.join(pwd, 'blink_lrcn.yml')
        with open(cfg_file, 'r') as f:
            cfg = edict(yaml.load(f, Loader=yaml.FullLoader))

        self.cfg = cfg
        self.img_size = cfg.IMG_SIZE
        self.num_classes = cfg.NUM_CLASS
        self.is_train = is_train

        self.rnn_type = cfg.RNN_TYPE
        self.max_time = cfg.MAX_TIME
        self.hidden_unit = cfg.HIDDEN_UNIT

        if self.is_train:
            self.batch_size = cfg.TRAIN.BATCH_SIZE
        else:
            self.batch_size = cfg.TEST.BATCH_SIZE
        self._layers = {}  # Renamed from `layers`
        self._params = {}  # Renamed from `params`

    def build(self):
        # Use tf.keras.Input for input placeholders in TensorFlow 2.x
        self.input = tf.keras.Input(shape=(self.max_time, self.img_size[0], self.img_size[1], self.img_size[2]))
        self.blined_gt = tf.keras.Input(shape=(self.batch_size,), dtype=tf.int32)
        self.eye_state_gt = tf.keras.Input(shape=(self.batch_size, self.max_time), dtype=tf.int32)
        self.seq_len = tf.keras.Input(shape=(self.batch_size,), dtype=tf.int32)

        # Build the network
        self.vgg16_fc6 = self._vgg16(self.input)
        print(self.vgg16_fc6)
        self.rnn_out = self._rnn_cell(self.vgg16_fc6)
        self.out = self._fc(self.rnn_out)
        self.prob = tf.nn.softmax(self.out, axis=-1)

    def _vgg16(self, input):
        # Reshape from NxTxHxWxC to (NxT)xHxWxC
        reshape_layer = ReshapeLayer(target_shape=[-1, self.img_size[0], self.img_size[1], self.img_size[2]])
        input_reshaped = reshape_layer(input)
        
        # Call the VGG16 base model for feature extraction
        layers = base.get_vgg16_pool5(input_reshaped, self._params)  # Ensure base.get_vgg16_pool5 works with reshaped input
        
        # layers.fc6 = net_ops.fully_connected(input=layers.pool5, num_neuron=4096, name='fc6', params=self._params)
        fc6=net_ops.fully_connected(input=layers,num_neuron=4096,name='fc6',params=self._params)

        if self.is_train:
            # layers.fc6 = tf.nn.dropout(layers.fc6, keep_prob=0.5)
            fc6=tf.nn.dropout(fc6,keep_prob=0.5)
            
        # layers.fc6_relu = net_ops.activate(input=layers.fc6, act_type='relu', name='fc6_relu')
        fc6_relu = net_ops.activate(input=fc6, act_type='relu', name='fc6_relu')
        # Reshape to match the expected dimensions for the RNN layer
        # cnn_out = tf.reshape(layers.fc6_relu, [-1, self.max_time, 4096])
        cnn_out = tf.reshape(fc6_relu, [-1, self.max_time, 4096])
        print(cnn_out)
        return cnn_out

    def _rnn_cell(self, input):
        with tf.variable_scope('rnn_cell'):
            size = np.prod(input.get_shape().as_list()[2:])
            rnn_inputs = tf.reshape(input, (-1, self.max_time, size))
            if self.rnn_type == 'LSTM':
                cell = tf.keras.layers.LSTMCell(self.hidden_unit)
            elif self.rnn_type == 'GRU':
                cell = tf.keras.layers.GRUCell(self.hidden_unit)
            else:
                raise ValueError('We only support LSTM or GRU...')
            rnn_outputs, _ = tf.keras.layers.RNN(cell)(rnn_inputs, sequence_length=self.seq_len)
            return rnn_outputs

    def _fc(self, input):
        # Reshape from NxTx256 to (NxT)x256
        input = tf.reshape(input, [-1, self.hidden_unit])
        out = net_ops.fully_connected(input=input, num_neuron=self.num_classes, name='fc_after_rnn', params=self._params)
        out = tf.reshape(out, [-1, self.max_time, self.num_classes])
        return out

    def loss(self):
        self.net_loss = []
        for batch_id in range(self.batch_size):
            out_cur = self.out[batch_id, :, :]
            eye_state_cur = self.eye_state_gt[batch_id, :]
            weights = tf.gather(tf.constant(self.cfg.TRAIN.CLASS_WEIGHTS, dtype=tf.float32), eye_state_cur)
            loss_per_batch = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=eye_state_cur, logits=out_cur)  # T x num_class
            loss_per_batch = loss_per_batch * weights
            # Select loss by real len
            seq_len = tf.cast(self.seq_len[batch_id], dtype=tf.float32)
            tf_idx = tf.range(0, self.seq_len[batch_id])
            loss_per_batch = tf.reduce_sum(tf.gather(loss_per_batch, tf_idx, axis=0)) / seq_len
            self.net_loss.append(loss_per_batch)
        self.net_loss = tf.reduce_mean(self.net_loss)
        tf.losses.add_loss(self.net_loss)
        # L2 weight regularize
        self.L2_loss = tf.reduce_mean([self.cfg.TRAIN.BETA * tf.nn.l2_loss(v)
                                       for v in tf.trainable_variables() if 'weights' in v.name or 'kernel' in v.name])
        tf.losses.add_loss(self.L2_loss)
        self.total_loss = tf.losses.get_total_loss()
