# ================================
# Wrapper for common utils in DNN
# ================================

import tensorflow as tf
import numpy as np
from tensorflow.keras.initializers import GlorotUniform,Zeros
from tensorflow.python.ops import init_ops

## modified
xavier_initializer = tf.keras.initializers.GlorotUniform()


# =========================================
# =========================================
# Common operation in network
# =========================================
# =========================================
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Layer, Conv2D, Dense
from tensorflow.keras.initializers import GlorotUniform, Zeros, VarianceScaling

def activate(input, name, act_type='relu'):
  """
  Activation function using Keras layers
  """
  with tf.name_scope(name):
    if act_type == 'relu':
      return tf.keras.layers.ReLU()(input)  # Use Keras ReLU layer
    elif act_type == 'sigmoid':
      return tf.keras.layers.Activation('sigmoid')(input)  # Use Keras Activation layer
    else:
      raise ValueError('act_type is not valid.')

def conv2D(input,
           shape,
           name,
           padding='SAME',
           strides=(1, 1),
           weights_initializer=xavier_initializer,
           bias_initializer=None,
           weights_regularizer=None,
           bias_regularizer=None,
           params={}):
    """
    Convolution layer using Keras Conv2D
    """
    # Use default initializers if not provided
    weights_initializer = weights_initializer or GlorotUniform()
    bias_initializer = bias_initializer or Zeros()
    
    # Create Conv2D layer
    conv_layer = Conv2D(
        filters=shape[2],
        kernel_size=(shape[0], shape[1]),
        strides=strides,
        padding=padding,
        kernel_initializer=weights_initializer,
        bias_initializer=bias_initializer,
        kernel_regularizer=weights_regularizer,
        bias_regularizer=bias_regularizer,
        name=name
    )
    
    # Apply convolution
    out = conv_layer(input)
    
    # Store parameters if params dict is provided
    if params is not None:
        params[name] = [conv_layer.kernel, conv_layer.bias]
    
    return out

def max_pool(input, name, ksize=(2, 2), strides=(2, 2), padding='SAME'):
    """
    Max pooling using tf.nn.max_pool2d
    """
    out = tf.nn.max_pool2d(
        input, 
        ksize=[1, ksize[0], ksize[1], 1], 
        strides=[1, strides[0], strides[1], 1], 
        padding=padding
    )
    return out


# =========================================
def avg_pool(input,
             name,
             ksize=(2, 2),
             strides=(2, 2),
             padding='SAME'
             ):
    with tf.variable_scope(name) as scope:
        ksize = [1, ksize[0], ksize[1], 1]
        strides = [1, strides[0], strides[1], 1]
        out = tf.nn.avg_pool(input, ksize=ksize, strides=strides, padding=padding)
        print('{} avg pool out: {}'.format(name, out))
    return out


# =========================================
def fully_connected(input,
                    num_neuron,
                    name,
                    weights_initializer=xavier_initializer,
                    bias_initializer=init_ops.zeros_initializer(),
                    weights_regularizer=None,
                    bias_regularizer=None,
                    params={}
                    ):
    use_bias = use_bias_helper(bias_initializer)
    # with tf.variable_scope(name) as scope:
    #     input_dim = int(np.prod(input.get_shape().as_list()[1:]))
    #     kernel = tf.get_variable(
    #         name='weights',
    #         shape=[input_dim, num_neuron],
    #         dtype=tf.float32,
    #         initializer=weights_initializer,
    #         regularizer=weights_regularizer
    #     )
    #     flat = tf.reshape(input, [-1, input_dim])
    #     out = tf.matmul(flat, kernel)
    #     bias = None
    #     if use_bias:
    #         bias = tf.get_variable(
    #             name='biases',
    #             shape=num_neuron,
    #             dtype=tf.float32,
    #             initializer=bias_initializer,
    #             regularizer=bias_regularizer
    #         )
    #         out = tf.nn.bias_add(out, bias)

    #     print('{} weights: {}, bias: {}, out: {}'.format(name, kernel, bias, out))
    #     params[name] = [kernel, bias]
    # Using tf.name_scope instead of tf.variable_scope
    with tf.name_scope(name):
        input_dim = int(tf.reduce_prod(input.shape[1:]))  # Adjusted for TensorFlow 2.x
        # Initialize weights
        kernel = tf.Variable(
            initial_value=weights_initializer(shape=(input_dim, num_neuron), dtype=tf.float32),
            trainable=True,
            name=f"{name}_weights"
        )
        
        # Flatten input if necessary
        flat = tf.reshape(input, [-1, input_dim])
        # Matrix multiplication
        out = tf.matmul(flat, kernel)

        bias = None
        if use_bias:
            # Initialize bias
            bias = tf.Variable(
                initial_value=bias_initializer(shape=(num_neuron,), dtype=tf.float32),
                trainable=True,
                name=f"{name}_biases"
            )
            # Add bias
            out = tf.nn.bias_add(out, bias)

        # Logging
        print('{} weights: {}, bias: {}, out: {}'.format(name, kernel.shape, bias.shape if bias is not None else None, out.shape))

        # Store parameters
        params[name] = [kernel, bias]


    return out


# =========================================
def batch_norm(input, name, is_train=True, params={}):
    batch_norm_out = tf.contrib.layers.batch_norm(inputs=input, scale=True, is_training=is_train, scope=name)
    # Get gamma and beta
    # trainable_vars = tf.trainable_variables()
    # gamma = [var for var in trainable_vars if name in var.name and 'gamma' in var.name]
    # beta = [var for var in trainable_vars if name in var.name and 'beta' in var.name]
    # for i in tf.get_default_graph().get_operations():
    #     print i.name
    var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)  # beta, gamma, moving_mean, moving_variance
    params[name] = var_list
    print('{} {}'.format(name, batch_norm_out))
    return batch_norm_out

# =========================================
def use_bias_helper(bias_initializer):
    """
    Determine if a layer needs bias
    :param bias_initializer:
    :return:
    """
    if bias_initializer is None:
        return False
    else:
        return True

# ============================================
def get_restore_var_list(path):
    """
    Get variable list when restore from ckpt. This is mainly for transferring model to another network
    """
    global_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)  # Variables in graph
    saved_vars = list_vars_in_ckpt(path)
    saved_vars_name = [var[0] for var in saved_vars]
    restore_var_list = [var for var in global_vars if var.name[:-2] in saved_vars_name]# or 'vgg_' + var.name[:-2] in saved_vars_name]

    return restore_var_list


# ============================================
def list_vars_in_ckpt(path):
    """List all variables in checkpoint"""
    saved_vars = tf.contrib.framework.list_variables(path)  # List of tuples (name, shape)
    return saved_vars