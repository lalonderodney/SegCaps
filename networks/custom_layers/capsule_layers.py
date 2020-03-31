'''
Capsules for Object Segmentation (SegCaps)
Original Paper: https://arxiv.org/abs/1804.04241
Code written by: Rodney LaLonde
If you use significant portions of this code or the ideas from our paper, please cite it :)
If you have any questions, please email me at lalonde@knights.ucf.edu.

This file contains the definitions of the various capsule layers and dynamic routing and squashing functions.
'''

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import initializers, layers
from keras.utils.conv_utils import conv_output_length, deconv_length
import numpy as np

class ExpandDim(layers.Layer):
    def call(self, inputs, **kwargs):
        return K.expand_dims(inputs, axis=-2)

    def compute_output_shape(self, input_shape):
        return (input_shape[0:-1] + (1,) + input_shape[-1:])

    def get_config(self):
        config = {}
        base_config = super(ExpandDim, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))

class RemoveDim(layers.Layer):
    def call(self, inputs, **kwargs):
        return K.squeeze(inputs, axis=-2)

    def compute_output_shape(self, input_shape):
        return (input_shape[0:-2] + input_shape[-1:])

    def get_config(self):
        config = {}
        base_config = super(RemoveDim, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Length(layers.Layer):
    def __init__(self, num_classes, seg=True, **kwargs):
        super(Length, self).__init__(**kwargs)
        if num_classes == 2:
            self.num_classes = 1
        else:
            self.num_classes = num_classes
        self.seg = seg

    def call(self, inputs, **kwargs):
        if inputs.get_shape().ndims == 5:
            inputs = K.squeeze(inputs, axis=-2)
        return K.expand_dims(tf.norm(inputs, axis=-1), axis=-1)

    def compute_output_shape(self, input_shape):
        if len(input_shape) == 5:
            input_shape = input_shape[0:-2] + input_shape[-1:]
        if self.seg:
            return input_shape[:-1] + (self.num_classes,)
        else:
            return input_shape[:-1]

    def get_config(self):
        config = {'num_classes': self.num_classes, 'seg': self.seg}
        base_config = super(Length, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Mask(layers.Layer):
    def __init__(self, resize_masks=False, **kwargs):
        super(Mask, self).__init__(**kwargs)
        self.resize_masks = resize_masks

    def call(self, inputs, **kwargs):
        if type(inputs) is list:
            assert len(inputs) == 2
            input, mask = inputs
            _, hei, wid, _, _ = input.get_shape()
            if self.resize_masks:
                mask = tf.image.resize_bicubic(mask, (hei.value, wid.value))
            mask = K.expand_dims(mask, -1)
            if input.get_shape().ndims == 3:
                masked = K.batch_flatten(mask * input)
            else:
                masked = mask * input

        else:
            if inputs.get_shape().ndims == 3:
                x = K.sqrt(K.sum(K.square(inputs), -1))
                mask = K.one_hot(indices=K.argmax(x, 1), num_classes=x.get_shape().as_list()[1])
                masked = K.batch_flatten(K.expand_dims(mask, -1) * inputs)
            else:
                masked = inputs

        return masked

    def compute_output_shape(self, input_shape):
        if type(input_shape[0]) is tuple:  # true label provided
            if len(input_shape[0]) == 3:
                return tuple([None, input_shape[0][1] * input_shape[0][2]])
            else:
                return input_shape[0]
        else:  # no true label provided
            if len(input_shape) == 3:
                return tuple([None, input_shape[1] * input_shape[2]])
            else:
                return input_shape

    def get_config(self):
        config = {'resize_masks': self.resize_masks}
        base_config = super(Mask, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class ConvCapsuleLayer(layers.Layer):
    def __init__(self, kernel_size, num_capsule, num_atoms, strides=1, padding='same', routings=3,
                 kernel_initializer='he_normal', **kwargs):
        super(ConvCapsuleLayer, self).__init__(**kwargs)
        self.kernel_size = kernel_size
        self.num_capsule = num_capsule
        self.num_atoms = num_atoms
        self.strides = strides
        self.padding = padding
        self.routings = routings
        self.kernel_initializer = initializers.get(kernel_initializer)

    def build(self, input_shape):
        assert len(input_shape) == 5, "The input Tensor should have shape=[None, input_height, input_width," \
                                      " input_num_capsule, input_num_atoms]"
        self.input_height = input_shape[1]
        self.input_width = input_shape[2]
        self.input_num_capsule = input_shape[3]
        self.input_num_atoms = input_shape[4]

        # Transform matrix
        self.W = self.add_weight(shape=[self.kernel_size, self.kernel_size,
                                 self.input_num_atoms, self.num_capsule * self.num_atoms],
                                 initializer=self.kernel_initializer,
                                 name='W')

        self.b = self.add_weight(shape=[1, 1, self.num_capsule, self.num_atoms],
                                 initializer=initializers.constant(0.1),
                                 name='b')

        self.built = True

    def call(self, input_tensor, training=None):

        input_shape = K.shape(input_tensor)

        input_transposed = tf.transpose(input_tensor, [0, 3, 1, 2, 4])
        input_tensor_reshaped = K.reshape(input_transposed, [
            input_shape[0] * input_shape[3], input_shape[1], input_shape[2], self.input_num_atoms])
        input_tensor_reshaped.set_shape((None, self.input_height, self.input_width, self.input_num_atoms))

        conv = K.conv2d(input_tensor_reshaped, self.W, (self.strides, self.strides),
                        padding=self.padding, data_format='channels_last')

        votes_shape = K.shape(conv)
        # Reshape back to 6D by splitting first dimmension to batch and input_dim
        # and splitting last dimmension to output_dim and output_atoms.

        votes = K.reshape(conv, [input_shape[0], input_shape[3], votes_shape[1], votes_shape[2],
                                 self.num_capsule, self.num_atoms])
        vote_height = conv_output_length(self.input_height, self.kernel_size, padding=self.padding,
                                         stride=self.strides, dilation=1)
        vote_width = conv_output_length(self.input_width, self.kernel_size, padding=self.padding,
                                         stride=self.strides, dilation=1)
        votes.set_shape((None, self.input_num_capsule, vote_height, vote_width, self.num_capsule, self.num_atoms))

        logit_shape = K.stack([
            input_shape[0], input_shape[3], votes_shape[1], votes_shape[2], self.num_capsule])
        biases_replicated = K.tile(self.b, [votes_shape[1], votes_shape[2], 1, 1])

        activations = _update_routing(
            votes=votes,
            biases=biases_replicated,
            logit_shape=logit_shape,
            num_dims=6,
            input_dim=self.input_num_capsule,
            output_dim=self.num_capsule,
            num_routing=self.routings)

        return activations

    def compute_output_shape(self, input_shape):
        space = input_shape[1:-2]
        new_space = []
        for i in range(len(space)):
            new_dim = conv_output_length(
                space[i],
                self.kernel_size,
                padding=self.padding,
                stride=self.strides,
                dilation=1)
            new_space.append(new_dim)

        return (input_shape[0],) + tuple(new_space) + (self.num_capsule, self.num_atoms)

    def get_config(self):
        config = {
            'kernel_size': self.kernel_size,
            'num_capsule': self.num_capsule,
            'num_atoms': self.num_atoms,
            'strides': self.strides,
            'padding': self.padding,
            'routings': self.routings,
            'kernel_initializer': initializers.serialize(self.kernel_initializer)
        }
        base_config = super(ConvCapsuleLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class DeconvCapsuleLayer(layers.Layer):
    # TODO: Change this description
    """
    The capsule layer. It is similar to Dense layer. Dense layer has `in_num` inputs, each is a scalar, the output of the
    neuron from the former layer, and it has `out_num` output neurons. CapsuleLayer just expand the output of the neuron
    from scalar to vector. So its input shape = [None, input_num_capsule, input_num_atoms] and output shape = \
    [None, num_capsule, num_atoms]. For Dense Layer, input_num_atoms = num_atoms = 1.

    :param num_capsule: number of capsules in this layer
    :param num_atoms: dimension of the output vectors of the capsules in this layer
    :param routings: number of iterations for the routing algorithm
    """

    def __init__(self, kernel_size, num_capsule, num_atoms, scaling=2, upsamp_type='deconv', padding='same', routings=3,
                 leaky_routing=False, kernel_initializer='he_normal', **kwargs):
        super(DeconvCapsuleLayer, self).__init__(**kwargs)
        self.kernel_size = kernel_size
        self.num_capsule = num_capsule
        self.num_atoms = num_atoms
        self.scaling = scaling
        self.upsamp_type = upsamp_type
        self.padding = padding
        self.routings = routings
        self.leaky_routing = leaky_routing
        self.kernel_initializer = initializers.get(kernel_initializer)

    def build(self, input_shape):
        assert len(input_shape) == 5, "The input Tensor should have shape=[None, input_height, input_width," \
                                      " input_num_capsule, input_num_atoms]"
        self.input_height = input_shape[1]
        self.input_width = input_shape[2]
        self.input_num_capsule = input_shape[3]
        self.input_num_atoms = input_shape[4]

        # Transform matrix
        if self.upsamp_type == 'subpix':
            self.W = self.add_weight(shape=[self.kernel_size, self.kernel_size,
                                            self.input_num_atoms,
                                            self.num_capsule * self.num_atoms * self.scaling * self.scaling],
                                     initializer=self.kernel_initializer,
                                     name='W')
        elif self.upsamp_type == 'resize':
            self.W = self.add_weight(shape=[self.kernel_size, self.kernel_size,
                                     self.input_num_atoms, self.num_capsule * self.num_atoms],
                                     initializer=self.kernel_initializer, name='W')
        elif self.upsamp_type == 'deconv':
            self.W = self.add_weight(shape=[self.kernel_size, self.kernel_size,
                                            self.num_capsule * self.num_atoms, self.input_num_atoms],
                                     initializer=self.kernel_initializer, name='W')
        else:
            raise NotImplementedError('Upsampling must be one of: "deconv", "resize", or "subpix"')

        self.b = self.add_weight(shape=[1, 1, self.num_capsule, self.num_atoms],
                                 initializer=initializers.constant(0.1),
                                 name='b')

        self.built = True

    def call(self, input_tensor, training=None):
        """Builds a slim upsampling capsule layer.

        This layer performs 2D convolution given 5D input tensor of shape
        `[batch, input_dim, input_atoms, input_height, input_width]`. Then refines
        the votes with routing and applies Squash non linearity for each capsule.

        Each capsule in this layer is a convolutional unit and shares its kernel over
        the position grid and different capsules of layer below. Therefore, number
        of trainable variables in this layer is:

          kernel: [kernel_size, kernel_size, input_atoms, output_dim * output_atoms]
          bias: [output_dim, output_atoms]

        Output of a conv2d layer is a single capsule with channel number of atoms.
        Therefore conv_slim_capsule is suitable to be added on top of a conv2d layer
        with num_routing=1, input_dim=1 and input_atoms=conv_channels.

        Args:
          input_tensor: tensor, of rank 5. Last two dimmensions representing height
            and width position grid.
          input_dim: scalar, number of capsules in the layer below.
          output_dim: scalar, number of capsules in this layer.
          layer_name: string, Name of this layer.
          input_atoms: scalar, number of units in each capsule of input layer.
          output_atoms: scalar, number of units in each capsule of output layer.
          stride: scalar, stride of the convolutional kernel.
          kernel_size: scalar, convolutional kernels are [kernel_size, kernel_size].
          padding: 'SAME' or 'VALID', padding mechanism for convolutional kernels.
          **routing_args: dictionary {leaky, num_routing}, args to be passed to the
            update_routing function.

        Returns:
          Tensor of activations for this layer of shape
            `[batch, output_dim, output_atoms, out_height, out_width]`. If padding is
            'SAME', out_height = in_height and out_width = in_width. Otherwise, height
            and width is adjusted with same rules as 'VALID' in tf.nn.conv2d.
        """


        input_shape = K.shape(input_tensor)

        input_transposed = tf.transpose(input_tensor, [0, 3, 1, 2, 4])
        input_tensor_reshaped = K.reshape(input_transposed, [
            input_shape[0] * input_shape[3], input_shape[1], input_shape[2], self.input_num_atoms])
        input_tensor_reshaped.set_shape((None, self.input_height, self.input_width, self.input_num_atoms))


        if self.upsamp_type == 'resize':
            upsamp = K.resize_images(input_tensor_reshaped, self.scaling, self.scaling, 'channels_last')
            outputs = K.conv2d(upsamp, kernel=self.W, strides=(1, 1), padding=self.padding, data_format='channels_last')
        elif self.upsamp_type == 'subpix':
            conv = K.conv2d(input_tensor_reshaped, kernel=self.W, strides=(1, 1), padding='same',
                            data_format='channels_last')
            outputs = tf.depth_to_space(conv, self.scaling)
        else:
            batch_size = input_shape[0] * input_shape[3]

            # Infer the dynamic output shape:
            out_height = deconv_length(input_shape[1], self.scaling, self.kernel_size, self.padding, None)
            out_width = deconv_length(input_shape[2], self.scaling, self.kernel_size, self.padding, None)
            output_shape = (batch_size, out_height, out_width, self.num_capsule * self.num_atoms)

            outputs = K.conv2d_transpose(input_tensor_reshaped, self.W, output_shape, (self.scaling, self.scaling),
                                     padding=self.padding, data_format='channels_last')

        votes_shape = K.shape(outputs)
        # Reshape back to 6D by splitting first dimmension to batch and input_dim
        # and splitting last dimmension to output_dim and output_atoms.

        votes = K.reshape(outputs, [input_shape[0], input_shape[3], votes_shape[1], votes_shape[2],
                                 self.num_capsule, self.num_atoms])
        conv_height = deconv_length(self.input_height, self.scaling, self.kernel_size, self.padding, None)
        conv_width = deconv_length(self.input_width, self.scaling, self.kernel_size, self.padding, None)
        votes.set_shape((None, self.input_num_capsule, conv_height, conv_width,
                         self.num_capsule, self.num_atoms))

        # Trying to do a for loop but it's not working... tried tf.map_fn and other but not working...
        # input_stack = tf.unstack(input_tensor, axis=3)
        # conv_list = []
        # for cap_grid in input_stack:
        #     conv_list.append(K.conv2d(cap_grid, self.W, (self.strides, self.strides),
        #                                     padding=self.padding, data_format='channels_last'))
        # conv = K.stack(conv_list)
        # # Reshape back to 6D by splitting last dimmension to output_dim and output_atoms.
        # conv_transposed = tf.transpose(conv, [1, 0, 2, 3, 4])
        # votes_shape = K.shape(conv_transposed)
        # _, _, conv_height, conv_width, _ = conv_transposed.get_shape()
        #
        # votes = K.reshape(conv_transposed, [input_shape[0], self.input_num_capsule, votes_shape[1], votes_shape[2],
        #                          self.num_capsule, self.num_atoms])
        # votes.set_shape((None, self.input_num_capsule, conv_height.value, conv_width.value,
        #                  self.num_capsule, self.num_atoms))

        logit_shape = K.stack([
            input_shape[0], input_shape[3], votes_shape[1], votes_shape[2], self.num_capsule])
        biases_replicated = K.tile(self.b, [votes_shape[1], votes_shape[2], 1, 1])

        activations = _update_routing(
            votes=votes,
            biases=biases_replicated,
            logit_shape=logit_shape,
            num_dims=6,
            input_dim=self.input_num_capsule,
            output_dim=self.num_capsule,
            num_routing=self.routings)

        return activations

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)

        output_shape[1] = deconv_length(output_shape[1], self.scaling, self.kernel_size, self.padding, None)
        output_shape[2] = deconv_length(output_shape[2], self.scaling, self.kernel_size, self.padding, None)
        output_shape[3] = self.num_capsule
        output_shape[4] = self.num_atoms

        return tuple(output_shape)

    def get_config(self):
        config = {
            'kernel_size': self.kernel_size,
            'num_capsule': self.num_capsule,
            'num_atoms': self.num_atoms,
            'scaling': self.scaling,
            'padding': self.padding,
            'upsamp_type': self.upsamp_type,
            'routings': self.routings,
            'kernel_initializer': initializers.serialize(self.kernel_initializer)
        }
        base_config = super(DeconvCapsuleLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def _update_routing(votes, biases, logit_shape, num_dims, input_dim, output_dim,
                    num_routing):
    """Sums over scaled votes and applies squash to compute the activations.

    Iteratively updates routing logits (scales) based on the similarity between
    the activation of this layer and the votes of the layer below.

    Args:
    votes: tensor, The transformed outputs of the layer below.
    biases: tensor, Bias variable.
    logit_shape: tensor, shape of the logit to be initialized.
    num_dims: scalar, number of dimmensions in votes. For fully connected
      capsule it is 4, for convolutional 6.
    input_dim: scalar, number of capsules in the input layer.
    output_dim: scalar, number of capsules in the output layer.
    num_routing: scalar, Number of routing iterations.
    leaky: boolean, if set use leaky routing.

    Returns:
    The activation tensor of the output layer after num_routing iterations.
    """
    if num_dims == 6:
        votes_t_shape = [5, 0, 1, 2, 3, 4]
        r_t_shape = [1, 2, 3, 4, 5, 0]
    elif num_dims == 4:
        votes_t_shape = [3, 0, 1, 2]
        r_t_shape = [1, 2, 3, 0]
    else:
        raise NotImplementedError('Not implemented')

    votes_trans = tf.transpose(votes, votes_t_shape)

    def _body(i, logits, activations):
        """Routing while loop."""
        # route: [batch, input_dim, output_dim, ...]
        route = tf.nn.softmax(logits, axis=-1)
        preactivate_unrolled = route * votes_trans
        preact_trans = tf.transpose(preactivate_unrolled, r_t_shape)
        preactivate = tf.reduce_sum(preact_trans, axis=1) + biases
        activation = _squash(preactivate)
        activations = activations.write(i, activation)
        act_3d = K.expand_dims(activation, 1)
        tile_shape = np.ones(num_dims, dtype=np.int32).tolist()
        tile_shape[1] = input_dim
        act_replicated = tf.tile(act_3d, tile_shape)
        distances = tf.reduce_sum(votes * act_replicated, axis=-1)
        logits += distances
        return (i + 1, logits, activations)

    activations = tf.TensorArray(
      dtype=tf.float32, size=num_routing, clear_after_read=False)
    logits = tf.fill(logit_shape, 0.0)

    # logits = K.reshape(logits, (-1, logits.get_shape()[1].value, height.value,
    #                             width.value, caps.value))
    # logits.set_shape((logits.get_shape()[0].value, logits.get_shape()[1].value, height.value, width.value, caps.value))
    #
    # votes = K.reshape(votes, (-1, logits.get_shape()[1].value, votes.get_shape()[2].value,
    #                  votes.get_shape()[3].value, votes.get_shape()[4].value, votes.get_shape()[5].value))
    # votes.set_shape((votes.get_shape()[0].value, logits.get_shape()[1].value, votes.get_shape()[2].value,
    #                  votes.get_shape()[3].value, votes.get_shape()[4].value, votes.get_shape()[5].value))

    i = tf.constant(0, dtype=tf.int32)
    _, logits, activations = tf.while_loop(
      lambda i, logits, activations: i < num_routing,
      _body,
      loop_vars=[i, logits, activations],
      swap_memory=True)

    return K.cast(activations.read(num_routing - 1), dtype='float32')


def _squash(input_tensor):
    norm = tf.norm(input_tensor, axis=-1, keepdims=True)
    norm_squared = norm * norm
    return (input_tensor / norm) * (norm_squared / (1 + norm_squared))
