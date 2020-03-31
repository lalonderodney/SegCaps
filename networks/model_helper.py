'''
Capsules for Object Segmentation (SegCaps)
Original Paper by Rodney LaLonde and Ulas Bagci (https://arxiv.org/abs/1804.04241)
Code written by: Rodney LaLonde
If you use significant portions of this code or the ideas from our paper, please cite it :)
If you have any questions, please email me at lalonde@knights.ucf.edu.

This is a helper file for choosing which model to create.
'''
from typing import Tuple, List, Any


def create_model(net: str, r1: int, r2: int, input_shape: Tuple[Any, Any, int], compute_resource_strategy: Any) -> List:
    """
    This is a helper function to create the requested model by calling the respective function needed.

    :param net: Name of the network chosen by user.
    :param r1: Number of routing iterations (when not changing spatial resolution) if using a capsule network.
    :param r2: Number of routing iterations (when changing spatial resolution) if using a capsule network.
    :param compute_resource_strategy: Scoping for multi-GPU compatibility.
    :param input_shape: Tuple(height, width, num_slices) Shape of input to network.
    :return: List of 3 or 1 TF models depending on capsule network or CNN, respectively.
    """
    if net == 'unet':
        from networks.unet import UNet
        with compute_resource_strategy.scope():
            model = UNet(input_shape)
        return [model]
    elif net == 'd_unet':
        from networks.unet import UNet
        with compute_resource_strategy.scope():
            model = UNet(input_shape, 4.68)
        return [model]

    elif net == 'phnn':
        from networks.phnn import PHNN
        with compute_resource_strategy.scope():
            model = PHNN(input_shape)
        return [model]
    elif net == 'd_phnn':
        from networks.phnn import PHNN
        with compute_resource_strategy.scope():
            model = PHNN(input_shape, 3.2)
        return [model]

    elif net == 'tiramisu56':
        from networks.densenets import DenseNetFCN
        with compute_resource_strategy.scope():
            model = DenseNetFCN(input_shape, growth_rate=12, nb_layers_per_block=4)
        return [model]
    elif net == 'tiramisu67':
        from networks.densenets import DenseNetFCN
        with compute_resource_strategy.scope():
            model = DenseNetFCN(input_shape, growth_rate=16, nb_layers_per_block=5)
        return [model]
    elif net == 'tiramisu103':
        from networks.densenets import DenseNetFCN
        with compute_resource_strategy.scope():
            model = DenseNetFCN(input_shape, growth_rate=16, nb_layers_per_block=[4, 5, 7, 10, 12, 15])
        return [model]

    elif net == 'segcaps':
        from networks.segcaps import SegCaps
        with compute_resource_strategy.scope():
            model_list = SegCaps(input_shape, r1, r2)
        return model_list
    elif net == 'segcapsbasic':
        from networks.segcaps import CapsNetBasic
        with compute_resource_strategy.scope():
            model_list = CapsNetBasic(input_shape)
        return model_list
    else:
        raise Exception('Unknown network type specified: {}'.format(net))
