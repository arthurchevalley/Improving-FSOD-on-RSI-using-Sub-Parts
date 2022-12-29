# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule, auto_fp16
from mmcv.ops import DeformConv2dPack
from mmdet.models.builder import NECKS, build_loss

import torch
@NECKS.register_module()
class FaPN(BaseModule):
    r"""Feature Pyramid Network.

    This is an implementation of paper `Feature Pyramid Networks for Object
    Detection <https://arxiv.org/abs/1612.03144>`_.

    Args:
        in_channels (list[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale).
        num_outs (int): Number of output scales.
        start_level (int): Index of the start input backbone level used to
            build the feature pyramid. Default: 0.
        end_level (int): Index of the end input backbone level (exclusive) to
            build the feature pyramid. Default: -1, which means the last level.
        add_extra_convs (bool | str): If bool, it decides whether to add conv
            layers on top of the original feature maps. Default to False.
            If True, it is equivalent to `add_extra_convs='on_input'`.
            If str, it specifies the source feature map of the extra convs.
            Only the following options are allowed

            - 'on_input': Last feat map of neck inputs (i.e. backbone feature).
            - 'on_lateral': Last feature map after lateral convs.
            - 'on_output': The last output feature map after fpn convs.
        relu_before_extra_convs (bool): Whether to apply relu before the extra
            conv. Default: False.
        no_norm_on_lateral (bool): Whether to apply norm on lateral.
            Default: False.
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        act_cfg (dict): Config dict for activation layer in ConvModule.
            Default: None.
        upsample_cfg (dict): Config dict for interpolate layer.
            Default: dict(mode='nearest').
        init_cfg (dict or list[dict], optional): Initialization config dict.

    Example:
        >>> import torch
        >>> in_channels = [2, 3, 5, 7]
        >>> scales = [340, 170, 84, 43]
        >>> inputs = [torch.rand(1, c, s, s)
        ...           for c, s in zip(in_channels, scales)]
        >>> self = FPN(in_channels, 11, len(in_channels)).eval()
        >>> outputs = self.forward(inputs)
        >>> for i in range(len(outputs)):
        ...     print(f'outputs[{i}].shape = {outputs[i].shape}')
        outputs[0].shape = torch.Size([1, 11, 340, 340])
        outputs[1].shape = torch.Size([1, 11, 170, 170])
        outputs[2].shape = torch.Size([1, 11, 84, 84])
        outputs[3].shape = torch.Size([1, 11, 43, 43])
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 dilatation_size = [3,12,24],
                 add_extra_convs=False,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 upsample_cfg=dict(mode='nearest'),
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d', distribution='uniform')):
        super(FaPN, self).__init__(init_cfg)
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg.copy()
        self.dilatation_size = dilatation_size


        if end_level == -1 or end_level == self.num_ins - 1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level is not the last level, no extra level is allowed
            self.backbone_end_level = end_level + 1
            assert end_level < self.num_ins
            assert num_outs == end_level - start_level + 1
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        assert isinstance(add_extra_convs, (str, bool))
        if isinstance(add_extra_convs, str):
            # Extra_convs_source choices: 'on_input', 'on_lateral', 'on_output'
            assert add_extra_convs in ('on_input', 'on_lateral', 'on_output')
        elif add_extra_convs:  # True
            self.add_extra_convs = 'on_input'

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            if i != self.backbone_end_level - 1:
                l_conv = FSM(in_chan=in_channels[i], out_chan=out_channels, norm="")
                self.lateral_convs.append(l_conv)
            else:
                l_conv = self.fapn_conv(self.in_channels[i], self.out_channels, 3, 1, 1, self.dilatation_size)
                self.lateral_convs += l_conv

            fpn_conv = ConvModule(
                    out_channels,
                    out_channels,
                    3,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=False)
            self.fpn_convs.append(fpn_conv)
        # add extra conv layers (e.g., RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if self.add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.add_extra_convs == 'on_input':
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels
                extra_fpn_conv = ConvModule(
                    in_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    inplace=False)
                self.fpn_convs.append(extra_fpn_conv)

    def fapn_conv(self, in_channels, out_channels, kernel_size, stride, padding, dilatation_size):
        fapn_branch = []
        
        for id_dil, dil in enumerate(dilatation_size):
            extra_fpn_conv = ConvModule(
                    in_channels,
                    2*out_channels,
                    1,
                    stride=1,
                    padding=0,
                    act_cfg =None,
                    inplace=False)
            #fapn_def_conv = DeformConv2dPack(
            fapn_def_conv = ConvModule(
                        in_channels=2*out_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        stride= stride,
                        padding= dil,
                        dilation=dil)
            in_channels = in_channels+out_channels


            fapn_branch.append(extra_fpn_conv)
            fapn_branch.append(fapn_def_conv)

        extra_fpn_conv = ConvModule(
                    len(dilatation_size)*out_channels,
                    out_channels,
                    1,
                    stride=1,
                    padding=0,
                    act_cfg =None,
                    inplace=False)
        fapn_branch.append(extra_fpn_conv)
        extra_fpn_conv = ConvModule(
                    out_channels,
                    2,
                    1,
                    stride=1,
                    padding=0,
                    act_cfg =None,
                    inplace=False)
        fapn_branch.append(extra_fpn_conv)
        return fapn_branch

    
    @auto_fp16()
    def forward(self, inputs):
        """Forward function."""
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = []
        for i in range(self.backbone_end_level -1):
            laterals.append(self.lateral_convs[i](inputs[i + self.start_level]))
        # fapn branch

        input_fapn = inputs[self.backbone_end_level - 1 + self.start_level]

        input_fapn_def_conv = []
        for i in range(self.backbone_end_level-1,len(self.lateral_convs)-2, 2):
            # take lateral i then i+1, apply input_fapn to them then concat with 
            # the corresponding input end the ith iteration
            
            input_fapn_conv = self.lateral_convs[i](input_fapn)

            input_fapn_def_conv.append(self.lateral_convs[i+1](input_fapn_conv))
            
            if i != len(self.lateral_convs) - 4:
                input_fapn = torch.cat((input_fapn,input_fapn_def_conv[-1]),dim=1)
            else:
                input_fapn = torch.cat([input_fapn_def_conv[i] for i in range(len(input_fapn_def_conv))], dim=1)
        
        out_fapn = self.lateral_convs[-2](input_fapn)

        # fapn loss
        #out_loss = self.lateral_convs[-1](out_fapn)
        #out_loss = nn.Softmax(dim=1)(out_loss)
        
        #loss_fapn = self.loss_fapn(out_loss, gt, img_size)
        #print(dict().shape)
        # build top-down path
        used_backbone_levels = len(laterals)
        
        for i in range(used_backbone_levels, 0, -1):
            # In some cases, fixing `scale factor` (e.g. 2) is preferred, but
        #  it cannot co-exist with `size` in `F.interpolate`.
        
            if i == used_backbone_levels:
                if 'scale_factor' in self.upsample_cfg:
                    # fix runtime error of "+=" inplace operation in PyTorch 1.10
                    laterals[i - 1] = laterals[i - 1] + F.interpolate(
                        out_fapn, **self.upsample_cfg)
                else:
                    prev_shape = laterals[i - 1].shape[2:]
                    laterals[i - 1] = laterals[i - 1] + F.interpolate(
                        out_fapn, size=prev_shape, **self.upsample_cfg)

            else:
                if 'scale_factor' in self.upsample_cfg:
                    # fix runtime error of "+=" inplace operation in PyTorch 1.10
                    laterals[i - 1] = laterals[i - 1] + F.interpolate(
                        laterals[i], **self.upsample_cfg)
                else:
                    prev_shape = laterals[i - 1].shape[2:]
                    laterals[i - 1] = laterals[i - 1] + F.interpolate(
                        laterals[i], size=prev_shape, **self.upsample_cfg)

        
        # build outputs
        # part 1: from original levels
        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]

        # part 2: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.add_extra_convs == 'on_input':
                    extra_source = inputs[self.backbone_end_level - 1]
                elif self.add_extra_convs == 'on_lateral':
                    extra_source = laterals[-1]
                elif self.add_extra_convs == 'on_output':
                    extra_source = outs[-1]
                else:
                    raise NotImplementedError
                outs.append(self.fpn_convs[used_backbone_levels](extra_source))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))
        return tuple(outs)

@NECKS.register_module()
class DefFaPN(BaseModule):
    r"""Feature Pyramid Network.

    This is an implementation of paper `Feature Pyramid Networks for Object
    Detection <https://arxiv.org/abs/1612.03144>`_.

    Args:
        in_channels (list[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale).
        num_outs (int): Number of output scales.
        start_level (int): Index of the start input backbone level used to
            build the feature pyramid. Default: 0.
        end_level (int): Index of the end input backbone level (exclusive) to
            build the feature pyramid. Default: -1, which means the last level.
        add_extra_convs (bool | str): If bool, it decides whether to add conv
            layers on top of the original feature maps. Default to False.
            If True, it is equivalent to `add_extra_convs='on_input'`.
            If str, it specifies the source feature map of the extra convs.
            Only the following options are allowed

            - 'on_input': Last feat map of neck inputs (i.e. backbone feature).
            - 'on_lateral': Last feature map after lateral convs.
            - 'on_output': The last output feature map after fpn convs.
        relu_before_extra_convs (bool): Whether to apply relu before the extra
            conv. Default: False.
        no_norm_on_lateral (bool): Whether to apply norm on lateral.
            Default: False.
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        act_cfg (dict): Config dict for activation layer in ConvModule.
            Default: None.
        upsample_cfg (dict): Config dict for interpolate layer.
            Default: dict(mode='nearest').
        init_cfg (dict or list[dict], optional): Initialization config dict.

    Example:
        >>> import torch
        >>> in_channels = [2, 3, 5, 7]
        >>> scales = [340, 170, 84, 43]
        >>> inputs = [torch.rand(1, c, s, s)
        ...           for c, s in zip(in_channels, scales)]
        >>> self = FPN(in_channels, 11, len(in_channels)).eval()
        >>> outputs = self.forward(inputs)
        >>> for i in range(len(outputs)):
        ...     print(f'outputs[{i}].shape = {outputs[i].shape}')
        outputs[0].shape = torch.Size([1, 11, 340, 340])
        outputs[1].shape = torch.Size([1, 11, 170, 170])
        outputs[2].shape = torch.Size([1, 11, 84, 84])
        outputs[3].shape = torch.Size([1, 11, 43, 43])
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 dilatation_size = [3,12,24],
                 add_extra_convs=False,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 upsample_cfg=dict(mode='nearest'),
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d', distribution='uniform')):
        super(DefFaPN, self).__init__(init_cfg)
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg.copy()
        self.dilatation_size = dilatation_size


        if end_level == -1 or end_level == self.num_ins - 1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level is not the last level, no extra level is allowed
            self.backbone_end_level = end_level + 1
            assert end_level < self.num_ins
            assert num_outs == end_level - start_level + 1
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        assert isinstance(add_extra_convs, (str, bool))
        if isinstance(add_extra_convs, str):
            # Extra_convs_source choices: 'on_input', 'on_lateral', 'on_output'
            assert add_extra_convs in ('on_input', 'on_lateral', 'on_output')
        elif add_extra_convs:  # True
            self.add_extra_convs = 'on_input'

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            if i != self.backbone_end_level - 1:
                l_conv = FSM(in_chan=in_channels[i], out_chan=out_channels, norm="")
                self.lateral_convs.append(l_conv)
            else:
                l_conv = self.fapn_conv(self.in_channels[i], self.out_channels, 3, 1, 1, self.dilatation_size)
                self.lateral_convs += l_conv

            fpn_conv = ConvModule(
                    out_channels,
                    out_channels,
                    3,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=False)
            self.fpn_convs.append(fpn_conv)
        # add extra conv layers (e.g., RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if self.add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.add_extra_convs == 'on_input':
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels
                extra_fpn_conv = ConvModule(
                    in_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    inplace=False)
                self.fpn_convs.append(extra_fpn_conv)

    def fapn_conv(self, in_channels, out_channels, kernel_size, stride, padding, dilatation_size):
        fapn_branch = []
        
        for id_dil, dil in enumerate(dilatation_size):
            extra_fpn_conv = ConvModule(
                    in_channels,
                    2*out_channels,
                    1,
                    stride=1,
                    padding=0,
                    act_cfg =None,
                    inplace=False)
            fapn_def_conv = DeformConv2dPack(
                        in_channels=2*out_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        stride= stride,
                        padding= dil,
                        dilation=dil)
            in_channels = in_channels+out_channels


            fapn_branch.append(extra_fpn_conv)
            fapn_branch.append(fapn_def_conv)

        extra_fpn_conv = ConvModule(
                    len(dilatation_size)*out_channels,
                    out_channels,
                    1,
                    stride=1,
                    padding=0,
                    act_cfg =None,
                    inplace=False)
        fapn_branch.append(extra_fpn_conv)
        extra_fpn_conv = ConvModule(
                    out_channels,
                    2,
                    1,
                    stride=1,
                    padding=0,
                    act_cfg =None,
                    inplace=False)
        fapn_branch.append(extra_fpn_conv)
        return fapn_branch

    
    @auto_fp16()
    def forward(self, inputs):
        """Forward function."""
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = []
        for i in range(self.backbone_end_level -1):
            laterals.append(self.lateral_convs[i](inputs[i + self.start_level]))
        # fapn branch

        input_fapn = inputs[self.backbone_end_level - 1 + self.start_level]

        input_fapn_def_conv = []
        for i in range(self.backbone_end_level-1,len(self.lateral_convs)-2, 2):
            # take lateral i then i+1, apply input_fapn to them then concat with 
            # the corresponding input end the ith iteration
            
            input_fapn_conv = self.lateral_convs[i](input_fapn)

            input_fapn_def_conv.append(self.lateral_convs[i+1](input_fapn_conv))
            
            if i != len(self.lateral_convs) - 4:
                input_fapn = torch.cat((input_fapn,input_fapn_def_conv[-1]),dim=1)
            else:
                input_fapn = torch.cat([input_fapn_def_conv[i] for i in range(len(input_fapn_def_conv))], dim=1)
        
        out_fapn = self.lateral_convs[-2](input_fapn)

        # fapn loss
        #out_loss = self.lateral_convs[-1](out_fapn)
        #out_loss = nn.Softmax(dim=1)(out_loss)
        
        #loss_fapn = self.loss_fapn(out_loss, gt, img_size)
        #print(dict().shape)
        # build top-down path
        used_backbone_levels = len(laterals)
        
        for i in range(used_backbone_levels, 0, -1):
            # In some cases, fixing `scale factor` (e.g. 2) is preferred, but
        #  it cannot co-exist with `size` in `F.interpolate`.
        
            if i == used_backbone_levels:
                if 'scale_factor' in self.upsample_cfg:
                    # fix runtime error of "+=" inplace operation in PyTorch 1.10
                    laterals[i - 1] = laterals[i - 1] + F.interpolate(
                        out_fapn, **self.upsample_cfg)
                else:
                    prev_shape = laterals[i - 1].shape[2:]
                    laterals[i - 1] = laterals[i - 1] + F.interpolate(
                        out_fapn, size=prev_shape, **self.upsample_cfg)

            else:
                if 'scale_factor' in self.upsample_cfg:
                    # fix runtime error of "+=" inplace operation in PyTorch 1.10
                    laterals[i - 1] = laterals[i - 1] + F.interpolate(
                        laterals[i], **self.upsample_cfg)
                else:
                    prev_shape = laterals[i - 1].shape[2:]
                    laterals[i - 1] = laterals[i - 1] + F.interpolate(
                        laterals[i], size=prev_shape, **self.upsample_cfg)

        
        # build outputs
        # part 1: from original levels
        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]

        # part 2: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.add_extra_convs == 'on_input':
                    extra_source = inputs[self.backbone_end_level - 1]
                elif self.add_extra_convs == 'on_lateral':
                    extra_source = laterals[-1]
                elif self.add_extra_convs == 'on_output':
                    extra_source = outs[-1]
                else:
                    raise NotImplementedError
                outs.append(self.fpn_convs[used_backbone_levels](extra_source))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))
        return tuple(outs)



@NECKS.register_module()
class DenseFaPN(BaseModule):
    r"""Feature Pyramid Network.

    This is an implementation of paper `Feature Pyramid Networks for Object
    Detection <https://arxiv.org/abs/1612.03144>`_.

    Args:
        in_channels (list[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale).
        num_outs (int): Number of output scales.
        start_level (int): Index of the start input backbone level used to
            build the feature pyramid. Default: 0.
        end_level (int): Index of the end input backbone level (exclusive) to
            build the feature pyramid. Default: -1, which means the last level.
        add_extra_convs (bool | str): If bool, it decides whether to add conv
            layers on top of the original feature maps. Default to False.
            If True, it is equivalent to `add_extra_convs='on_input'`.
            If str, it specifies the source feature map of the extra convs.
            Only the following options are allowed

            - 'on_input': Last feat map of neck inputs (i.e. backbone feature).
            - 'on_lateral': Last feature map after lateral convs.
            - 'on_output': The last output feature map after fpn convs.
        relu_before_extra_convs (bool): Whether to apply relu before the extra
            conv. Default: False.
        no_norm_on_lateral (bool): Whether to apply norm on lateral.
            Default: False.
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        act_cfg (dict): Config dict for activation layer in ConvModule.
            Default: None.
        upsample_cfg (dict): Config dict for interpolate layer.
            Default: dict(mode='nearest').
        init_cfg (dict or list[dict], optional): Initialization config dict.

    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 int_dilatation_size = [6, 12, 24],
                 CIR_dilatation_size = [3,6,12,18,24],
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 upsample_cfg=dict(mode='nearest'),
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d', distribution='uniform')):
        super(DenseFaPN, self).__init__(init_cfg)
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg.copy()

        self.int_dilatation_size = [6, 12, 24]
        self.CIR_dilatation_size = [3,6,12,18,24]


        if end_level == -1 or end_level == self.num_ins - 1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level is not the last level, no extra level is allowed
            self.backbone_end_level = end_level + 1
            assert end_level < self.num_ins
            assert num_outs == end_level - start_level + 1
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        assert isinstance(add_extra_convs, (str, bool))
        if isinstance(add_extra_convs, str):
            # Extra_convs_source choices: 'on_input', 'on_lateral', 'on_output'
            assert add_extra_convs in ('on_input', 'on_lateral', 'on_output')
        elif add_extra_convs:  # True
            self.add_extra_convs = 'on_input'

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            
            if i != self.backbone_end_level - 1:
                if not len(self.int_dilatation_size):
                    l_conv = ConvModule(
                        in_channels[i],
                        out_channels,
                        1,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                        act_cfg=act_cfg,
                        inplace=False)
                    self.lateral_convs.append(l_conv)
                else:
                    l_conv = self.int_cir_conv(self.in_channels[i], self.out_channels, 3, 1, 1, self.int_dilatation_size)
                    self.lateral_convs += l_conv
                
            else:
                l_conv = self.cir_conv(self.in_channels[i], self.out_channels, 3, 1, 1, self.CIR_dilatation_size)
                self.lateral_convs += l_conv

            fpn_conv = ConvModule(
                    out_channels,
                    out_channels,
                    3,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=False)
            self.fpn_convs.append(fpn_conv)
        # add extra conv layers (e.g., RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if self.add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.add_extra_convs == 'on_input':
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels
                extra_fpn_conv = ConvModule(
                    in_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    inplace=False)
                self.fpn_convs.append(extra_fpn_conv)

    def cir_conv(self, in_channels, out_channels, kernel_size, stride, padding, dilatation_size):
        cir_branch = []
        
        for id_dil, dil in enumerate(dilatation_size):
            extra_fpn_conv = ConvModule(
                    in_channels,
                    2*out_channels,
                    1,
                    stride=1,
                    padding=0,
                    act_cfg =None,
                    inplace=False)
            cir_def_conv = DeformConv2dPack(
                        in_channels=2*out_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        stride= stride,
                        padding= dil,
                        dilation=dil,
                        act_cfg =None,
                        inplace=False)
            in_channels = in_channels+out_channels


            cir_branch.append(extra_fpn_conv)
            cir_branch.append(cir_def_conv)

        extra_fpn_conv = ConvModule(
                    len(dilatation_size)*out_channels,
                    out_channels,
                    1,
                    stride=1,
                    padding=0,
                    act_cfg =None,
                    inplace=False)
        cir_branch.append(extra_fpn_conv)
        extra_fpn_conv = ConvModule(
                    out_channels,
                    2,
                    1,
                    stride=1,
                    padding=0,
                    act_cfg =None,
                    inplace=False)
        cir_branch.append(extra_fpn_conv)
        return cir_branch

    def int_cir_conv(self, in_channels, out_channels, kernel_size, stride, padding, dilatation_size):
        cir_branch = []
        
        for id_dil, dil in enumerate(dilatation_size):

            extra_fpn_conv = FSM(
                    in_channels,
                    2*out_channels,
                    None)
                    
            cir_def_conv = DeformConv2dPack(
                        in_channels=2*out_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        stride= stride,
                        padding= dil,
                        dilation=dil,
                        act_cfg =None,
                        inplace=False)
            in_channels = in_channels+out_channels


            cir_branch.append(extra_fpn_conv)
            cir_branch.append(cir_def_conv)

        extra_fpn_conv = ConvModule(
                    len(dilatation_size)*out_channels,
                    out_channels,
                    1,
                    stride=1,
                    padding=0,
                    act_cfg =None,
                    inplace=False)
        cir_branch.append(extra_fpn_conv)
        
        return cir_branch

    
    @auto_fp16()
    def forward(self, inputs):
        """Forward function."""

        assert len(inputs) == len(self.in_channels)
        if len(self.int_dilatation_size):
            # build laterals
            laterals = []
            for i in range(0, (self.backbone_end_level -1)*(len(self.int_dilatation_size)*2+1), len(self.int_dilatation_size)*2+1):

                input_cir_conv = inputs[int(i/(len(self.int_dilatation_size)*2+1)) + self.start_level].clone()
                input_cir_def_conv = [input_cir_conv]
                for j in range(0,len(self.int_dilatation_size)*2,2):
                    input_cir_conv = self.lateral_convs[i+j](input_cir_conv)
                    input_cir_def_conv.append(self.lateral_convs[i+j+1](input_cir_conv))
                    input_cir_conv = torch.cat([input_cir_def_conv[i] for i in range(len(input_cir_def_conv))], dim=1)
                input_cir_def_conv = input_cir_def_conv[1:]
                input_cir_conv = torch.cat([input_cir_def_conv[i] for i in range(len(input_cir_def_conv))], dim=1)
                laterals.append(self.lateral_convs[i+len(self.int_dilatation_size)*2](input_cir_conv[1:]))
        else:
            laterals = []
            for i in range(self.backbone_end_level -1):
                laterals.append(self.lateral_convs[i](inputs[i + self.start_level]))
        # CIR branch

        input_cir = inputs[self.backbone_end_level - 1 + self.start_level]

        input_cir_def_conv = []
        for i in range((self.backbone_end_level -1)*(len(self.int_dilatation_size)*2+1),len(self.lateral_convs)-2, 2):
            # take lateral i then i+1, apply input_cir to them then concat with 
            # the corresponding input end the ith iteration
            
            input_cir_conv = self.lateral_convs[i](input_cir)

            input_cir_def_conv.append(self.lateral_convs[i+1](input_cir_conv))
            
            if i != len(self.lateral_convs) - 4:
                input_cir = torch.cat((input_cir,input_cir_def_conv[-1]),dim=1)
            else:
                input_cir = torch.cat([input_cir_def_conv[i] for i in range(len(input_cir_def_conv))], dim=1)
        
        out_cir = self.lateral_convs[-2](input_cir)

        # CIR loss
        #out_loss = self.lateral_convs[-1](out_cir)
        #out_loss = nn.Softmax(dim=1)(out_loss)
        
        #loss_cir = self.loss_cir(out_loss, gt, img_size)
        #print(dict().shape)
        # build top-down path
        used_backbone_levels = len(laterals)
        
        for i in range(used_backbone_levels, 0, -1):
            # In some cases, fixing `scale factor` (e.g. 2) is preferred, but
        #  it cannot co-exist with `size` in `F.interpolate`.
        
            if i == used_backbone_levels:
                if 'scale_factor' in self.upsample_cfg:
                    # fix runtime error of "+=" inplace operation in PyTorch 1.10
                    laterals[i - 1] = laterals[i - 1] + F.interpolate(
                        out_cir, **self.upsample_cfg)
                else:
                    prev_shape = laterals[i - 1].shape[2:]
                    laterals[i - 1] = laterals[i - 1] + F.interpolate(
                        out_cir, size=prev_shape, **self.upsample_cfg)

            else:
                if 'scale_factor' in self.upsample_cfg:
                    # fix runtime error of "+=" inplace operation in PyTorch 1.10
                    laterals[i - 1] = laterals[i - 1] + F.interpolate(
                        laterals[i], **self.upsample_cfg)
                else:
                    prev_shape = laterals[i - 1].shape[2:]
                    laterals[i - 1] = laterals[i - 1] + F.interpolate(
                        laterals[i], size=prev_shape, **self.upsample_cfg)

        
        # build outputs
        # part 1: from original levels
        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]

        # part 2: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.add_extra_convs == 'on_input':
                    extra_source = inputs[self.backbone_end_level - 1]
                elif self.add_extra_convs == 'on_lateral':
                    extra_source = laterals[-1]
                elif self.add_extra_convs == 'on_output':
                    extra_source = outs[-1]
                else:
                    raise NotImplementedError
                outs.append(self.fpn_convs[used_backbone_levels](extra_source))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))
        return tuple(outs)


@NECKS.register_module()
class DenseConvFaPN(BaseModule):
    r"""Feature Pyramid Network.

    This is an implementation of paper `Feature Pyramid Networks for Object
    Detection <https://arxiv.org/abs/1612.03144>`_.

    Args:
        in_channels (list[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale).
        num_outs (int): Number of output scales.
        start_level (int): Index of the start input backbone level used to
            build the feature pyramid. Default: 0.
        end_level (int): Index of the end input backbone level (exclusive) to
            build the feature pyramid. Default: -1, which means the last level.
        add_extra_convs (bool | str): If bool, it decides whether to add conv
            layers on top of the original feature maps. Default to False.
            If True, it is equivalent to `add_extra_convs='on_input'`.
            If str, it specifies the source feature map of the extra convs.
            Only the following options are allowed

            - 'on_input': Last feat map of neck inputs (i.e. backbone feature).
            - 'on_lateral': Last feature map after lateral convs.
            - 'on_output': The last output feature map after fpn convs.
        relu_before_extra_convs (bool): Whether to apply relu before the extra
            conv. Default: False.
        no_norm_on_lateral (bool): Whether to apply norm on lateral.
            Default: False.
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        act_cfg (dict): Config dict for activation layer in ConvModule.
            Default: None.
        upsample_cfg (dict): Config dict for interpolate layer.
            Default: dict(mode='nearest').
        init_cfg (dict or list[dict], optional): Initialization config dict.

    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 int_dilatation_size = [6, 12, 24],
                 CIR_dilatation_size = [3,6,12,18,24],
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 upsample_cfg=dict(mode='nearest'),
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d', distribution='uniform')):
        super(DenseConvFaPN, self).__init__(init_cfg)
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg.copy()

        self.int_dilatation_size = [6, 12, 24]
        self.CIR_dilatation_size = [3,6,12,18,24]


        if end_level == -1 or end_level == self.num_ins - 1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level is not the last level, no extra level is allowed
            self.backbone_end_level = end_level + 1
            assert end_level < self.num_ins
            assert num_outs == end_level - start_level + 1
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        assert isinstance(add_extra_convs, (str, bool))
        if isinstance(add_extra_convs, str):
            # Extra_convs_source choices: 'on_input', 'on_lateral', 'on_output'
            assert add_extra_convs in ('on_input', 'on_lateral', 'on_output')
        elif add_extra_convs:  # True
            self.add_extra_convs = 'on_input'

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            
            if i != self.backbone_end_level - 1:
                if not len(self.int_dilatation_size):
                    l_conv = ConvModule(
                        in_channels[i],
                        out_channels,
                        1,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                        act_cfg=act_cfg,
                        inplace=False)
                    self.lateral_convs.append(l_conv)
                else:
                    l_conv = self.int_cir_conv(self.in_channels[i], self.out_channels, 3, 1, 1, self.int_dilatation_size)
                    self.lateral_convs += l_conv
                
            else:
                l_conv = self.cir_conv(self.in_channels[i], self.out_channels, 3, 1, 1, self.CIR_dilatation_size)
                self.lateral_convs += l_conv

            fpn_conv = ConvModule(
                    out_channels,
                    out_channels,
                    3,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=False)
            self.fpn_convs.append(fpn_conv)
        # add extra conv layers (e.g., RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if self.add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.add_extra_convs == 'on_input':
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels
                extra_fpn_conv = ConvModule(
                    in_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    inplace=False)
                self.fpn_convs.append(extra_fpn_conv)

    def cir_conv(self, in_channels, out_channels, kernel_size, stride, padding, dilatation_size):
        cir_branch = []
        
        for id_dil, dil in enumerate(dilatation_size):
            extra_fpn_conv = ConvModule(
                    in_channels,
                    2*out_channels,
                    1,
                    stride=1,
                    padding=0,
                    act_cfg =None,
                    inplace=False)
            #cir_def_conv = DeformConv2dPack(
            cir_def_conv = ConvModule(
                        in_channels=2*out_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        stride= stride,
                        padding= dil,
                        dilation=dil,
                        conv_cfg=None,
                        norm_cfg=None,
                        act_cfg=None,
                        inplace=False)
            in_channels = in_channels+out_channels


            cir_branch.append(extra_fpn_conv)
            cir_branch.append(cir_def_conv)

        extra_fpn_conv = ConvModule(
                    len(dilatation_size)*out_channels,
                    out_channels,
                    1,
                    stride=1,
                    padding=0,
                    act_cfg =None,
                    inplace=False)
        cir_branch.append(extra_fpn_conv)
        extra_fpn_conv = ConvModule(
                    out_channels,
                    2,
                    1,
                    stride=1,
                    padding=0,
                    act_cfg =None,
                    inplace=False)
        cir_branch.append(extra_fpn_conv)
        return cir_branch

    def int_cir_conv(self, in_channels, out_channels, kernel_size, stride, padding, dilatation_size):
        cir_branch = []
        
        for id_dil, dil in enumerate(dilatation_size):

            extra_fpn_conv = FSM(
                    in_channels,
                    2*out_channels,
                    None)
                    
            cir_def_conv = ConvModule(
                        in_channels=2*out_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        stride= stride,
                        padding= dil,
                        dilation=dil,
                        conv_cfg=None,
                        norm_cfg=None,
                        act_cfg=None,
                        inplace=False)
            in_channels = in_channels+out_channels


            cir_branch.append(extra_fpn_conv)
            cir_branch.append(cir_def_conv)

        extra_fpn_conv = ConvModule(
                    len(dilatation_size)*out_channels,
                    out_channels,
                    1,
                    stride=1,
                    padding=0,
                    act_cfg =None,
                    inplace=False)
        cir_branch.append(extra_fpn_conv)
        
        return cir_branch

    
    @auto_fp16()
    def forward(self, inputs):
        """Forward function."""

        assert len(inputs) == len(self.in_channels)
        if len(self.int_dilatation_size):
            # build laterals
            laterals = []
            for i in range(0, (self.backbone_end_level -1)*(len(self.int_dilatation_size)*2+1), len(self.int_dilatation_size)*2+1):

                input_cir_conv = inputs[int(i/(len(self.int_dilatation_size)*2+1)) + self.start_level].clone()
                input_cir_def_conv = [input_cir_conv]
                for j in range(0,len(self.int_dilatation_size)*2,2):
                    input_cir_conv = self.lateral_convs[i+j](input_cir_conv)
                    input_cir_def_conv.append(self.lateral_convs[i+j+1](input_cir_conv))
                    input_cir_conv = torch.cat([input_cir_def_conv[i] for i in range(len(input_cir_def_conv))], dim=1)
                input_cir_def_conv = input_cir_def_conv[1:]
                input_cir_conv = torch.cat([input_cir_def_conv[i] for i in range(len(input_cir_def_conv))], dim=1)
                laterals.append(self.lateral_convs[i+len(self.int_dilatation_size)*2](input_cir_conv[1:]))
        else:
            laterals = []
            for i in range(self.backbone_end_level -1):
                laterals.append(self.lateral_convs[i](inputs[i + self.start_level]))
        # CIR branch

        input_cir = inputs[self.backbone_end_level - 1 + self.start_level]

        input_cir_def_conv = []
        for i in range((self.backbone_end_level -1)*(len(self.int_dilatation_size)*2+1),len(self.lateral_convs)-2, 2):
            # take lateral i then i+1, apply input_cir to them then concat with 
            # the corresponding input end the ith iteration
            
            input_cir_conv = self.lateral_convs[i](input_cir)

            input_cir_def_conv.append(self.lateral_convs[i+1](input_cir_conv))
            
            if i != len(self.lateral_convs) - 4:
                input_cir = torch.cat((input_cir,input_cir_def_conv[-1]),dim=1)
            else:
                input_cir = torch.cat([input_cir_def_conv[i] for i in range(len(input_cir_def_conv))], dim=1)
        
        out_cir = self.lateral_convs[-2](input_cir)

        # CIR loss
        #out_loss = self.lateral_convs[-1](out_cir)
        #out_loss = nn.Softmax(dim=1)(out_loss)
        
        #loss_cir = self.loss_cir(out_loss, gt, img_size)
        #print(dict().shape)
        # build top-down path
        used_backbone_levels = len(laterals)
        
        for i in range(used_backbone_levels, 0, -1):
            # In some cases, fixing `scale factor` (e.g. 2) is preferred, but
        #  it cannot co-exist with `size` in `F.interpolate`.
        
            if i == used_backbone_levels:
                if 'scale_factor' in self.upsample_cfg:
                    # fix runtime error of "+=" inplace operation in PyTorch 1.10
                    laterals[i - 1] = laterals[i - 1] + F.interpolate(
                        out_cir, **self.upsample_cfg)
                else:
                    prev_shape = laterals[i - 1].shape[2:]
                    laterals[i - 1] = laterals[i - 1] + F.interpolate(
                        out_cir, size=prev_shape, **self.upsample_cfg)

            else:
                if 'scale_factor' in self.upsample_cfg:
                    # fix runtime error of "+=" inplace operation in PyTorch 1.10
                    laterals[i - 1] = laterals[i - 1] + F.interpolate(
                        laterals[i], **self.upsample_cfg)
                else:
                    prev_shape = laterals[i - 1].shape[2:]
                    laterals[i - 1] = laterals[i - 1] + F.interpolate(
                        laterals[i], size=prev_shape, **self.upsample_cfg)

        
        # build outputs
        # part 1: from original levels
        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]

        # part 2: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.add_extra_convs == 'on_input':
                    extra_source = inputs[self.backbone_end_level - 1]
                elif self.add_extra_convs == 'on_lateral':
                    extra_source = laterals[-1]
                elif self.add_extra_convs == 'on_output':
                    extra_source = outs[-1]
                else:
                    raise NotImplementedError
                outs.append(self.fpn_convs[used_backbone_levels](extra_source))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))
        return tuple(outs)




class FeatureAlign(BaseModule):  # FaPN full version
    def __init__(self, in_nc=128, out_nc=128, norm=None):
        super(FeatureAlign, self).__init__()
        self.lateral_conv = FSM(in_nc, out_nc, norm="")
        self.offset = ConvModule(out_nc * 2, out_nc, kernel_size=1, stride=1, padding=0, bias=False, norm_cfg=norm)
        self.dcpack_L2 = DeformConv2dPack(out_nc, out_nc, 3, stride=1, padding=1, dilation=1, deformable_groups=8,
                                extra_offset_mask=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, feat_l, feat_s, main_path=None):
        HW = feat_l.size()[2:]
        if feat_l.size()[2:] != feat_s.size()[2:]:
            feat_up = F.interpolate(feat_s, HW, mode='bilinear', align_corners=False)
        else:
            feat_up = feat_s
        feat_arm = self.lateral_conv(feat_l)  # 0~1 * feats
        offset = self.offset(torch.cat([feat_arm, feat_up * 2], dim=1))  # concat for offset by compute the dif
        feat_align = self.relu(self.dcpack_L2([feat_up, offset], main_path))  # [feat, offset]
        return feat_align + feat_arm

class FSM(BaseModule):
    def __init__(self, in_chan, out_chan, norm="GN"):
        super(FSM, self).__init__()
        self.conv_atten = ConvModule(in_chan, in_chan, kernel_size=1, bias=False, norm_cfg=None)
        self.sigmoid = nn.Sigmoid()
        self.conv = ConvModule(in_chan, out_chan, kernel_size=1, bias=False, norm_cfg=None)


    def forward(self, x):
        atten = self.sigmoid(self.conv_atten(F.avg_pool2d(x, x.size()[2:])))
        feat = torch.mul(x, atten)
        x = x + feat
        feat = self.conv(x)
        return feat


