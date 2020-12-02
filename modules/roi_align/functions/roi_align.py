import torch
from torch.autograd import Function
from .._ext2 import roi_align


# TODO use save_for_backward instead
class RoIAlignFunction(Function):
    @staticmethod
    def forward(ctx, features, rois, aligned_height, aligned_width, spatial_scale):
        aligned_width = int(aligned_width)
        aligned_height = int(aligned_height)
        spatial_scale = float(spatial_scale)
        feature_size = features.size()

        batch_size, num_channels, data_height, data_width = features.size()
        num_rois = rois.size(0)

        output = features.new(num_rois, num_channels, aligned_height, aligned_width).zero_()
        #help(roi_align.roi_align_forward_cuda)
        #print(features.type())
        #print(rois.type())
        #print(output.type())
        if features.is_cuda:
            success = roi_align.roi_align_forward_cuda(aligned_height,
                                             aligned_width,
                                             spatial_scale, features,
                                             rois, output)
        else:
            raise NotImplementedError

        ctx.save_for_backward(rois, aligned_height, aligned_width, spatial_scale, feature_size)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        rois, aligned_height, aligned_width, spatial_scale, feature_size = ctx.saved_tensors
        assert(feature_size is not None and grad_output.is_cuda)

        batch_size, num_channels, data_height, data_width = feature_size

        grad_input = rois.new(batch_size, num_channels, data_height,
                                  data_width).zero_()
        roi_align.roi_align_backward_cuda(aligned_height,
                                          aligned_width,
                                          spatial_scale, grad_output,
                                          rois, grad_input)

        # print grad_input

        return grad_input, None


# TODO use save_for_backward instead
class RoIAlignAdaFunction(Function):
    @staticmethod
    def forward(ctx, features, rois, aligned_height, aligned_width, spatial_scale):
        aligned_width = int(aligned_width)
        aligned_height = int(aligned_height)
        spatial_scale = float(spatial_scale)
        feature_size = features.size()

        batch_size, num_channels, data_height, data_width = features.size()
        num_rois = rois.size(0)

        output = features.new(num_rois, num_channels, aligned_height, aligned_width).zero_()
        if features.is_cuda:
            success = roi_align.roi_align_ada_forward_cuda(aligned_height,
                                             aligned_width,
                                             spatial_scale, features,
                                             rois, output)
        else:
            raise NotImplementedError

        ctx.save_for_backward(rois, aligned_height, aligned_width, spatial_scale, feature_size)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        rois, aligned_height, aligned_width, spatial_scale, feature_size = ctx.saved_tensors
        assert(feature_size is not None and grad_output.is_cuda)

        batch_size, num_channels, data_height, data_width = feature_size

        grad_input = rois.new(batch_size, num_channels, data_height,
                                  data_width).zero_()
        roi_align.roi_align_ada_backward_cuda(aligned_height,
                                          aligned_width,
                                          spatial_scale, grad_output,
                                          rois, grad_input)

        # print grad_input

        return grad_input, None


# TODO use save_for_backward instead
class RoIAlignDenseAdaFunction(Function):
    @staticmethod
    def forward(ctx, features, rois, aligned_height, aligned_width, spatial_scale):
        aligned_width = int(aligned_width)
        aligned_height = int(aligned_height)
        spatial_scale = float(spatial_scale)
        feature_size = features.size()

        batch_size, num_channels, data_height, data_width = features.size()
        num_rois = rois.size(0)

        output = features.new(num_rois, num_channels, aligned_height, aligned_width).zero_()
        if features.is_cuda:
            success = roi_align.roi_align_dense_ada_forward_cuda(aligned_height,
                                             aligned_width,
                                             spatial_scale, features,
                                             rois, output)
        else:
            raise NotImplementedError

        ctx.save_for_backward(rois, aligned_height, aligned_width, spatial_scale, feature_size)

        return output

    @staticmethod
    def backward(self, grad_output):
        rois, aligned_height, aligned_width, spatial_scale, feature_size = ctx.saved_tensors
        assert(feature_size is not None and grad_output.is_cuda)

        batch_size, num_channels, data_height, data_width = feature_size

        grad_input = rois.new(batch_size, num_channels, data_height,
                                  data_width).zero_()
        roi_align.roi_align_dense_ada_backward_cuda(aligned_height,
                                          aligned_width,
                                          spatial_scale, grad_output,
                                          rois, grad_input)

        # print grad_input

        return grad_input, None
