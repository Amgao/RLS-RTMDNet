//#include <THC/THC.h>
//#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include "roi_align_cuda.hpp"
#include <math.h>
#include "cuda/roi_align_kernel.h"
//#include <torch/extension.h>
extern THCState *state;

extern "C" int roi_align_forward_cuda(int aligned_height, int aligned_width, float spatial_scale,
                        at::Tensor& features, at::Tensor& rois, at::Tensor& output)
{
    // Number of ROIs
    int num_rois = rois.size(0);
    int size_rois = rois.size(1);
    if (size_rois != 5)
    {
        return 0;
    }
    // batch size
    //int batch_size = THCudaTensor_size(state, features, 0);
    //if (batch_size != 1)
    //{
    //    return 0;
    //}


    // data height
    int data_height = features.size(2);
    // data width
    int data_width = features.size(3);
    // Number of channels
    int num_channels = features.size(1);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    ROIAlignForwardLaucher(
        features.contiguous().data<float>(), spatial_scale, num_rois, data_height,
        data_width, num_channels, aligned_height,
        aligned_width, rois.contiguous().data<float>(),
        output.contiguous().data<float>(), stream);

    return 1;
}

extern "C" int roi_align_backward_cuda(int aligned_height, int aligned_width, float spatial_scale,
                        at::Tensor& top_grad, at::Tensor& rois, at::Tensor& bottom_grad)
{

    // Number of ROIs
    int num_rois = rois.size(0);
    int size_rois = rois.size(1);
    if (size_rois != 5)
    {
        return 0;
    }

    // batch size
    int batch_size = bottom_grad.size(0);
    //if (batch_size != 1)
    //{
     //   return 0;
    //}
    // data height
    int data_height = bottom_grad.size(2);
    // data width
    int data_width = bottom_grad.size(3);
    // Number of channels
    int num_channels = bottom_grad.size(1);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    ROIAlignBackwardLaucher(
        top_grad.contiguous().data<float>(), spatial_scale, batch_size, num_rois, data_height,
        data_width, num_channels, aligned_height,
        aligned_width, rois.contiguous().data<float>(),
        bottom_grad.contiguous().data<float>(), stream);

    return 1;
}


extern "C" int roi_align_ada_forward_cuda(int aligned_height, int aligned_width, float spatial_scale,
                        at::Tensor& features, at::Tensor& rois, at::Tensor& output)
{
    // Number of ROIs
    int num_rois = rois.size(0);
    int size_rois = rois.size(1);
    if (size_rois != 5)
    {
        return 0;
    }
    // batch size
    //int batch_size = THCudaTensor_size(state, features, 0);
    //if (batch_size != 1)
    //{
    //    return 0;
    //}


    // data height
    int data_height = features.size(2);
    // data width
    int data_width = features.size(3);
    // Number of channels
    int num_channels = features.size(1);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    ROIAlignAdaForwardLaucher(
        features.contiguous().data<float>(), spatial_scale, num_rois, data_height,
        data_width, num_channels, aligned_height,
        aligned_width, rois.contiguous().data<float>(),
        output.contiguous().data<float>(), stream);

    return 1;
}

extern "C" int roi_align_ada_backward_cuda(int aligned_height, int aligned_width, float spatial_scale,
                        at::Tensor& top_grad, at::Tensor& rois, at::Tensor& bottom_grad)
{
    // Number of ROIs
    int num_rois = rois.size(0);
    int size_rois = rois.size(1);
    if (size_rois != 5)
    {
        return 0;
    }

    // batch size
    int batch_size = bottom_grad.size(0);
    //if (batch_size != 1)
    //{
     //   return 0;
    //}
    // data height
    int data_height = bottom_grad.size(2);
    // data width
    int data_width = bottom_grad.size(3);
    // Number of channels
    int num_channels = bottom_grad.size(1);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    ROIAlignAdaBackwardLaucher(
        top_grad.contiguous().data<float>(), spatial_scale, batch_size, num_rois, data_height,
        data_width, num_channels, aligned_height,
        aligned_width, rois.contiguous().data<float>(),
        bottom_grad.contiguous().data<float>(), stream);

    return 1;
}


extern "C" int roi_align_dense_ada_forward_cuda(int aligned_height, int aligned_width, float spatial_scale,
                        at::Tensor& features, at::Tensor& rois, at::Tensor& output)
{
    // Number of ROIs
    int num_rois = rois.size(0);
    int size_rois = rois.size(1);
    if (size_rois != 5)
    {
        return 0;
    }
    // batch size
    //int batch_size = THCudaTensor_size(state, features, 0);
    //if (batch_size != 1)
    //{
    //    return 0;
    //}


    // data height
    int data_height = features.size(2);
    // data width
    int data_width = features.size(3);
    // Number of channels
    int num_channels = features.size(1);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    ROIAlignDenseAdaForwardLaucher(
        features.contiguous().data<float>(), spatial_scale, num_rois, data_height,
        data_width, num_channels, aligned_height,
        aligned_width, rois.contiguous().data<float>(),
        output.contiguous().data<float>(), stream);

    return 1;
}

extern "C" int roi_align_dense_ada_backward_cuda(int aligned_height, int aligned_width, float spatial_scale,
                        at::Tensor& top_grad, at::Tensor& rois, at::Tensor& bottom_grad)
{
    // Grab the input tensor
    /*float * top_grad_flat = THCudaTensor_data(state, top_grad);
    float * rois_flat = THCudaTensor_data(state, rois);

    float * bottom_grad_flat = THCudaTensor_data(state, bottom_grad);

    // Number of ROIs
    int num_rois = THCudaTensor_size(state, rois, 0);
    int size_rois = THCudaTensor_size(state, rois, 1);
    if (size_rois != 5)
    {
        return 0;
    }

    // batch size
    int batch_size = THCudaTensor_size(state, bottom_grad, 0);
    //if (batch_size != 1)
    //{
     //   return 0;
    //}
    // data height
    int data_height = THCudaTensor_size(state, bottom_grad, 2);
    // data width
    int data_width = THCudaTensor_size(state, bottom_grad, 3);
    // Number of channels
    int num_channels = THCudaTensor_size(state, bottom_grad, 1);

    cudaStream_t stream = THCState_getCurrentStream(state);*/
    // Number of ROIs
    int num_rois = rois.size(0);
    int size_rois = rois.size(1);
    if (size_rois != 5)
    {
        return 0;
    }

    // batch size
    int batch_size = bottom_grad.size(0);
    //if (batch_size != 1)
    //{
     //   return 0;
    //}
    // data height
    int data_height = bottom_grad.size(2);
    // data width
    int data_width = bottom_grad.size(3);
    // Number of channels
    int num_channels = bottom_grad.size(1);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    ROIAlignDenseAdaBackwardLaucher(
        top_grad.contiguous().data<float>(), spatial_scale, batch_size, num_rois, data_height,
        data_width, num_channels, aligned_height,
        aligned_width, rois.contiguous().data<float>(),
        bottom_grad.contiguous().data<float>(), stream);

    return 1;
}

/*PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("roi_align_forward_cuda", &roi_align_forward_cuda, "ROIAlign_forward");
  m.def("roi_align_backward_cuda", &roi_align_backward_cuda, "ROIAlign_backward");
  m.def("roi_align_ada_forward_cuda", &roi_align_ada_forward_cuda, "ROIAlign_Ada_forward");
  m.def("roi_align_ada_backward_cuda", &roi_align_ada_backward_cuda, "ROIAlign_Ada_backward");
  m.def("roi_align_dense_ada_forward_cuda", &roi_align_dense_ada_forward_cuda, "ROIAlign_Dense_Ada_forward");
  m.def("roi_align_dense_ada_backward_cuda", &roi_align_dense_ada_backward_cuda, "ROIAlign_Dense_Ada_backward");
}*/
