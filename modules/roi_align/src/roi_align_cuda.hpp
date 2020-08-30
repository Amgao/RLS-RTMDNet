#include <THC/THC.h>
#include <ATen/ATen.h>
#include <torch/extension.h>

//THC_CLASS at::Context& at::globalContext();
//THCState *state = at::globalContext().getTHCState();
//ATen_CLASS at::Context& at::globalContext();
//THCState *state = at::globalContext().thc_state;
//extern THCState *state;

#ifdef __cplusplus
extern "C" {
#endif

int roi_align_forward_cuda(int aligned_height, int aligned_width, float spatial_scale,
                        at::Tensor& features, at::Tensor& rois, at::Tensor& output);

int roi_align_backward_cuda(int aligned_height, int aligned_width, float spatial_scale,
                        at::Tensor& top_grad, at::Tensor& rois, at::Tensor& bottom_grad);


int roi_align_ada_forward_cuda(int aligned_height, int aligned_width, float spatial_scale,
                        at::Tensor& features, at::Tensor& rois, at::Tensor& output);

int roi_align_ada_backward_cuda(int aligned_height, int aligned_width, float spatial_scale,
                        at::Tensor& top_grad, at::Tensor& rois, at::Tensor& bottom_grad);

int roi_align_dense_ada_forward_cuda(int aligned_height, int aligned_width, float spatial_scale,
                        at::Tensor& features, at::Tensor& rois, at::Tensor& output);

int roi_align_dense_ada_backward_cuda(int aligned_height, int aligned_width, float spatial_scale,
                        at::Tensor& top_grad, at::Tensor& rois, at::Tensor& bottom_grad);


#ifdef __cplusplus
}
#endif

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("roi_align_forward_cuda", &roi_align_forward_cuda, "ROIAlign_forward");
  m.def("roi_align_backward_cuda", &roi_align_backward_cuda, "ROIAlign_backward");
  m.def("roi_align_ada_forward_cuda", &roi_align_ada_forward_cuda, "ROIAlign_Ada_forward");
  m.def("roi_align_ada_backward_cuda", &roi_align_ada_backward_cuda, "ROIAlign_Ada_backward");
  m.def("roi_align_dense_ada_forward_cuda", &roi_align_dense_ada_forward_cuda, "ROIAlign_Dense_Ada_forward");
  m.def("roi_align_dense_ada_backward_cuda", &roi_align_dense_ada_backward_cuda, "ROIAlign_Dense_Ada_backward");
}
