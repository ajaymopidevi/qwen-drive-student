// binding for merged_pool_cuda.cu.
#include <torch/extension.h>

void merged_pool_cuda(const at::Tensor& img_feats, const at::Tensor& img_depth, const at::Tensor& sorted_points,
                      const at::Tensor& interval_starts, const at::Tensor& interval_lengths,
                      const at::Tensor& interval_voxels, at::Tensor& out, int N_sweep, int N_cam, int D, int H, int W,
                      int faithful, const at::Tensor& scale, const at::Tensor& voxel_interval, int Z, int XY);

void merged_pool(const at::Tensor& img_feats, const at::Tensor& img_depth, const at::Tensor& sorted_points,
                 const at::Tensor& interval_starts, const at::Tensor& interval_lengths,
                 const at::Tensor& interval_voxels, at::Tensor out, int N_sweep, int N_cam, int D, int H, int W,
                 int faithful, const at::Tensor& scale, const at::Tensor& voxel_interval, int Z, int XY) {
  TORCH_CHECK(img_feats.is_cuda() && img_feats.is_contiguous() && img_depth.is_contiguous(), "inputs: contiguous CUDA");
  TORCH_CHECK(out.scalar_type() == at::kHalf && out.is_contiguous(), "out: contiguous fp16");
  for (const auto* t : {&sorted_points, &interval_starts, &interval_lengths, &interval_voxels})
    TORCH_CHECK(t->scalar_type() == at::kInt && t->is_contiguous(), "index tensors: contiguous int32");
  TORCH_CHECK(scale.numel() == 0 || (scale.scalar_type() == at::kHalf && scale.is_contiguous() && scale.numel() == XY),
              "scale: contiguous fp16 [X * Y] or empty");
  TORCH_CHECK(voxel_interval.numel() == 0 || (voxel_interval.scalar_type() == at::kInt && voxel_interval.is_contiguous() &&
              voxel_interval.numel() * img_feats.size(2) == out.numel()), "voxel_interval: contiguous int32 [B X Y Z] or empty");
  merged_pool_cuda(img_feats, img_depth, sorted_points, interval_starts, interval_lengths, interval_voxels, out,
                   N_sweep, N_cam, D, H, W, faithful, scale, voxel_interval, Z, XY);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("merged_pool", &merged_pool, "depth-weighted voxel pooling, cameras merged, fp16 [B, 1, X, Y, Z, C] out");
}
