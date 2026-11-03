// depth-weighted voxel pooling with the cameras merged inside the kernel.
//
// Upstream (qwen_drive_perception/ops/voxel_pool) writes one fp32 volume per camera (torch::zeros
// [B, N_cam, X, Y, Z, C], 3.93 GB), then the head casts it to bf16, sums the cameras and casts to fp16 for the
// voxel engine. Here one thread per (voxel interval, channel) walks the voxel's points, which are sorted
// camera-minor, and writes the fp16 engine input [B, 1, X, Y, Z, C] once.
//
// faithful = 1 reproduces upstream's rounding: fp32 sum per camera with the same expression and point order
// (psum += feat * depth, compiled with --use_fast_math like upstream), rounded to bf16 per camera (.to(bf16)),
// cameras accumulated in fp32 and rounded to bf16 once (torch's bf16 sum), then bf16 -> fp16 (.to(fp16)).
// faithful = 0: one fp32 sum over every camera's points, one rounding to fp16.
//
// (optional, both bit-exact vs the eager ops they replace):
//   scale [X * Y] fp16: out = fp16(float(fp16(v)) * float(scale[x, y])), = the eager s * cell_norm (warped
//     grid: 1 / fine cells per warped cell; torch's fp16 MulFunctor<float> computes exactly this).
//   voxel_interval [B * X * Y * Z] int32 (-1 = no points): one thread per (voxel, channel) writes every voxel (0 for an
//     empty one), so the caller's zero fill of the 184 MB volume goes away (it was written ~1.7 times).

#include <ATen/ATen.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

template <typename T>
__device__ __forceinline__ float load_as_float(const T* ptr) {
  return static_cast<float>(*ptr);
}

template <>
__device__ __forceinline__ float load_as_float<c10::Half>(const c10::Half* ptr) {
  return __half2float(*reinterpret_cast<const __half*>(ptr));
}

template <>
__device__ __forceinline__ float load_as_float<c10::BFloat16>(const c10::BFloat16* ptr) {
  return __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(ptr));
}

__device__ __forceinline__ float round_bf16(float x) {
  return __bfloat162float(__float2bfloat16_rn(x));
}

template <typename feat_t, typename depth_t>
__device__ __forceinline__ float interval_sum(
    int N_sweep, int N_cam, int D, int H, int W, int feat_channels, int faithful, int start, int length, int channel,
    const feat_t* __restrict__ img_feats, const depth_t* __restrict__ img_depth, const int* __restrict__ sorted_points) {
  float psum = 0.0f;   // current camera's fp32 sum
  float total = 0.0f;  // sum over cameras
  int current_cam = -1;
  for (int i = 0; i < length; ++i) {
    int p = sorted_points[start + i];
    int w = p % W;
    p /= W;
    int h = p % H;
    p /= H;
    int d = p % D;
    p /= D;
    int cam = p % N_cam;
    p /= N_cam;
    int sweep = p % N_sweep;
    int batch = p / N_sweep;
    if (faithful && cam != current_cam) {
      if (current_cam >= 0) total += round_bf16(psum);
      psum = 0.0f;
      current_cam = cam;
    }
    int image_index = (batch * N_sweep + sweep) * N_cam + cam;
    int feat_offset = ((image_index * feat_channels + channel) * H + h) * W + w;
    int depth_offset = ((image_index * D + d) * H + h) * W + w;
    psum += load_as_float(img_feats + feat_offset) * load_as_float(img_depth + depth_offset);
  }
  return faithful ? round_bf16(total + round_bf16(psum)) : psum;
}

__device__ __forceinline__ __half scaled(float v, const __half* __restrict__ scale, long voxel, int Z, int XY) {
  __half h = __float2half_rn(v);
  if (scale == nullptr) return h;
  return __float2half_rn(__half2float(h) * __half2float(scale[(voxel / Z) % XY]));
}

// dense variant: one thread per (voxel, channel); voxel_interval maps a voxel to its interval or -1.
template <typename feat_t, typename depth_t>
__global__ void merged_pool_dense_kernel(
    int N_sweep, int N_cam, int D, int H, int W, int feat_channels, long n_voxels, int faithful, int Z, int XY,
    const feat_t* __restrict__ img_feats, const depth_t* __restrict__ img_depth,
    const int* __restrict__ sorted_points, const int* __restrict__ interval_starts,
    const int* __restrict__ interval_lengths, const int* __restrict__ voxel_interval,
    const __half* __restrict__ scale, __half* __restrict__ out) {
  long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
  long voxel = idx / feat_channels;
  int channel = idx % feat_channels;
  if (voxel >= n_voxels) return;
  int iv = voxel_interval[voxel];
  float v = iv < 0 ? 0.0f : interval_sum(N_sweep, N_cam, D, H, W, feat_channels, faithful, interval_starts[iv],
                                         interval_lengths[iv], channel, img_feats, img_depth, sorted_points);
  out[idx] = scaled(v, scale, voxel, Z, XY);
}

template <typename feat_t, typename depth_t>
__global__ void merged_pool_kernel(
    int N_sweep, int N_cam, int D, int H, int W, int feat_channels, int n_intervals, int faithful,
    const feat_t* __restrict__ img_feats,
    const depth_t* __restrict__ img_depth,
    const int* __restrict__ sorted_points,     // point index (upstream flat frustum index) per sorted point
    const int* __restrict__ interval_starts,
    const int* __restrict__ interval_lengths,
    const int* __restrict__ interval_voxels,   // ((b * X + x) * Y + y) * Z + z per interval
    const __half* __restrict__ scale, int Z, int XY,
    __half* __restrict__ out) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int interval_index = idx / feat_channels;
  int channel = idx % feat_channels;
  if (interval_index >= n_intervals) return;

  float v = interval_sum(N_sweep, N_cam, D, H, W, feat_channels, faithful, interval_starts[interval_index],
                         interval_lengths[interval_index], channel, img_feats, img_depth, sorted_points);
  long voxel = interval_voxels[interval_index];
  out[voxel * feat_channels + channel] = scaled(v, scale, voxel, Z, XY);
}

void merged_pool_cuda(const at::Tensor& img_feats, const at::Tensor& img_depth, const at::Tensor& sorted_points,
                      const at::Tensor& interval_starts, const at::Tensor& interval_lengths,
                      const at::Tensor& interval_voxels, at::Tensor& out, int N_sweep, int N_cam, int D, int H, int W,
                      int faithful, const at::Tensor& scale, const at::Tensor& voxel_interval, int Z, int XY) {
  int feat_channels = img_feats.size(2);
  int n_intervals = interval_starts.numel();
  bool dense = voxel_interval.numel() > 0;
  if (n_intervals == 0 && !dense) return;
  const int threads = 256;
  long n_voxels = voxel_interval.numel();
  long total = dense ? n_voxels * feat_channels : (long)n_intervals * feat_channels;
  int blocks = (int)((total + threads - 1) / threads);
  auto stream = c10::cuda::getCurrentCUDAStream();
  const __half* sc = scale.numel() ? reinterpret_cast<const __half*>(scale.data_ptr<at::Half>()) : nullptr;
  __half* o = reinterpret_cast<__half*>(out.data_ptr<at::Half>());
#define LAUNCH(FT, DT)                                                                                          \
  if (dense)                                                                                                    \
    merged_pool_dense_kernel<FT, DT><<<blocks, threads, 0, stream>>>(                                           \
        N_sweep, N_cam, D, H, W, feat_channels, n_voxels, faithful, Z, XY, img_feats.data_ptr<FT>(),            \
        img_depth.data_ptr<DT>(), sorted_points.data_ptr<int>(), interval_starts.data_ptr<int>(),               \
        interval_lengths.data_ptr<int>(), voxel_interval.data_ptr<int>(), sc, o);                               \
  else                                                                                                          \
    merged_pool_kernel<FT, DT><<<blocks, threads, 0, stream>>>(                                                 \
        N_sweep, N_cam, D, H, W, feat_channels, n_intervals, faithful, img_feats.data_ptr<FT>(),                \
        img_depth.data_ptr<DT>(), sorted_points.data_ptr<int>(), interval_starts.data_ptr<int>(),               \
        interval_lengths.data_ptr<int>(), interval_voxels.data_ptr<int>(), sc, Z, XY, o)
  auto ft = img_feats.scalar_type(), dt = img_depth.scalar_type();
  if (ft == at::kBFloat16 && dt == at::kBFloat16) { LAUNCH(c10::BFloat16, c10::BFloat16); }
  else if (ft == at::kBFloat16 && dt == at::kFloat) { LAUNCH(c10::BFloat16, float); }
  else if (ft == at::kFloat && dt == at::kFloat) { LAUNCH(float, float); }
  else if (ft == at::kHalf && dt == at::kHalf) { LAUNCH(c10::Half, c10::Half); }
  else TORCH_CHECK(false, "merged_pool: unsupported dtypes ", ft, " / ", dt);
#undef LAUNCH
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
