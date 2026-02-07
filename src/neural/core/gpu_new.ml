open Metal
open Ctypes

(* ============================================================================
   Metal Shader Source - Optimized kernels inspired by PyTorch MPS backend
   ============================================================================ *)

let shader_source = {|
#include <metal_stdlib>
using namespace metal;

// Packed parameter structure for reduced overhead
struct MatMulParams {
    int M;
    int N;
    int K;
};

struct ConvParams {
    int N;
    int InH;
    int InW;
    int OutH;
    int OutW;
    int KH;
    int KW;
    int InD;
    int OutD;
    int act;
};

struct LinearParams {
    int M;
    int N;
    int K;
    int act;
};

struct PoolParams {
    int N;
    int C;
    int InH;
    int InW;
    int OutH;
    int OutW;
    int KH;
    int KW;
    int stride;
};

// ============================================================================
// Core Matrix Operations
// ============================================================================

kernel void matmul(
    device const float *A [[buffer(0)]],
    device const float *B [[buffer(1)]],
    device float *C [[buffer(2)]],
    constant MatMulParams &p [[buffer(3)]],
    uint2 id [[thread_position_in_grid]])
{
    if (id.x >= uint(p.N) || id.y >= uint(p.M)) return;
    
    float sum = 0.0;
    int a_row = id.y * p.K;
    for (int k = 0; k < p.K; k++) {
        sum += A[a_row + k] * B[k * p.N + id.x];
    }
    C[id.y * p.N + id.x] = sum;
}

kernel void mat_add(
    device const float *A [[buffer(0)]],
    device const float *B [[buffer(1)]],
    device float *C [[buffer(2)]],
    constant uint &len [[buffer(3)]],
    uint id [[thread_position_in_grid]])
{
    if (id >= len) return;
    C[id] = A[id] + B[id];
}

kernel void mat_mul_el(
    device const float *A [[buffer(0)]],
    device const float *B [[buffer(1)]],
    device float *C [[buffer(2)]],
    constant uint &len [[buffer(3)]],
    uint id [[thread_position_in_grid]])
{
    if (id >= len) return;
    C[id] = A[id] * B[id];
}

kernel void mat_transpose(
    device const float *A [[buffer(0)]],
    device float *B [[buffer(1)]],
    constant int &rows [[buffer(2)]],
    constant int &cols [[buffer(3)]],
    uint2 id [[thread_position_in_grid]])
{
    if (id.x >= uint(cols) || id.y >= uint(rows)) return;
    B[id.x * rows + id.y] = A[id.y * cols + id.x];
}

// ============================================================================
// Fused Operations for Performance
// ============================================================================

// Fused bias addition and activation
kernel void add_bias_and_activate(
    device float *X [[buffer(0)]],
    device const float *bias [[buffer(1)]],
    constant int &cols [[buffer(2)]],
    constant int &act [[buffer(3)]],
    uint2 id [[thread_position_in_grid]])
{
    if (id.x >= uint(cols)) return;
    
    float val = X[id.y * cols + id.x] + bias[id.x];
    
    // Apply activation
    if (act == 1) val = max(0.0f, val);           // ReLU
    else if (act == 2) val = tanh(val);           // Tanh
    else if (act == 3) val = 1.0f/(1.0f+exp(-val)); // Sigmoid
    
    X[id.y * cols + id.x] = val;
}

kernel void add_bias(
    device float *X [[buffer(0)]],
    device const float *bias [[buffer(1)]],
    constant int &cols [[buffer(2)]],
    uint2 id [[thread_position_in_grid]])
{
    if (id.x >= uint(cols)) return;
    X[id.y * cols + id.x] += bias[id.x];
}

// ============================================================================
// Activation Functions - Forward Pass
// ============================================================================

kernel void relu_fwd(
    device float *X [[buffer(0)]],
    constant uint &len [[buffer(1)]],
    uint id [[thread_position_in_grid]])
{
    if (id >= len) return;
    X[id] = max(0.0f, X[id]);
}

kernel void sigmoid_fwd(
    device float *X [[buffer(0)]],
    constant uint &len [[buffer(1)]],
    uint id [[thread_position_in_grid]])
{
    if (id >= len) return;
    X[id] = 1.0f / (1.0f + exp(-X[id]));
}

kernel void tanh_fwd(
    device float *X [[buffer(0)]],
    constant uint &len [[buffer(1)]],
    uint id [[thread_position_in_grid]])
{
    if (id >= len) return;
    X[id] = tanh(X[id]);
}

// ============================================================================
// Activation Functions - Backward Pass
// ============================================================================

kernel void relu_bwd(
    device const float *Z [[buffer(0)]],
    device const float *upstream [[buffer(1)]],
    device float *out [[buffer(2)]],
    constant uint &len [[buffer(3)]],
    uint id [[thread_position_in_grid]])
{
    if (id >= len) return;
    out[id] = Z[id] > 0 ? upstream[id] : 0.0f;
}

kernel void sigmoid_bwd(
    device const float *Z [[buffer(0)]],
    device const float *upstream [[buffer(1)]],
    device float *out [[buffer(2)]],
    constant uint &len [[buffer(3)]],
    uint id [[thread_position_in_grid]])
{
    if (id >= len) return;
    float s = 1.0f / (1.0f + exp(-Z[id]));
    out[id] = upstream[id] * s * (1.0f - s);
}

kernel void tanh_bwd(
    device const float *Z [[buffer(0)]],
    device const float *upstream [[buffer(1)]],
    device float *out [[buffer(2)]],
    constant uint &len [[buffer(3)]],
    uint id [[thread_position_in_grid]])
{
    if (id >= len) return;
    float t = tanh(Z[id]);
    out[id] = upstream[id] * (1.0f - t * t);
}

// ============================================================================
// Linear Layer Operations
// ============================================================================

kernel void linear_fwd(
    device const float *in [[buffer(0)]],
    device const float *weights [[buffer(1)]],
    device const float *bias [[buffer(2)]],
    device float *out [[buffer(3)]],
    constant LinearParams &p [[buffer(4)]],
    uint2 id [[thread_position_in_grid]])
{
    if (id.x >= uint(p.N) || id.y >= uint(p.M)) return;
    
    float sum = bias[id.x];
    for (int k = 0; k < p.K; k++) {
        sum += in[id.y * p.K + k] * weights[id.x * p.K + k];
    }
    
    // Apply activation
    if (p.act == 1) sum = max(0.0f, sum);
    else if (p.act == 2) sum = tanh(sum);
    else if (p.act == 3) sum = 1.0f / (1.0f + exp(-sum));
    
    out[id.y * p.N + id.x] = sum;
}

kernel void linear_bwd_weights(
    device const float *upstream [[buffer(0)]],
    device const float *in_cache [[buffer(1)]],
    device const float *out_cache [[buffer(2)]],
    device float *grad_w [[buffer(3)]],
    device float *grad_b [[buffer(4)]],
    constant LinearParams &p [[buffer(5)]],
    uint2 id [[thread_position_in_grid]])
{
    if (id.x >= uint(p.K) || id.y >= uint(p.N)) return;
    
    float dw = 0.0f;
    float db = 0.0f;
    
    for (int b = 0; b < p.M; b++) {
        float dy = upstream[b * p.N + id.y];
        float y = out_cache[b * p.N + id.y];
        
        // Activation gradient
        if (p.act == 1) {
            dy = y > 0 ? dy : 0;
        } else if (p.act == 2) {
            dy = dy * (1.0f - y * y);
        } else if (p.act == 3) {
            dy = dy * y * (1.0f - y);
        }
        
        dw += dy * in_cache[b * p.K + id.x];
        if (id.x == 0) db += dy;
    }
    
    grad_w[id.y * p.K + id.x] += dw;
    if (id.x == 0) grad_b[id.y] += db;
}

kernel void linear_bwd_input(
    device const float *upstream [[buffer(0)]],
    device const float *weights [[buffer(1)]],
    device const float *out_cache [[buffer(2)]],
    device float *grad_in [[buffer(3)]],
    constant LinearParams &p [[buffer(4)]],
    uint2 id [[thread_position_in_grid]])
{
    if (id.x >= uint(p.K) || id.y >= uint(p.M)) return;
    
    float sum = 0.0f;
    for (int n = 0; n < p.N; n++) {
        float dy = upstream[id.y * p.N + n];
        float y = out_cache[id.y * p.N + n];
        
        // Activation gradient
        if (p.act == 1) {
            dy = y > 0 ? dy : 0;
        } else if (p.act == 2) {
            dy = dy * (1.0f - y * y);
        } else if (p.act == 3) {
            dy = dy * y * (1.0f - y);
        }
        
        sum += dy * weights[n * p.K + id.x];
    }
    grad_in[id.y * p.K + id.x] = sum;
}

// ============================================================================
// Convolution Operations
// ============================================================================

kernel void conv2d_direct_fwd(
    device const float *input [[buffer(0)]],
    device const float *weights [[buffer(1)]],
    device const float *bias [[buffer(2)]],
    device float *output [[buffer(3)]],
    constant ConvParams &p [[buffer(4)]],
    uint3 id [[thread_position_in_grid]])
{
    int w = id.x % p.OutW;
    int h = id.x / p.OutW;
    int oc = id.y;
    int b = id.z;

    if (w >= p.OutW || h >= p.OutH || oc >= p.OutD || b >= p.N) return;
    
    float sum = bias[oc];
    int in_batch_offset = b * (p.InD * p.InH * p.InW);
    int weight_oc_offset = oc * (p.InD * p.KH * p.KW);
    
    for (int ic = 0; ic < p.InD; ic++) {
        int in_channel_offset = in_batch_offset + ic * (p.InH * p.InW);
        int weight_ic_offset = weight_oc_offset + ic * (p.KH * p.KW);
        
        for (int i = 0; i < p.KH; i++) {
            for (int j = 0; j < p.KW; j++) {
                sum += input[in_channel_offset + (h + i) * p.InW + (w + j)] * 
                       weights[weight_ic_offset + i * p.KW + j];
            }
        }
    }
    
    // Apply activation
    if (p.act == 1) sum = max(0.0f, sum);
    else if (p.act == 2) sum = tanh(sum);
    else if (p.act == 3) sum = 1.0f / (1.0f + exp(-sum));
    
    output[b * (p.OutD * p.OutH * p.OutW) + oc * (p.OutH * p.OutW) + h * p.OutW + w] = sum;
}

kernel void conv2d_direct_bwd_input(
    device const float *grad_out [[buffer(0)]],
    device const float *weights [[buffer(1)]],
    device float *grad_in [[buffer(2)]],
    constant ConvParams &p [[buffer(3)]],
    uint3 id [[thread_position_in_grid]])
{
    int w = id.x % p.InW;
    int h = id.x / p.InW;
    int ic = id.y;
    int b = id.z;

    if (w >= p.InW || h >= p.InH || ic >= p.InD || b >= p.N) return;
    
    float sum = 0.0f;
    int grad_out_batch = b * (p.OutD * p.OutH * p.OutW);
    
    for (int oc = 0; oc < p.OutD; oc++) {
        int grad_out_ch = grad_out_batch + oc * (p.OutH * p.OutW);
        int w_offset = oc * (p.InD * p.KH * p.KW) + ic * (p.KH * p.KW);
        
        for (int i = 0; i < p.KH; i++) {
            for (int j = 0; j < p.KW; j++) {
                int out_y = h - i;
                int out_x = w - j;
                
                if (out_y >= 0 && out_y < p.OutH && out_x >= 0 && out_x < p.OutW) {
                    sum += grad_out[grad_out_ch + out_y * p.OutW + out_x] * 
                           weights[w_offset + i * p.KW + j];
                }
            }
        }
    }
    grad_in[b * (p.InD * p.InH * p.InW) + ic * (p.InH * p.InW) + h * p.InW + w] = sum;
}

kernel void conv2d_direct_bwd_weights(
    device const float *input [[buffer(0)]],
    device const float *grad_out [[buffer(1)]],
    device float *grad_w [[buffer(2)]],
    constant ConvParams &p [[buffer(3)]],
    uint3 id [[thread_position_in_grid]])
{
    int k_idx = id.x;
    int ki = k_idx / p.KW;
    int kj = k_idx % p.KW;
    int ic = id.y;
    int oc = id.z;
    
    if (ki >= p.KH || ic >= p.InD || oc >= p.OutD) return;
    
    float sum = 0.0f;
    for (int b = 0; b < p.N; b++) {
        int in_base = b * (p.InD * p.InH * p.InW) + ic * (p.InH * p.InW);
        int grad_base = b * (p.OutD * p.OutH * p.OutW) + oc * (p.OutH * p.OutW);
        
        for (int r = 0; r < p.OutH; r++) {
            for (int c = 0; c < p.OutW; c++) {
                sum += input[in_base + (r + ki) * p.InW + (c + kj)] * 
                       grad_out[grad_base + r * p.OutW + c];
            }
        }
    }
    grad_w[oc * (p.InD * p.KH * p.KW) + ic * (p.KH * p.KW) + k_idx] += sum / float(p.N);
}

kernel void conv2d_bias_bwd(
    device const float *grad_out [[buffer(0)]],
    device float *grad_b [[buffer(1)]],
    constant ConvParams &p [[buffer(2)]],
    uint oc [[thread_position_in_grid]])
{
    if (oc >= uint(p.OutD)) return;
    
    float sum = 0.0f;
    int spatial_size = p.OutH * p.OutW;
    
    for (int b = 0; b < p.N; b++) {
        int base = b * p.OutD * spatial_size + oc * spatial_size;
        for (int i = 0; i < spatial_size; i++) {
            sum += grad_out[base + i];
        }
    }
    grad_b[oc] += sum / float(p.N);
}

// ============================================================================
// Im2Col/Col2Im for Convolution
// ============================================================================

kernel void im2col(
    device const float *img [[buffer(0)]],
    device float *cols [[buffer(1)]],
    constant ConvParams &p [[buffer(2)]],
    uint id [[thread_position_in_grid]])
{
    int total = p.N * p.OutH * p.OutW * p.InD * p.KH * p.KW;
    if (id >= uint(total)) return;
    
    int kj = id % p.KW;
    int ki = (id / p.KW) % p.KH;
    int c = (id / (p.KW * p.KH)) % p.InD;
    int s = (id / (p.KW * p.KH * p.InD)) % (p.OutH * p.OutW);
    int b = id / (p.KW * p.KH * p.InD * p.OutH * p.OutW);
    
    int out_y = s / p.OutW;
    int out_x = s % p.OutW;
    int in_y = out_y + ki;
    int in_x = out_x + kj;
    
    cols[id] = img[b * (p.InD * p.InH * p.InW) + c * (p.InH * p.InW) + in_y * p.InW + in_x];
}

kernel void col2im_optimized(
    device const float *cols [[buffer(0)]],
    device float *img [[buffer(1)]],
    constant ConvParams &p [[buffer(2)]],
    uint id [[thread_position_in_grid]])
{
    int total = p.N * p.InD * p.InH * p.InW;
    if (id >= uint(total)) return;
    
    int px = id % p.InW;
    int py = (id / p.InW) % p.InH;
    int c = (id / (p.InW * p.InH)) % p.InD;
    int b = id / (p.InW * p.InH * p.InD);
    
    float sum = 0.0f;
    int col_width = p.InD * p.KH * p.KW;
    
    for (int ki = 0; ki < p.KH; ki++) {
        for (int kj = 0; kj < p.KW; kj++) {
            int out_y = py - ki;
            int out_x = px - kj;
            
            if (out_y >= 0 && out_y < p.OutH && out_x >= 0 && out_x < p.OutW) {
                int s = out_y * p.OutW + out_x;
                int col_idx = (((b * p.OutH * p.OutW + s) * p.InD + c) * p.KH + ki) * p.KW + kj;
                sum += cols[col_idx];
            }
        }
    }
    img[id] = sum;
}

// ============================================================================
// Permutation Operations
// ============================================================================

kernel void permute_nhwc_nchw(
    device const float *in [[buffer(0)]],
    device float *out [[buffer(1)]],
    constant int4 &dims [[buffer(2)]],  // N, H, W, C
    uint3 id [[thread_position_in_grid]])
{
    int w = id.x;
    int h = id.y;
    int bc = id.z;
    
    if (w >= dims.z || h >= dims.y || bc >= dims.x * dims.w) return;
    
    int c = bc % dims.w;
    int b = bc / dims.w;
    
    int nchw_idx = b * (dims.w * dims.y * dims.z) + c * (dims.y * dims.z) + h * dims.z + w;
    int nhwc_idx = b * (dims.y * dims.z * dims.w) + h * (dims.z * dims.w) + w * dims.w + c;
    
    out[nchw_idx] = in[nhwc_idx];
}

kernel void permute_nchw_nhwc(
    device const float *in [[buffer(0)]],
    device float *out [[buffer(1)]],
    constant int4 &dims [[buffer(2)]],  // N, H, W, C
    uint3 id [[thread_position_in_grid]])
{
    int w = id.x;
    int h = id.y;
    int bc = id.z;
    
    if (w >= dims.z || h >= dims.y || bc >= dims.x * dims.w) return;
    
    int c = bc % dims.w;
    int b = bc / dims.w;
    
    int nchw_idx = b * (dims.w * dims.y * dims.z) + c * (dims.y * dims.z) + h * dims.z + w;
    int nhwc_idx = b * (dims.y * dims.z * dims.w) + h * (dims.z * dims.w) + w * dims.w + c;
    
    out[nhwc_idx] = in[nchw_idx];
}

// ============================================================================
// Pooling Operations
// ============================================================================

kernel void maxpool_fwd(
    device const float *in [[buffer(0)]],
    device float *out [[buffer(1)]],
    device int *indices [[buffer(2)]],
    constant PoolParams &p [[buffer(3)]],
    uint3 id [[thread_position_in_grid]])
{
    int w = id.x;
    int h = id.y;
    int bc = id.z;
    
    if (w >= p.OutW || h >= p.OutH || bc >= p.N * p.C) return;
    
    int b = bc / p.C;
    int c = bc % p.C;
    int in_base = b * (p.C * p.InH * p.InW) + c * (p.InH * p.InW);
    
    float max_val = -1e38f;
    int max_idx = -1;
    
    for (int i = 0; i < p.KH; i++) {
        for (int j = 0; j < p.KW; j++) {
            int cur_h = h * p.stride + i;
            int cur_w = w * p.stride + j;
            
            if (cur_h < p.InH && cur_w < p.InW) {
                int idx = in_base + cur_h * p.InW + cur_w;
                float v = in[idx];
                if (v > max_val) {
                    max_val = v;
                    max_idx = idx;
                }
            }
        }
    }
    
    int out_idx = b * (p.C * p.OutH * p.OutW) + c * (p.OutH * p.OutW) + h * p.OutW + w;
    out[out_idx] = max_val;
    indices[out_idx] = max_idx;
}

kernel void maxpool_bwd(
    device const float *grad_out [[buffer(0)]],
    device const int *indices [[buffer(1)]],
    device float *grad_in [[buffer(2)]],
    constant uint &total_out [[buffer(3)]],
    uint id [[thread_position_in_grid]])
{
    if (id >= total_out) return;
    
    int in_idx = indices[id];
    // Note: atomic add for thread-safety with overlapping pools
    atomic_fetch_add_explicit((device atomic_float*)&grad_in[in_idx], grad_out[id], memory_order_relaxed);
}

// ============================================================================
// Optimizer Operations
// ============================================================================

kernel void adam_step(
    device float *W [[buffer(0)]],
    device const float *G [[buffer(1)]],
    device float *M_t [[buffer(2)]],
    device float *V_t [[buffer(3)]],
    constant float &lr [[buffer(4)]],
    constant float &b1 [[buffer(5)]],
    constant float &b2 [[buffer(6)]],
    constant float &b1p [[buffer(7)]],
    constant float &b2p [[buffer(8)]],
    constant float &eps [[buffer(9)]],
    constant float &wd [[buffer(10)]],
    constant uint &len [[buffer(11)]],
    uint id [[thread_position_in_grid]])
{
    if (id >= len) return;
    
    float g = G[id];
    float m = b1 * M_t[id] + (1.0f - b1) * g;
    float v = b2 * V_t[id] + (1.0f - b2) * g * g;
    
    M_t[id] = m;
    V_t[id] = v;
    
    W[id] = W[id] - lr * wd * W[id] - lr * (m / (1.0f - b1p)) / (sqrt(v / (1.0f - b2p)) + eps);
}

// ============================================================================
// Utility Operations
// ============================================================================

kernel void zero_buf(
    device float *X [[buffer(0)]],
    constant uint &len [[buffer(1)]],
    uint id [[thread_position_in_grid]])
{
    if (id >= len) return;
    X[id] = 0.0f;
}

kernel void mse_grad(
    device const float *p [[buffer(0)]],
    device const float *t [[buffer(1)]],
    device float *g [[buffer(2)]],
    constant float &scale [[buffer(3)]],
    constant uint &len [[buffer(4)]],
    uint id [[thread_position_in_grid]])
{
    if (id >= len) return;
    g[id] = scale * (p[id] - t[id]);
}

// ============================================================================
// Metrics Operations
// ============================================================================

kernel void confusion_matrix_update(
    device const float *preds [[buffer(0)]],
    device const float *targets [[buffer(1)]],
    device atomic_uint *cm [[buffer(2)]],
    constant int &N [[buffer(3)]],
    constant int &C [[buffer(4)]],
    uint id [[thread_position_in_grid]])
{
    if (id >= uint(N)) return;
    
    // Find argmax for pred
    int p_idx = 0;
    float max_p = preds[id * C];
    for (int i=1; i<C; i++) {
        float val = preds[id * C + i];
        if (val > max_p) { max_p = val; p_idx = i; }
    }
    
    // Find argmax for target
    int t_idx = 0;
    float max_t = targets[id * C];
    for (int i=1; i<C; i++) {
        float val = targets[id * C + i];
        if (val > max_t) { max_t = val; t_idx = i; }
    }
    
    // Atomic add
    atomic_fetch_add_explicit(&cm[t_idx * C + p_idx], 1, memory_order_relaxed);
}

kernel void cm_to_float(
    device const atomic_uint *cm_in [[buffer(0)]],
    device float *cm_out [[buffer(1)]],
    constant uint &len [[buffer(2)]],
    uint id [[thread_position_in_grid]])
{
    if (id >= len) return;
    cm_out[id] = float(atomic_load_explicit(&cm_in[id], memory_order_relaxed));
}
|}

(* ============================================================================
   Context and Buffer Management - PyTorch-inspired architecture
   ============================================================================ *)

(* Buffer size classes for power-of-2 aligned allocation *)
let size_class size =
  let rec find_class s = if s <= 0 then 4 else if s <= 64 then 64 
    else if s land (s - 1) = 0 then s  (* Already power of 2 *)
    else find_class ((s lsr 1) + 1 lsl 1)  (* Next power of 2 *)
  in find_class size

type buffer_pool = {
  mutable pools : (int, Buffer.t Queue.t) Hashtbl.t;
  mutable allocated_bytes : int;
  max_cached_bytes : int;
  max_pool_size : int;
}

type context = { 
  device : Device.t; 
  queue : CommandQueue.t; 
  pipelines : (string, ComputePipelineState.t) Hashtbl.t;
  mutable commands_in_buffer : int;
  buffer_pool : buffer_pool;
  mutable pending_buffers : (Buffer.t * int) list;  (* buffer * size *)
  mutable active_cb : CommandBuffer.t option;
}

type tensor = { 
  buffer : Buffer.t; 
  rows : int; 
  cols : int; 
  mutable released : bool 
}

let ctx_ref = ref None

(* Configuration constants *)
let max_commands_per_buffer = 100
let max_pool_size_per_class = 20
let max_cached_bytes = 256 * 1024 * 1024  (* 256 MB *)

(* ============================================================================
   Buffer Pool Management with Size Classes
   ============================================================================ *)

let create_buffer_pool () = {
  pools = Hashtbl.create 16;
  allocated_bytes = 0;
  max_cached_bytes;
  max_pool_size = max_pool_size_per_class;
}

let get_buffer_from_pool pool ctx len =
  let size_cls = size_class len in
  match Hashtbl.find_opt pool.pools size_cls with
  | Some q when not (Queue.is_empty q) -> 
      Queue.pop q
  | _ -> 
      Buffer.on_device ctx.device ~length:size_cls ResourceOptions.storage_mode_shared

let return_buffer_to_pool pool buffer size =
  let size_cls = size_class size in
  let q = match Hashtbl.find_opt pool.pools size_cls with
    | Some q -> q
    | None -> 
        let q = Queue.create () in 
        Hashtbl.add pool.pools size_cls q; 
        q
  in
  if Queue.length q < pool.max_pool_size && 
     pool.allocated_bytes < pool.max_cached_bytes then begin
    Queue.push buffer q;
    pool.allocated_bytes <- pool.allocated_bytes + size_cls
  end

let flush_pending_buffers ctx =
  List.iter (fun (b, size) -> 
    return_buffer_to_pool ctx.buffer_pool b size
  ) ctx.pending_buffers;
  ctx.pending_buffers <- []

(* ============================================================================
   Context Management
   ============================================================================ *)

let sync () = 
  match !ctx_ref with
  | Some ctx ->
      (match ctx.active_cb with 
       | Some cb -> 
           CommandBuffer.commit cb; 
           CommandBuffer.wait_until_completed cb; 
           ctx.active_cb <- None;
           ctx.commands_in_buffer <- 0;
           flush_pending_buffers ctx
       | None -> ())
  | None -> ()

let get_ctx () =
  match !ctx_ref with 
  | Some ctx -> ctx 
  | None ->
      let device = Device.create_system_default () in
      let queue = CommandQueue.on_device device in
      let library = Library.on_device device ~source:shader_source (CompileOptions.init ()) in
      let pipelines = Hashtbl.create 32 in
      
      (* Compile all kernels *)
      let kernel_names = [
        "matmul"; "mat_add"; "mat_mul_el"; "mat_transpose"; "add_bias"; "add_bias_and_activate";
        "relu_fwd"; "sigmoid_fwd"; "tanh_fwd"; 
        "relu_bwd"; "sigmoid_bwd"; "tanh_bwd";
        "linear_fwd"; "linear_bwd_weights"; "linear_bwd_input";
        "conv2d_direct_fwd"; "conv2d_direct_bwd_input"; "conv2d_direct_bwd_weights"; "conv2d_bias_bwd";
        "im2col"; "col2im_optimized"; 
        "permute_nhwc_nchw"; "permute_nchw_nhwc";
        "maxpool_fwd"; "maxpool_bwd";
        "adam_step"; "zero_buf"; "mse_grad";
        "confusion_matrix_update"; "cm_to_float"
      ] in
      
      List.iter (fun name -> 
        let func = Library.new_function_with_name library name in
        let pipeline, _ = ComputePipelineState.on_device_with_function device func in
        Hashtbl.add pipelines name pipeline
      ) kernel_names;
      
      let ctx = { 
        device; 
        queue; 
        pipelines; 
        commands_in_buffer = 0;
        buffer_pool = create_buffer_pool ();
        pending_buffers = [];
        active_cb = None;
      } in 
      ctx_ref := Some ctx; 
      ctx

let get_buffer ctx len =
  get_buffer_from_pool ctx.buffer_pool ctx len

let release_buffer ctx buffer size =
  ctx.pending_buffers <- (buffer, size) :: ctx.pending_buffers

let release (t : tensor) =
  if not t.released then begin
    t.released <- true;
    match !ctx_ref with 
    | Some ctx -> release_buffer ctx t.buffer (t.rows * t.cols * 4)
    | None -> ()
  end

let get_cb ctx = 
  (* Check if we need to flush the current command buffer *)
  if ctx.commands_in_buffer >= max_commands_per_buffer then begin
    match ctx.active_cb with
    | Some cb -> 
        CommandBuffer.commit cb; 
        CommandBuffer.wait_until_completed cb;
        ctx.active_cb <- None;
        ctx.commands_in_buffer <- 0;
        flush_pending_buffers ctx
    | None -> ()
  end;
  
  (* Get or create command buffer *)
  match ctx.active_cb with 
  | Some cb -> cb 
  | None -> 
      let cb = CommandBuffer.on_queue ctx.queue in 
      ctx.active_cb <- Some cb; 
      cb

let increment_command_count ctx = 
  ctx.commands_in_buffer <- ctx.commands_in_buffer + 1

let commit_batch () = 
  match !ctx_ref with
  | Some ctx ->
      (match ctx.active_cb with 
       | Some cb -> 
           CommandBuffer.commit cb; 
           ctx.active_cb <- None;
           ctx.commands_in_buffer <- 0
       | None -> ())
  | None -> ()

(* ============================================================================
   Tensor Creation and Conversion
   ============================================================================ *)

let of_cpu (t : float array array) =
  let ctx = get_ctx () in
  let r = Array.length t in
  let c = if r > 0 then Array.length t.(0) else 0 in
  let len = max 4 (r * c * 4) in
  let b = get_buffer ctx len in
  let p = Buffer.contents b |> coerce (ptr void) (ptr float) in
  let ca = CArray.from_ptr p (r * c) in
  Array.iteri (fun i row -> 
    Array.iteri (fun j v -> 
      CArray.set ca (i * c + j) v
    ) row
  ) t;
  { buffer = b; rows = r; cols = c; released = false }

let to_cpu (gt : tensor) =
  sync (); 
  let r, c = gt.rows, gt.cols in 
  if r = 0 then [||] else begin
    let t = Array.make_matrix r c 0.0 in
    let p = Buffer.contents gt.buffer |> coerce (ptr void) (ptr float) in
    let ca = CArray.from_ptr p (r * c) in
    for i = 0 to r - 1 do 
      for j = 0 to c - 1 do 
        t.(i).(j) <- CArray.get ca (i * c + j) 
      done 
    done; 
    t
  end

let copy_inplace src dst =
  if src.rows <> dst.rows || src.cols <> dst.cols then
    failwith "Gpu.copy_inplace: dimension mismatch";
  
  let ctx = get_ctx () in
  let total = src.rows * src.cols in
  let len_bytes = total * 4 in
  
  (* Use Metal's blit encoder for efficient copy *)
  let cb = get_cb ctx in
  let blit_enc = BlitCommandEncoder.on_buffer cb in
  BlitCommandEncoder.copy_from_buffer blit_enc 
    ~source_buffer:src.buffer ~source_offset:0
    ~destination_buffer:dst.buffer ~destination_offset:0
    ~size:len_bytes;
  BlitCommandEncoder.end_encoding blit_enc;
  increment_command_count ctx

let zero_tensor t =
  let ctx = get_ctx () in
  let total = t.rows * t.cols in
  let enc = ComputeCommandEncoder.on_buffer (get_cb ctx) in
  ComputeCommandEncoder.set_compute_pipeline_state enc (Hashtbl.find ctx.pipelines "zero_buf");
  ComputeCommandEncoder.set_buffer enc t.buffer ~index:0;
  
  (* Use inline parameter for small constant *)
  let len_buf = get_buffer ctx 4 in
  let p = Buffer.contents len_buf |> coerce (ptr void) (ptr uint) in
  (<-@) p (Unsigned.UInt.of_int total);
  ComputeCommandEncoder.set_buffer enc len_buf ~index:1;
  
  ComputeCommandEncoder.dispatch_threadgroups enc 
    ~threadgroups_per_grid:{width=(total+1023)/1024; height=1; depth=1} 
    ~threads_per_threadgroup:{width=1024; height=1; depth=1};
  ComputeCommandEncoder.end_encoding enc;
  increment_command_count ctx;
  release_buffer ctx len_buf 4

(* ============================================================================
   Core Matrix Operations
   ============================================================================ *)

let matmul a b =
  let ctx = get_ctx () in 
  let m, k = a.rows, a.cols in
  let k2, n = b.rows, b.cols in
  if k <> k2 then failwith "Gpu.matmul: dimension mismatch";
  
  let out_b = get_buffer ctx (m * n * 4) in
  let enc = ComputeCommandEncoder.on_buffer (get_cb ctx) in
  
  ComputeCommandEncoder.set_compute_pipeline_state enc (Hashtbl.find ctx.pipelines "matmul");
  ComputeCommandEncoder.set_buffer enc a.buffer ~index:0; 
  ComputeCommandEncoder.set_buffer enc b.buffer ~index:1; 
  ComputeCommandEncoder.set_buffer enc out_b ~index:2;
  
  (* Use packed parameters structure *)
  let params_buf = get_buffer ctx 12 in  (* 3 ints = 12 bytes *)
  let p = Buffer.contents params_buf |> coerce (ptr void) (ptr int) in
  let ca = CArray.from_ptr p 3 in
  CArray.set ca 0 m;
  CArray.set ca 1 n;
  CArray.set ca 2 k;
  ComputeCommandEncoder.set_buffer enc params_buf ~index:3;
  
  ComputeCommandEncoder.dispatch_threadgroups enc 
    ~threadgroups_per_grid:{width=(n+31)/32; height=(m+31)/32; depth=1} 
    ~threads_per_threadgroup:{width=32; height=32; depth=1};
  ComputeCommandEncoder.end_encoding enc;
  increment_command_count ctx;
  release_buffer ctx params_buf 12;
  
  { buffer = out_b; rows = m; cols = n; released = false }

let add a b =
  let ctx = get_ctx () in 
  let m, n = a.rows, a.cols in
  if b.rows <> m || b.cols <> n then failwith "Gpu.add: dimension mismatch";
  
  let total = m * n in
  let out_b = get_buffer ctx (total * 4) in
  let enc = ComputeCommandEncoder.on_buffer (get_cb ctx) in
  
  ComputeCommandEncoder.set_compute_pipeline_state enc (Hashtbl.find ctx.pipelines "mat_add");
  ComputeCommandEncoder.set_buffer enc a.buffer ~index:0; 
  ComputeCommandEncoder.set_buffer enc b.buffer ~index:1; 
  ComputeCommandEncoder.set_buffer enc out_b ~index:2;
  
  let len_buf = get_buffer ctx 4 in
  let p = Buffer.contents len_buf |> coerce (ptr void) (ptr uint) in
  (<-@) p (Unsigned.UInt.of_int total);
  ComputeCommandEncoder.set_buffer enc len_buf ~index:3;
  
  ComputeCommandEncoder.dispatch_threadgroups enc 
    ~threadgroups_per_grid:{width=(total+1023)/1024; height=1; depth=1} 
    ~threads_per_threadgroup:{width=1024; height=1; depth=1};
  ComputeCommandEncoder.end_encoding enc;
  increment_command_count ctx;
  release_buffer ctx len_buf 4;
  
  { buffer = out_b; rows = m; cols = n; released = false }

let mul a b =
  let ctx = get_ctx () in 
  let m, n = a.rows, a.cols in
  if b.rows <> m || b.cols <> n then failwith "Gpu.mul: dimension mismatch";
  
  let total = m * n in
  let out_b = get_buffer ctx (total * 4) in
  let enc = ComputeCommandEncoder.on_buffer (get_cb ctx) in
  
  ComputeCommandEncoder.set_compute_pipeline_state enc (Hashtbl.find ctx.pipelines "mat_mul_el");
  ComputeCommandEncoder.set_buffer enc a.buffer ~index:0; 
  ComputeCommandEncoder.set_buffer enc b.buffer ~index:1; 
  ComputeCommandEncoder.set_buffer enc out_b ~index:2;
  
  let len_buf = get_buffer ctx 4 in
  let p = Buffer.contents len_buf |> coerce (ptr void) (ptr uint) in
  (<-@) p (Unsigned.UInt.of_int total);
  ComputeCommandEncoder.set_buffer enc len_buf ~index:3;
  
  ComputeCommandEncoder.dispatch_threadgroups enc 
    ~threadgroups_per_grid:{width=(total+1023)/1024; height=1; depth=1} 
    ~threads_per_threadgroup:{width=1024; height=1; depth=1};
  ComputeCommandEncoder.end_encoding enc;
  increment_command_count ctx;
  release_buffer ctx len_buf 4;
  
  { buffer = out_b; rows = m; cols = n; released = false }

let transpose a =
  let ctx = get_ctx () in 
  let out_b = get_buffer ctx (a.rows * a.cols * 4) in
  let enc = ComputeCommandEncoder.on_buffer (get_cb ctx) in
  
  ComputeCommandEncoder.set_compute_pipeline_state enc (Hashtbl.find ctx.pipelines "mat_transpose");
  ComputeCommandEncoder.set_buffer enc a.buffer ~index:0; 
  ComputeCommandEncoder.set_buffer enc out_b ~index:1;
  
  let rows_buf = get_buffer ctx 4 in
  let cols_buf = get_buffer ctx 4 in
  let p1 = Buffer.contents rows_buf |> coerce (ptr void) (ptr int) in
  let p2 = Buffer.contents cols_buf |> coerce (ptr void) (ptr int) in
  (<-@) p1 a.rows;
  (<-@) p2 a.cols;
  ComputeCommandEncoder.set_buffer enc rows_buf ~index:2;
  ComputeCommandEncoder.set_buffer enc cols_buf ~index:3;
  
  ComputeCommandEncoder.dispatch_threadgroups enc 
    ~threadgroups_per_grid:{width=(a.cols+31)/32; height=(a.rows+31)/32; depth=1} 
    ~threads_per_threadgroup:{width=32; height=32; depth=1};
  ComputeCommandEncoder.end_encoding enc;
  increment_command_count ctx;
  release_buffer ctx rows_buf 4;
  release_buffer ctx cols_buf 4;
  
  { buffer = out_b; rows = a.cols; cols = a.rows; released = false }

let add_bias x bias =
  let ctx = get_ctx () in 
  let enc = ComputeCommandEncoder.on_buffer (get_cb ctx) in
  
  ComputeCommandEncoder.set_compute_pipeline_state enc (Hashtbl.find ctx.pipelines "add_bias");
  ComputeCommandEncoder.set_buffer enc x.buffer ~index:0; 
  ComputeCommandEncoder.set_buffer enc bias.buffer ~index:1;
  
  let cols_buf = get_buffer ctx 4 in
  let p = Buffer.contents cols_buf |> coerce (ptr void) (ptr int) in
  (<-@) p x.cols;
  ComputeCommandEncoder.set_buffer enc cols_buf ~index:2;
  
  ComputeCommandEncoder.dispatch_threadgroups enc 
    ~threadgroups_per_grid:{width=(x.cols+31)/32; height=(x.rows+31)/32; depth=1} 
    ~threads_per_threadgroup:{width=32; height=32; depth=1};
  ComputeCommandEncoder.end_encoding enc;
  increment_command_count ctx;
  release_buffer ctx cols_buf 4;
  x

(* ============================================================================
   Activation Functions
   ============================================================================ *)

let activation_fwd name x =
  let ctx = get_ctx () in
  let total = x.rows * x.cols in
  let enc = ComputeCommandEncoder.on_buffer (get_cb ctx) in
  
  let kernel_name = match name with
    | "relu" -> "relu_fwd"
    | "sigmoid" -> "sigmoid_fwd"
    | "tanh" -> "tanh_fwd"
    | _ -> failwith ("Unknown activation: " ^ name)
  in
  
  ComputeCommandEncoder.set_compute_pipeline_state enc (Hashtbl.find ctx.pipelines kernel_name);
  ComputeCommandEncoder.set_buffer enc x.buffer ~index:0;
  
  let len_buf = get_buffer ctx 4 in
  let p = Buffer.contents len_buf |> coerce (ptr void) (ptr uint) in
  (<-@) p (Unsigned.UInt.of_int total);
  ComputeCommandEncoder.set_buffer enc len_buf ~index:1;
  
  ComputeCommandEncoder.dispatch_threadgroups enc 
    ~threadgroups_per_grid:{width=(total+1023)/1024; height=1; depth=1} 
    ~threads_per_threadgroup:{width=1024; height=1; depth=1};
  ComputeCommandEncoder.end_encoding enc;
  increment_command_count ctx;
  release_buffer ctx len_buf 4;
  x

let relu = activation_fwd "relu"
let sigmoid = activation_fwd "sigmoid"
let tanh = activation_fwd "tanh"

let activation_bwd name z upstream =
  let ctx = get_ctx () in
  let total = z.rows * z.cols in
  let out_b = get_buffer ctx (total * 4) in
  let enc = ComputeCommandEncoder.on_buffer (get_cb ctx) in
  
  let kernel_name = match name with
    | "relu" -> "relu_bwd"
    | "sigmoid" -> "sigmoid_bwd"
    | "tanh" -> "tanh_bwd"
    | _ -> failwith ("Unknown activation: " ^ name)
  in
  
  ComputeCommandEncoder.set_compute_pipeline_state enc (Hashtbl.find ctx.pipelines kernel_name);
  ComputeCommandEncoder.set_buffer enc z.buffer ~index:0;
  ComputeCommandEncoder.set_buffer enc upstream.buffer ~index:1;
  ComputeCommandEncoder.set_buffer enc out_b ~index:2;
  
  let len_buf = get_buffer ctx 4 in
  let p = Buffer.contents len_buf |> coerce (ptr void) (ptr uint) in
  (<-@) p (Unsigned.UInt.of_int total);
  ComputeCommandEncoder.set_buffer enc len_buf ~index:3;
  
  ComputeCommandEncoder.dispatch_threadgroups enc 
    ~threadgroups_per_grid:{width=(total+1023)/1024; height=1; depth=1} 
    ~threads_per_threadgroup:{width=1024; height=1; depth=1};
  ComputeCommandEncoder.end_encoding enc;
  increment_command_count ctx;
  release_buffer ctx len_buf 4;
  
  { buffer = out_b; rows = z.rows; cols = z.cols; released = false }

(* ============================================================================
   Linear Layer Operations
   ============================================================================ *)

let linear_fwd input weights bias batch out_dim in_dim act_type =
  let ctx = get_ctx () in 
  let out_b = get_buffer ctx (batch * out_dim * 4) in
  let enc = ComputeCommandEncoder.on_buffer (get_cb ctx) in
  
  ComputeCommandEncoder.set_compute_pipeline_state enc (Hashtbl.find ctx.pipelines "linear_fwd");
  ComputeCommandEncoder.set_buffer enc input.buffer ~index:0; 
  ComputeCommandEncoder.set_buffer enc weights.buffer ~index:1; 
  ComputeCommandEncoder.set_buffer enc bias.buffer ~index:2; 
  ComputeCommandEncoder.set_buffer enc out_b ~index:3;
  
  (* Use packed parameters *)
  let params_buf = get_buffer ctx 16 in  (* 4 ints = 16 bytes *)
  let p = Buffer.contents params_buf |> coerce (ptr void) (ptr int) in
  let ca = CArray.from_ptr p 4 in
  CArray.set ca 0 batch;
  CArray.set ca 1 out_dim;
  CArray.set ca 2 in_dim;
  CArray.set ca 3 act_type;
  ComputeCommandEncoder.set_buffer enc params_buf ~index:4;
  
  ComputeCommandEncoder.dispatch_threadgroups enc 
    ~threadgroups_per_grid:{width=(out_dim+31)/32; height=(batch+31)/32; depth=1} 
    ~threads_per_threadgroup:{width=32; height=32; depth=1};
  ComputeCommandEncoder.end_encoding enc;
  increment_command_count ctx;
  release_buffer ctx params_buf 16;
  
  { buffer = out_b; rows = batch; cols = out_dim; released = false }

let linear_bwd upstream in_cache weights out_cache grad_w grad_b batch out_dim in_dim act_type =
  let ctx = get_ctx () in 
  let grad_in_b = get_buffer ctx (batch * in_dim * 4) in
  
  (* Backward for weights *)
  let enc = ComputeCommandEncoder.on_buffer (get_cb ctx) in
  ComputeCommandEncoder.set_compute_pipeline_state enc (Hashtbl.find ctx.pipelines "linear_bwd_weights");
  ComputeCommandEncoder.set_buffer enc upstream.buffer ~index:0; 
  ComputeCommandEncoder.set_buffer enc in_cache.buffer ~index:1; 
  ComputeCommandEncoder.set_buffer enc out_cache.buffer ~index:2; 
  ComputeCommandEncoder.set_buffer enc grad_w.buffer ~index:3; 
  ComputeCommandEncoder.set_buffer enc grad_b.buffer ~index:4;
  
  let params_buf = get_buffer ctx 16 in
  let p = Buffer.contents params_buf |> coerce (ptr void) (ptr int) in
  let ca = CArray.from_ptr p 4 in
  CArray.set ca 0 batch;
  CArray.set ca 1 out_dim;
  CArray.set ca 2 in_dim;
  CArray.set ca 3 act_type;
  ComputeCommandEncoder.set_buffer enc params_buf ~index:5;
  
  ComputeCommandEncoder.dispatch_threadgroups enc 
    ~threadgroups_per_grid:{width=(in_dim+31)/32; height=(out_dim+31)/32; depth=1} 
    ~threads_per_threadgroup:{width=32; height=32; depth=1};
  ComputeCommandEncoder.end_encoding enc;
  increment_command_count ctx;
  
  (* Backward for input *)
  let enc = ComputeCommandEncoder.on_buffer (get_cb ctx) in
  ComputeCommandEncoder.set_compute_pipeline_state enc (Hashtbl.find ctx.pipelines "linear_bwd_input");
  ComputeCommandEncoder.set_buffer enc upstream.buffer ~index:0; 
  ComputeCommandEncoder.set_buffer enc weights.buffer ~index:1; 
  ComputeCommandEncoder.set_buffer enc out_cache.buffer ~index:2; 
  ComputeCommandEncoder.set_buffer enc grad_in_b ~index:3;
  ComputeCommandEncoder.set_buffer enc params_buf ~index:4;
  
  ComputeCommandEncoder.dispatch_threadgroups enc 
    ~threadgroups_per_grid:{width=(in_dim+31)/32; height=(batch+31)/32; depth=1} 
    ~threads_per_threadgroup:{width=32; height=32; depth=1};
  ComputeCommandEncoder.end_encoding enc;
  increment_command_count ctx;
  release_buffer ctx params_buf 16;
  
  { buffer = grad_in_b; rows = batch; cols = in_dim; released = false }

(* Continue in next message due to length... *)
