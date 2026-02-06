open Metal
open Ctypes

let shader_source = {|
#include <metal_stdlib>
using namespace metal;

kernel void matmul(
    device const float *A [[buffer(0)]],
    device const float *B [[buffer(1)]],
    device float *C [[buffer(2)]],
    constant int &M [[buffer(3)]],
    constant int &N [[buffer(4)]],
    constant int &K [[buffer(5)]],
    uint2 id [[thread_position_in_grid]])
{
    if (id.x >= uint(N) || id.y >= uint(M)) return;
    float sum = 0.0;
    int a_row = id.y * K;
    for (int k = 0; k < K; k++) {
        sum += A[a_row + k] * B[k * N + id.x];
    }
    C[id.y * N + id.x] = sum;
}

kernel void mat_add(device const float *A [[buffer(0)]], device const float *B [[buffer(1)]], device float *C [[buffer(2)]], constant uint &len [[buffer(3)]], uint id [[thread_position_in_grid]]) {
    if (id >= len) return;
    C[id] = A[id] + B[id];
}

kernel void relu_fwd(device float *X [[buffer(0)]], constant uint &len [[buffer(1)]], uint id [[thread_position_in_grid]]) { 
    if (id >= len) return;
    X[id] = max(0.0, X[id]); 
}
kernel void sigmoid_fwd(device float *X [[buffer(0)]], constant uint &len [[buffer(1)]], uint id [[thread_position_in_grid]]) { 
    if (id >= len) return;
    X[id] = 1.0/(1.0+exp(-X[id])); 
}
kernel void tanh_fwd(device float *X [[buffer(0)]], constant uint &len [[buffer(1)]], uint id [[thread_position_in_grid]]) { 
    if (id >= len) return;
    X[id] = tanh(X[id]); 
}

kernel void relu_bwd(device const float *Z [[buffer(0)]], device const float *upstream [[buffer(1)]], device float *out [[buffer(2)]], constant uint &len [[buffer(3)]], uint id [[thread_position_in_grid]]) {
    if (id >= len) return;
    out[id] = Z[id] > 0 ? upstream[id] : 0.0;
}

kernel void sigmoid_bwd(device const float *Z [[buffer(0)]], device const float *upstream [[buffer(1)]], device float *out [[buffer(2)]], constant uint &len [[buffer(3)]], uint id [[thread_position_in_grid]]) {
    if (id >= len) return;
    float s = 1.0 / (1.0 + exp(-Z[id]));
    out[id] = upstream[id] * s * (1.0 - s);
}

kernel void tanh_bwd(device const float *Z [[buffer(0)]], device const float *upstream [[buffer(1)]], device float *out [[buffer(2)]], constant uint &len [[buffer(3)]], uint id [[thread_position_in_grid]]) {
    if (id >= len) return;
    float t = tanh(Z[id]);
    out[id] = upstream[id] * (1.0 - t * t);
}

kernel void conv2d_bias_bwd(
    device const float *grad_out [[buffer(0)]],
    device float *grad_b [[buffer(1)]],
    constant int &N [[buffer(2)]],
    constant int &OutH [[buffer(3)]],
    constant int &OutW [[buffer(4)]],
    constant int &out_depth [[buffer(5)]],
    uint oc [[thread_position_in_grid]])
{
    if (oc >= uint(out_depth)) return;
    float sum = 0.0;
    int spatial_size = OutH * OutW;
    for (int b = 0; b < N; b++) {
        int base = b * out_depth * spatial_size + oc * spatial_size;
        for (int i = 0; i < spatial_size; i++) {
            sum += grad_out[base + i];
        }
    }
    grad_b[oc] += sum / float(N);
}

kernel void linear_fwd(
    device const float *in [[buffer(0)]], device const float *weights [[buffer(1)]], device const float *bias [[buffer(2)]], device float *out [[buffer(3)]],
    constant int &M [[buffer(4)]], constant int &N [[buffer(5)]], constant int &K [[buffer(6)]], constant int &act [[buffer(7)]],
    uint2 id [[thread_position_in_grid]])
{
    if (id.x >= uint(N) || id.y >= uint(M)) return;
    float sum = bias[id.x];
    for (int k = 0; k < K; k++) sum += in[id.y * K + k] * weights[id.x * K + k];
    if (act == 1) sum = max(0.0, sum); else if (act == 2) sum = tanh(sum); else if (act == 3) sum = 1.0/(1.0+exp(-sum));
    out[id.y * N + id.x] = sum;
}

kernel void linear_bwd_weights(
    device const float *upstream [[buffer(0)]], device const float *in_cache [[buffer(1)]], device const float *out_cache [[buffer(2)]],
    device float *grad_w [[buffer(3)]], device float *grad_b [[buffer(4)]],
    constant int &M [[buffer(5)]], constant int &N [[buffer(6)]], constant int &K [[buffer(7)]], constant int &act [[buffer(8)]],
    uint2 id [[thread_position_in_grid]])
{
    if (id.x >= uint(K) || id.y >= uint(N)) return;
    float dw = 0.0; float db = 0.0;
    for (int b = 0; b < M; b++) {
        float dy = upstream[b * N + id.y];
        float y = out_cache[b * N + id.y];
        if (act == 1) dy = y > 0 ? dy : 0; else if (act == 2) dy = dy * (1.0 - y * y); else if (act == 3) dy = dy * y * (1.0 - y);
        dw += dy * in_cache[b * K + id.x]; if (id.x == 0) db += dy;
    }
    grad_w[id.y * K + id.x] += dw; if (id.x == 0) grad_b[id.y] += db;
}

kernel void linear_bwd_input(
    device const float *upstream [[buffer(0)]], device const float *weights [[buffer(1)]], device const float *out_cache [[buffer(2)]], device float *grad_in [[buffer(3)]],
    constant int &M [[buffer(4)]], constant int &N [[buffer(5)]], constant int &K [[buffer(6)]], constant int &act [[buffer(7)]],
    uint2 id [[thread_position_in_grid]])
{
    if (id.x >= uint(K) || id.y >= uint(M)) return;
    float sum = 0.0;
    for (int n = 0; n < N; n++) {
        float dy = upstream[id.y * N + n]; float y = out_cache[id.y * N + n];
        if (act == 1) dy = y > 0 ? dy : 0; else if (act == 2) dy = dy * (1.0 - y * y); else if (act == 3) dy = dy * y * (1.0 - y);
        sum += dy * weights[n * K + id.x];
    }
    grad_in[id.y * K + id.x] = sum;
}

kernel void adam_step(
    device float *W [[buffer(0)]], device const float *G [[buffer(1)]], device float *M_t [[buffer(2)]], device float *V_t [[buffer(3)]],
    constant float &lr [[buffer(4)]], constant float &b1 [[buffer(5)]], constant float &b2 [[buffer(6)]],
    constant float &b1p [[buffer(7)]], constant float &b2p [[buffer(8)]], constant float &eps [[buffer(9)]],
    constant float &wd [[buffer(10)]], constant uint &len [[buffer(11)]], uint id [[thread_position_in_grid]])
{
    if (id >= len) return;
    float g = G[id]; float m = b1 * M_t[id] + (1.0 - b1) * g; float v = b2 * V_t[id] + (1.0 - b2) * g * g;
    M_t[id] = m; V_t[id] = v;
    W[id] = W[id] - lr * wd * W[id] - lr * (m / (1.0 - b1p)) / (sqrt(v / (1.0 - b2p)) + eps);
}

kernel void zero_buf(device float *X [[buffer(0)]], constant uint &len [[buffer(1)]], uint id [[thread_position_in_grid]]) {
    if (id >= len) return;
    X[id] = 0.0;
}

kernel void mse_grad(device const float *p [[buffer(0)]], device const float *t [[buffer(1)]], device float *g [[buffer(2)]],
                     constant float &scale [[buffer(3)]], constant uint &len [[buffer(4)]], uint id [[thread_position_in_grid]]) {
    if (id >= len) return;
    g[id] = scale * (p[id] - t[id]);
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

kernel void add_bias(
    device float *X [[buffer(0)]],
    device const float *bias [[buffer(1)]],
    constant int &cols [[buffer(2)]],
    uint2 id [[thread_position_in_grid]])
{
    if (id.x >= uint(cols)) return;
    X[id.y * cols + id.x] += bias[id.x];
}

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

kernel void permute_nhwc_nchw(
    device const float *in [[buffer(0)]],
    device float *out [[buffer(1)]],
    constant int *params [[buffer(2)]],
    uint3 id [[thread_position_in_grid]])
{
    int N = params[0]; int H = params[1]; int W = params[2]; int C = params[3];
    int w = id.x; int h = id.y; int bc = id.z;
    if (w >= W || h >= H || bc >= N * C) return;
    int c = bc % C; int b = bc / C;
    int nchw_idx = b * (C * H * W) + c * (H * W) + h * W + w;
    int nhwc_idx = b * (H * W * C) + h * (W * C) + w * C + c;
    out[nchw_idx] = in[nhwc_idx];
}

kernel void permute_nchw_nhwc(
    device const float *in [[buffer(0)]],
    device float *out [[buffer(1)]],
    constant int *params [[buffer(2)]],
    uint3 id [[thread_position_in_grid]])
{
    int N = params[0]; int H = params[1]; int W = params[2]; int C = params[3];
    int w = id.x; int h = id.y; int bc = id.z;
    if (w >= W || h >= H || bc >= N * C) return;
    int c = bc % C; int b = bc / C;
    int nchw_idx = b * (C * H * W) + c * (H * W) + h * W + w;
    int nhwc_idx = b * (H * W * C) + h * (W * C) + w * C + c;
    out[nhwc_idx] = in[nchw_idx];
}

kernel void conv2d_direct_fwd(
    device const float *input [[buffer(0)]],
    device const float *weights [[buffer(1)]],
    device const float *bias [[buffer(2)]],
    device float *output [[buffer(3)]],
    constant int *params [[buffer(4)]],
    uint3 id [[thread_position_in_grid]])
{
    int N = params[0]; int InH = params[1]; int InW = params[2];
    int OutH = params[3]; int OutW = params[4]; int KH = params[5]; int KW = params[6];
    int InD = params[7]; int OutD = params[8]; int act = params[9]; 
    
    int w = id.x % OutW;
    int h = id.x / OutW;
    int oc = id.y;
    int b = id.z;

    if (w >= OutW || h >= OutH || oc >= OutD || b >= N) return;
    
    float sum = bias[oc];
    int in_batch_offset = b * (InD * InH * InW);
    int weight_oc_offset = oc * (InD * KH * KW);
    
    for (int ic = 0; ic < InD; ic++) {
        int in_channel_offset = in_batch_offset + ic * (InH * InW);
        int weight_ic_offset = weight_oc_offset + ic * (KH * KW);
        for (int i = 0; i < KH; i++) {
            for (int j = 0; j < KW; j++) {
                sum += input[in_channel_offset + (h + i) * InW + (w + j)] * weights[weight_ic_offset + i * KW + j];
            }
        }
    }
    
    if (act == 1) sum = max(0.0, sum); 
    else if (act == 2) sum = tanh(sum); 
    else if (act == 3) sum = 1.0/(1.0+exp(-sum));
    
    output[b * (OutD * OutH * OutW) + oc * (OutH * OutW) + h * OutW + w] = sum;
}

kernel void conv2d_direct_bwd_input(
    device const float *grad_out [[buffer(0)]],
    device const float *weights [[buffer(1)]],
    device float *grad_in [[buffer(2)]],
    constant int *params [[buffer(3)]],
    uint3 id [[thread_position_in_grid]])
{
    int N = params[0]; int InH = params[1]; int InW = params[2];
    int OutH = params[3]; int OutW = params[4]; int KH = params[5]; int KW = params[6];
    int InD = params[7]; int OutD = params[8];
    
    int w = id.x % InW;
    int h = id.x / InW;
    int ic = id.y;
    int b = id.z;

    if (w >= InW || h >= InH || ic >= InD || b >= N) return;
    
    float sum = 0.0;
    int grad_out_batch = b * (OutD * OutH * OutW);
    for (int oc = 0; oc < OutD; oc++) {
        int grad_out_ch = grad_out_batch + oc * (OutH * OutW);
        int w_offset = oc * (InD * KH * KW) + ic * (KH * KW);
        for (int i = 0; i < KH; i++) {
            for (int j = 0; j < KW; j++) {
                int out_y = h - i; int out_x = w - j;
                if (out_y >= 0 && out_y < OutH && out_x >= 0 && out_x < OutW) {
                    sum += grad_out[grad_out_ch + out_y * OutW + out_x] * weights[w_offset + i * KW + j];
                }
            }
        }
    }
    grad_in[b * (InD * InH * InW) + ic * (InH * InW) + h * InW + w] = sum;
}

kernel void conv2d_direct_bwd_weights(
    device const float *input [[buffer(0)]],
    device const float *grad_out [[buffer(1)]],
    device float *grad_w [[buffer(2)]],
    constant int *params [[buffer(3)]],
    uint3 id [[thread_position_in_grid]])
{
    int N = params[0]; int InH = params[1]; int InW = params[2];
    int OutH = params[3]; int OutW = params[4]; int KH = params[5]; int KW = params[6];
    int InD = params[7]; int OutD = params[8];
    int k_idx = id.x; int ki = k_idx / KW; int kj = k_idx % KW; int ic = id.y; int oc = id.z;
    if (ki >= KH || ic >= InD || oc >= OutD) return;
    float sum = 0.0;
    for (int b = 0; b < N; b++) {
        int in_base = b * (InD * InH * InW) + ic * (InH * InW);
        int grad_base = b * (OutD * OutH * OutW) + oc * (OutH * OutW);
        for (int r = 0; r < OutH; r++) {
            for (int c = 0; c < OutW; c++) {
                sum += input[in_base + (r + ki) * InW + (c + kj)] * grad_out[grad_base + r * OutW + c];
            }
        }
    }
    grad_w[oc * (InD * KH * KW) + ic * (KH * KW) + k_idx] += sum / float(N);
}

kernel void im2col(
    device const float *img [[buffer(0)]],
    device float *cols [[buffer(1)]],
    constant int *params [[buffer(2)]],
    uint id [[thread_position_in_grid]])
{
    int N = params[0]; int C = params[1]; int H = params[2]; int W = params[3];
    int OutH = params[4]; int OutW = params[5]; int KH = params[6]; int KW = params[7];
    
    int total = N * OutH * OutW * C * KH * KW;
    if (id >= uint(total)) return;
    
    int kj = id % KW;
    int ki = (id / KW) % KH;
    int c = (id / (KW * KH)) % C;
    int s = (id / (KW * KH * C)) % (OutH * OutW);
    int b = id / (KW * KH * C * OutH * OutW);
    
    int out_y = s / OutW;
    int out_x = s % OutW;
    int in_y = out_y + ki;
    int in_x = out_x + kj;
    
    cols[id] = img[b * (C * H * W) + c * (H * W) + in_y * W + in_x];
}

kernel void col2im_optimized(
    device const float *cols [[buffer(0)]],
    device float *img [[buffer(1)]],
    constant int *params [[buffer(2)]],
    uint id [[thread_position_in_grid]])
{
    int N = params[0]; int C = params[1]; int H = params[2]; int W = params[3];
    int OutH = params[4]; int OutW = params[5]; int KH = params[6]; int KW = params[7];
    
    int total = N * C * H * W;
    if (id >= uint(total)) return;
    
    int px = id % W;
    int py = (id / W) % H;
    int c = (id / (W * H)) % C;
    int b = id / (W * H * C);
    
    float sum = 0.0;
    int col_width = C * KH * KW;
    for (int ki = 0; ki < KH; ki++) {
        for (int kj = 0; kj < KW; kj++) {
            int out_y = py - ki; int out_x = px - kj;
            if (out_y >= 0 && out_y < OutH && out_x >= 0 && out_x < OutW) {
                int s = out_y * OutW + out_x;
                int col_idx = (((b * OutH * OutW + s) * C + c) * KH + ki) * KW + kj;
                sum += cols[col_idx];
            }
        }
    }
    img[id] = sum;
}
kernel void mat_mul_el(device const float *A [[buffer(0)]], device const float *B [[buffer(1)]], device float *C [[buffer(2)]], constant uint &len [[buffer(3)]], uint id [[thread_position_in_grid]]) {
    if (id >= len) return;
    C[id] = A[id] * B[id];
}

kernel void maxpool_fwd(
    device const float *in [[buffer(0)]],
    device float *out [[buffer(1)]],
    device int *indices [[buffer(2)]],
    constant int *params [[buffer(3)]],
    uint3 id [[thread_position_in_grid]])
{
    int N = params[0]; int C = params[1]; int InH = params[2]; int InW = params[3];
    int OutH = params[4]; int OutW = params[5]; int KH = params[6]; int KW = params[7]; int stride = params[8];
    
    int w = id.x; int h = id.y; int bc = id.z;
    if (w >= OutW || h >= OutH || bc >= N * C) return;
    
    int b = bc / C; int c = bc % C;
    int in_base = b * (C * InH * InW) + c * (InH * InW);
    
    float max_val = -1e38;
    int max_idx = -1;
    
    for (int i = 0; i < KH; i++) {
        for (int j = 0; j < KW; j++) {
            int cur_h = h * stride + i;
            int cur_w = w * stride + j;
            if (cur_h < InH && cur_w < InW) {
                int idx = in_base + cur_h * InW + cur_w;
                float v = in[idx];
                if (v > max_val) {
                    max_val = v;
                    max_idx = idx;
                }
            }
        }
    }
    
    int out_idx = b * (C * OutH * OutW) + c * (OutH * OutW) + h * OutW + w;
    out[out_idx] = max_val;
    indices[out_idx] = max_idx;
}

kernel void maxpool_bwd(
    device const float *grad_out [[buffer(0)]],
    device const int *indices [[buffer(1)]],
    device float *grad_in [[buffer(2)]],
    constant int &total_out [[buffer(3)]],
    uint id [[thread_position_in_grid]])
{
    if (id >= uint(total_out)) return;
    int in_idx = indices[id];
    // Note: This is not thread-safe for overlapping pools (requires atomic add)
    // But for standard stride=kernel_size it is fine.
    grad_in[in_idx] += grad_out[id]; 
}
|}

type context = { 
  device : Device.t; 
  queue : CommandQueue.t; 
  pipelines : (string, ComputePipelineState.t) Hashtbl.t;
  mutable commands_in_buffer : int;
  int_buffer_pool : Buffer.t Queue.t;
  float_buffer_pool : Buffer.t Queue.t;
  buffer_cache : (int, Buffer.t Queue.t) Hashtbl.t;
  mutable pending_buffers : Buffer.t list;
  mutable pending_ints : Buffer.t list;
  mutable pending_floats : Buffer.t list;
}

let ctx_ref = ref None
let active_cb = ref None

type tensor = { buffer : Buffer.t; rows : int; cols : int; mutable released : bool }

let max_commands_per_buffer = 100
let max_pool_size = 50

let flush_pending ctx =
  List.iter (fun b ->
    let len = Buffer.length b in
    let q = match Hashtbl.find_opt ctx.buffer_cache len with
      | Some q -> q
      | None -> let q = Queue.create () in Hashtbl.add ctx.buffer_cache len q; q
    in
    if Queue.length q < max_pool_size then Queue.push b q
  ) ctx.pending_buffers;
  ctx.pending_buffers <- [];
  List.iter (fun b -> if Queue.length ctx.int_buffer_pool < max_pool_size then Queue.push b ctx.int_buffer_pool) ctx.pending_ints;
  ctx.pending_ints <- [];
  List.iter (fun b -> if Queue.length ctx.float_buffer_pool < max_pool_size then Queue.push b ctx.float_buffer_pool) ctx.pending_floats;
  ctx.pending_floats <- []

let sync () = 
  match !active_cb with 
  | Some cb -> 
      CommandBuffer.commit cb; 
      CommandBuffer.wait_until_completed cb; 
      active_cb := None;
      (match !ctx_ref with Some ctx -> 
        ctx.commands_in_buffer <- 0;
        flush_pending ctx
      | None -> ())
  | None -> ()

let get_ctx () =
  match !ctx_ref with Some ctx -> ctx | None ->
      let device = Device.create_system_default () in
      let queue = CommandQueue.on_device device in
      let library = Library.on_device device ~source:shader_source (CompileOptions.init ()) in
      let pipelines = Hashtbl.create 16 in
      let names = ["matmul";"mat_add";"mat_mul_el";"maxpool_fwd";"maxpool_bwd";"relu_fwd";"sigmoid_fwd";"tanh_fwd";"linear_fwd";"linear_bwd_weights";"linear_bwd_input";"adam_step";"mse_grad";"mat_transpose";"add_bias";"zero_buf";"confusion_matrix_update";"cm_to_float";"relu_bwd";"sigmoid_bwd";"tanh_bwd";"conv2d_bias_bwd";"im2col";"col2im_optimized";"permute_nhwc_nchw";"permute_nchw_nhwc";"conv2d_direct_fwd";"conv2d_direct_bwd_input";"conv2d_direct_bwd_weights"] in
      List.iter (fun n -> Hashtbl.add pipelines n (let f = Library.new_function_with_name library n in fst (ComputePipelineState.on_device_with_function device f))) names;
      let ctx = { 
        device; 
        queue; 
        pipelines; 
        commands_in_buffer = 0;
        int_buffer_pool = Queue.create ();
        float_buffer_pool = Queue.create ();
        buffer_cache = Hashtbl.create 16;
        pending_buffers = [];
        pending_ints = [];
        pending_floats = [];
      } in 
      ctx_ref := Some ctx; 
      ctx

let get_buffer ctx len =
  match Hashtbl.find_opt ctx.buffer_cache len with
  | Some q when not (Queue.is_empty q) -> Queue.pop q
  | _ -> Buffer.on_device ctx.device ~length:len ResourceOptions.storage_mode_shared

let release_buffer ctx (b : Buffer.t) =
  ctx.pending_buffers <- b :: ctx.pending_buffers

let release (t : tensor) =
  if not t.released then (t.released <- true; match !ctx_ref with Some ctx -> release_buffer ctx t.buffer | None -> ())

let get_cb ctx = 
  if ctx.commands_in_buffer >= max_commands_per_buffer then begin
    match !active_cb with
    | Some cb -> 
        CommandBuffer.commit cb; 
        CommandBuffer.wait_until_completed cb;
        active_cb := None;
        ctx.commands_in_buffer <- 0;
        flush_pending ctx
    | None -> ()
  end;
  match !active_cb with 
  | Some cb -> cb 
  | None -> 
      let cb = CommandBuffer.on_queue ctx.queue in 
      active_cb := Some cb; 
      cb

let increment_command_count ctx = ctx.commands_in_buffer <- ctx.commands_in_buffer + 1

let commit_batch () = 
  match !active_cb with 
  | Some cb -> 
      CommandBuffer.commit cb; 
      active_cb := None;
      (match !ctx_ref with Some ctx -> ctx.commands_in_buffer <- 0 | None -> ())
  | None -> ()

let of_cpu (t : float array array) =
    let ctx = get_ctx () in
    let r, c = Array.length t, if Array.length t > 0 then Array.length t.(0) else 0 in
    let len = max 4 (r * c * 4) in
    let b = get_buffer ctx len in
    let p = Buffer.contents b |> coerce (ptr void) (ptr float) in
    let ca = CArray.from_ptr p (r * c) in
    Array.iteri (fun i row -> Array.iteri (fun j v -> CArray.set ca (i * c + j) v) row) t;
    { buffer = b; rows = r; cols = c; released = false }

let to_cpu (gt : tensor) =
    sync (); 
    let r, c = gt.rows, gt.cols in if r = 0 then [||] else
    let t = Array.make_matrix r c 0.0 in
    let p = Buffer.contents gt.buffer |> coerce (ptr void) (ptr float) in
    let ca = CArray.from_ptr p (r * c) in
    for i = 0 to r - 1 do for j = 0 to c - 1 do t.(i).(j) <- CArray.get ca (i * c + j) done done; t

let make_int_buf ctx v = 
  let b = 
    if Queue.is_empty ctx.int_buffer_pool then
      Buffer.on_device ctx.device ~length:4 ResourceOptions.storage_mode_shared
    else
      Queue.pop ctx.int_buffer_pool
  in
  let p = Buffer.contents b |> coerce (ptr void) (ptr int) in 
  (<-@) p v; 
  b

let make_float_buf ctx v = 
  let b = 
    if Queue.is_empty ctx.float_buffer_pool then
      Buffer.on_device ctx.device ~length:4 ResourceOptions.storage_mode_shared
    else
      Queue.pop ctx.float_buffer_pool
  in
  let p = Buffer.contents b |> coerce (ptr void) (ptr float) in 
  (<-@) p v; 
  b

let make_int_array_buf ctx arr =
  let len = Array.length arr in
  let b = Buffer.on_device ctx.device ~length:(len * 4) ResourceOptions.storage_mode_shared in
  let p = Buffer.contents b |> coerce (ptr void) (ptr int) in
  let ca = CArray.from_ptr p len in
  Array.iteri (fun i v -> CArray.set ca i v) arr;
  b

let return_int_buf ctx b = ctx.pending_ints <- b :: ctx.pending_ints
let return_float_buf ctx b = ctx.pending_floats <- b :: ctx.pending_floats

let mul a b =
    let ctx = get_ctx () in 
    let m, n = a.rows, a.cols in let total = m * n in
    let out_b = get_buffer ctx (total * 4) in
    let enc = ComputeCommandEncoder.on_buffer (get_cb ctx) in
    ComputeCommandEncoder.set_compute_pipeline_state enc (Hashtbl.find ctx.pipelines "mat_mul_el");
    ComputeCommandEncoder.set_buffer enc a.buffer ~index:0; 
    ComputeCommandEncoder.set_buffer enc b.buffer ~index:1; 
    ComputeCommandEncoder.set_buffer enc out_b ~index:2;
    let b1 = make_int_buf ctx total in
    ComputeCommandEncoder.set_buffer enc b1 ~index:3;
    ComputeCommandEncoder.dispatch_threadgroups enc ~threadgroups_per_grid:{width=(total+1023)/1024; height=1; depth=1} ~threads_per_threadgroup:{width=1024; height=1; depth=1};
    ComputeCommandEncoder.end_encoding enc; increment_command_count ctx;
    return_int_buf ctx b1;
    { buffer = out_b; rows = m; cols = n; released = false }

let maxpool_fwd input n c in_h in_w out_h out_w kh kw stride =
    let ctx = get_ctx () in
    let out_b = get_buffer ctx (n * c * out_h * out_w * 4) in
    let idx_b = get_buffer ctx (n * c * out_h * out_w * 4) in
    let enc = ComputeCommandEncoder.on_buffer (get_cb ctx) in
    ComputeCommandEncoder.set_compute_pipeline_state enc (Hashtbl.find ctx.pipelines "maxpool_fwd");
    ComputeCommandEncoder.set_buffer enc input.buffer ~index:0;
    ComputeCommandEncoder.set_buffer enc out_b ~index:1;
    ComputeCommandEncoder.set_buffer enc idx_b ~index:2;
    let b_params = make_int_array_buf ctx [|n; c; in_h; in_w; out_h; out_w; kh; kw; stride|] in
    ComputeCommandEncoder.set_buffer enc b_params ~index:3;
    ComputeCommandEncoder.dispatch_threadgroups enc ~threadgroups_per_grid:{width=(out_w+7)/8; height=(out_h+7)/8; depth=(n*c+7)/8} ~threads_per_threadgroup:{width=8; height=8; depth=8};
    ComputeCommandEncoder.end_encoding enc; increment_command_count ctx;
    let out_t = { buffer = out_b; rows = n; cols = c * out_h * out_w; released = false } in
    let idx_t = { buffer = idx_b; rows = n; cols = c * out_h * out_w; released = false } in
    out_t, idx_t

let maxpool_bwd grad_out indices n c in_h in_w out_h out_w =
    let ctx = get_ctx () in
    let total_in = n * c * in_h * in_w in
    let total_out = n * c * out_h * out_w in
    let grad_in_b = get_buffer ctx (total_in * 4) in
    let enc = ComputeCommandEncoder.on_buffer (get_cb ctx) in
    (* Zero grad_in first *)
    ComputeCommandEncoder.set_compute_pipeline_state enc (Hashtbl.find ctx.pipelines "zero_buf");
    ComputeCommandEncoder.set_buffer enc grad_in_b ~index:0;
    let b1 = make_int_buf ctx total_in in ComputeCommandEncoder.set_buffer enc b1 ~index:1;
    ComputeCommandEncoder.dispatch_threadgroups enc ~threadgroups_per_grid:{width=(total_in+1023)/1024; height=1; depth=1} ~threads_per_threadgroup:{width=1024; height=1; depth=1};
    
    (* Accumulate gradients *)
    ComputeCommandEncoder.set_compute_pipeline_state enc (Hashtbl.find ctx.pipelines "maxpool_bwd");
    ComputeCommandEncoder.set_buffer enc grad_out.buffer ~index:0;
    ComputeCommandEncoder.set_buffer enc indices.buffer ~index:1;
    ComputeCommandEncoder.set_buffer enc grad_in_b ~index:2;
    let b2 = make_int_buf ctx total_out in ComputeCommandEncoder.set_buffer enc b2 ~index:3;
    ComputeCommandEncoder.dispatch_threadgroups enc ~threadgroups_per_grid:{width=(total_out+1023)/1024; height=1; depth=1} ~threads_per_threadgroup:{width=1024; height=1; depth=1};
    
    ComputeCommandEncoder.end_encoding enc; increment_command_count ctx;
    List.iter (return_int_buf ctx) [b1;b2];
    { buffer = grad_in_b; rows = n; cols = c * in_h * in_w; released = false }

let linear_fwd input weights bias batch out_dim in_dim act_type =
    let ctx = get_ctx () in 
    let out_b = get_buffer ctx (batch * out_dim * 4) in
    let enc = ComputeCommandEncoder.on_buffer (get_cb ctx) in
    ComputeCommandEncoder.set_compute_pipeline_state enc (Hashtbl.find ctx.pipelines "linear_fwd");
    ComputeCommandEncoder.set_buffer enc input.buffer ~index:0; 
    ComputeCommandEncoder.set_buffer enc weights.buffer ~index:1; 
    ComputeCommandEncoder.set_buffer enc bias.buffer ~index:2; 
    ComputeCommandEncoder.set_buffer enc out_b ~index:3;
    let b1 = make_int_buf ctx batch in let b2 = make_int_buf ctx out_dim in
    let b3 = make_int_buf ctx in_dim in let b4 = make_int_buf ctx act_type in
    ComputeCommandEncoder.set_buffer enc b1 ~index:4; ComputeCommandEncoder.set_buffer enc b2 ~index:5; 
    ComputeCommandEncoder.set_buffer enc b3 ~index:6; ComputeCommandEncoder.set_buffer enc b4 ~index:7;
    ComputeCommandEncoder.dispatch_threadgroups enc ~threadgroups_per_grid:{width=(out_dim+31)/32; height=(batch+31)/32; depth=1} ~threads_per_threadgroup:{width=32; height=32; depth=1};
    ComputeCommandEncoder.end_encoding enc; increment_command_count ctx;
    List.iter (return_int_buf ctx) [b1;b2;b3;b4];
    { buffer = out_b; rows = batch; cols = out_dim; released = false }

let linear_bwd upstream in_cache weights out_cache grad_w grad_b batch out_dim in_dim act_type =
    let ctx = get_ctx () in 
    let grad_in_b = get_buffer ctx (batch * in_dim * 4) in
    let enc = ComputeCommandEncoder.on_buffer (get_cb ctx) in
    ComputeCommandEncoder.set_compute_pipeline_state enc (Hashtbl.find ctx.pipelines "linear_bwd_weights");
    ComputeCommandEncoder.set_buffer enc upstream.buffer ~index:0; 
    ComputeCommandEncoder.set_buffer enc in_cache.buffer ~index:1; 
    ComputeCommandEncoder.set_buffer enc out_cache.buffer ~index:2; 
    ComputeCommandEncoder.set_buffer enc grad_w.buffer ~index:3; 
    ComputeCommandEncoder.set_buffer enc grad_b.buffer ~index:4;
    let b1 = make_int_buf ctx batch in let b2 = make_int_buf ctx out_dim in
    let b3 = make_int_buf ctx in_dim in let b4 = make_int_buf ctx act_type in
    ComputeCommandEncoder.set_buffer enc b1 ~index:5; ComputeCommandEncoder.set_buffer enc b2 ~index:6; 
    ComputeCommandEncoder.set_buffer enc b3 ~index:7; ComputeCommandEncoder.set_buffer enc b4 ~index:8;
    ComputeCommandEncoder.dispatch_threadgroups enc ~threadgroups_per_grid:{width=(in_dim+31)/32; height=(out_dim+31)/32; depth=1} ~threads_per_threadgroup:{width=32; height=32; depth=1};
    ComputeCommandEncoder.set_compute_pipeline_state enc (Hashtbl.find ctx.pipelines "linear_bwd_input");
    ComputeCommandEncoder.set_buffer enc upstream.buffer ~index:0; 
    ComputeCommandEncoder.set_buffer enc weights.buffer ~index:1; 
    ComputeCommandEncoder.set_buffer enc out_cache.buffer ~index:2; 
    ComputeCommandEncoder.set_buffer enc grad_in_b ~index:3;
    ComputeCommandEncoder.set_buffer enc b1 ~index:4; ComputeCommandEncoder.set_buffer enc b2 ~index:5; 
    ComputeCommandEncoder.set_buffer enc b3 ~index:6; ComputeCommandEncoder.set_buffer enc b4 ~index:7;
    ComputeCommandEncoder.dispatch_threadgroups enc ~threadgroups_per_grid:{width=(in_dim+31)/32; height=(batch+31)/32; depth=1} ~threads_per_threadgroup:{width=32; height=32; depth=1};
    ComputeCommandEncoder.end_encoding enc; increment_command_count ctx;
    List.iter (return_int_buf ctx) [b1;b2;b3;b4];
    { buffer = grad_in_b; rows = batch; cols = in_dim; released = false }

let adam_step w g m v lr b1 b2 b1p b2p eps wd =
    let ctx = get_ctx () in 
    let total = w.rows * w.cols in 
    let enc = ComputeCommandEncoder.on_buffer (get_cb ctx) in
    ComputeCommandEncoder.set_compute_pipeline_state enc (Hashtbl.find ctx.pipelines "adam_step");
    ComputeCommandEncoder.set_buffer enc w.buffer ~index:0; 
    ComputeCommandEncoder.set_buffer enc g.buffer ~index:1; 
    ComputeCommandEncoder.set_buffer enc m.buffer ~index:2; 
    ComputeCommandEncoder.set_buffer enc v.buffer ~index:3;
    let f1 = make_float_buf ctx lr in let f2 = make_float_buf ctx b1 in
    let f3 = make_float_buf ctx b2 in let f4 = make_float_buf ctx b1p in
    let f5 = make_float_buf ctx b2p in let f6 = make_float_buf ctx eps in
    let f7 = make_float_buf ctx wd in let i1 = make_int_buf ctx total in
    ComputeCommandEncoder.set_buffer enc f1 ~index:4; ComputeCommandEncoder.set_buffer enc f2 ~index:5; 
    ComputeCommandEncoder.set_buffer enc f3 ~index:6; ComputeCommandEncoder.set_buffer enc f4 ~index:7; 
    ComputeCommandEncoder.set_buffer enc f5 ~index:8; ComputeCommandEncoder.set_buffer enc f6 ~index:9; 
    ComputeCommandEncoder.set_buffer enc f7 ~index:10; ComputeCommandEncoder.set_buffer enc i1 ~index:11;
    ComputeCommandEncoder.dispatch_threadgroups enc ~threadgroups_per_grid:{width=(total+1023)/1024; height=1; depth=1} ~threads_per_threadgroup:{width=1024; height=1; depth=1};
    ComputeCommandEncoder.end_encoding enc; increment_command_count ctx;
    List.iter (return_float_buf ctx) [f1;f2;f3;f4;f5;f6;f7]; return_int_buf ctx i1

let mse_grad p t scale =
    let ctx = get_ctx () in 
    let total = p.rows * p.cols in
    let res_b = get_buffer ctx (total * 4) in
    let enc = ComputeCommandEncoder.on_buffer (get_cb ctx) in
    ComputeCommandEncoder.set_compute_pipeline_state enc (Hashtbl.find ctx.pipelines "mse_grad");
    ComputeCommandEncoder.set_buffer enc p.buffer ~index:0; 
    ComputeCommandEncoder.set_buffer enc t.buffer ~index:1; 
    ComputeCommandEncoder.set_buffer enc res_b ~index:2;
    let f1 = make_float_buf ctx scale in let i1 = make_int_buf ctx total in
    ComputeCommandEncoder.set_buffer enc f1 ~index:3; ComputeCommandEncoder.set_buffer enc i1 ~index:4;
    ComputeCommandEncoder.dispatch_threadgroups enc ~threadgroups_per_grid:{width=(total+1023)/1024; height=1; depth=1} ~threads_per_threadgroup:{width=1024; height=1; depth=1};
    ComputeCommandEncoder.end_encoding enc; increment_command_count ctx;
    return_float_buf ctx f1; return_int_buf ctx i1;
    { buffer = res_b; rows = p.rows; cols = p.cols; released = false }

let copy_inplace src dst = 
    let ctx = get_ctx () in 
    let enc = BlitCommandEncoder.on_buffer (get_cb ctx) in 
    BlitCommandEncoder.copy_from_buffer enc ~source_buffer:src.buffer ~source_offset:0 ~destination_buffer:dst.buffer ~destination_offset:0 ~size:(src.rows*src.cols*4); 
    BlitCommandEncoder.end_encoding enc; increment_command_count ctx

let matmul a b =
    let ctx = get_ctx () in 
    let m, k, n = a.rows, a.cols, b.cols in 
    let out_b = get_buffer ctx (m * n * 4) in
    let enc = ComputeCommandEncoder.on_buffer (get_cb ctx) in
    ComputeCommandEncoder.set_compute_pipeline_state enc (Hashtbl.find ctx.pipelines "matmul");
    ComputeCommandEncoder.set_buffer enc a.buffer ~index:0; 
    ComputeCommandEncoder.set_buffer enc b.buffer ~index:1; 
    ComputeCommandEncoder.set_buffer enc out_b ~index:2;
    let b1 = make_int_buf ctx m in let b2 = make_int_buf ctx n in let b3 = make_int_buf ctx k in
    ComputeCommandEncoder.set_buffer enc b1 ~index:3; ComputeCommandEncoder.set_buffer enc b2 ~index:4; 
    ComputeCommandEncoder.set_buffer enc b3 ~index:5;
    ComputeCommandEncoder.dispatch_threadgroups enc ~threadgroups_per_grid:{width=(n+31)/32; height=(m+31)/32; depth=1} ~threads_per_threadgroup:{width=32; height=32; depth=1};
    ComputeCommandEncoder.end_encoding enc; increment_command_count ctx;
    List.iter (return_int_buf ctx) [b1;b2;b3];
    { buffer = out_b; rows = m; cols = n; released = false }

let add a b =
    let ctx = get_ctx () in 
    let m, n = a.rows, a.cols in let total = m * n in
    let out_b = get_buffer ctx (total * 4) in
    let enc = ComputeCommandEncoder.on_buffer (get_cb ctx) in
    ComputeCommandEncoder.set_compute_pipeline_state enc (Hashtbl.find ctx.pipelines "mat_add");
    ComputeCommandEncoder.set_buffer enc a.buffer ~index:0; 
    ComputeCommandEncoder.set_buffer enc b.buffer ~index:1; 
    ComputeCommandEncoder.set_buffer enc out_b ~index:2;
    let b1 = make_int_buf ctx total in
    ComputeCommandEncoder.set_buffer enc b1 ~index:3;
    ComputeCommandEncoder.dispatch_threadgroups enc ~threadgroups_per_grid:{width=(total+1023)/1024; height=1; depth=1} ~threads_per_threadgroup:{width=1024; height=1; depth=1};
    ComputeCommandEncoder.end_encoding enc; increment_command_count ctx;
    return_int_buf ctx b1;
    { buffer = out_b; rows = m; cols = n; released = false }

let relu x = 
    let ctx = get_ctx () in let total = x.rows * x.cols in
    let enc = ComputeCommandEncoder.on_buffer (get_cb ctx) in 
    ComputeCommandEncoder.set_compute_pipeline_state enc (Hashtbl.find ctx.pipelines "relu_fwd"); 
    ComputeCommandEncoder.set_buffer enc x.buffer ~index:0; 
    let b1 = make_int_buf ctx total in
    ComputeCommandEncoder.set_buffer enc b1 ~index:1;
    ComputeCommandEncoder.dispatch_threadgroups enc ~threadgroups_per_grid:{width=(total+1023)/1024; height=1; depth=1} ~threads_per_threadgroup:{width=1024; height=1; depth=1}; 
    ComputeCommandEncoder.end_encoding enc; increment_command_count ctx;
    return_int_buf ctx b1; x

let sigmoid x = 
    let ctx = get_ctx () in let total = x.rows * x.cols in
    let enc = ComputeCommandEncoder.on_buffer (get_cb ctx) in 
    ComputeCommandEncoder.set_compute_pipeline_state enc (Hashtbl.find ctx.pipelines "sigmoid_fwd"); 
    ComputeCommandEncoder.set_buffer enc x.buffer ~index:0; 
    let b1 = make_int_buf ctx total in
    ComputeCommandEncoder.set_buffer enc b1 ~index:1;
    ComputeCommandEncoder.dispatch_threadgroups enc ~threadgroups_per_grid:{width=(total+1023)/1024; height=1; depth=1} ~threads_per_threadgroup:{width=1024; height=1; depth=1}; 
    ComputeCommandEncoder.end_encoding enc; increment_command_count ctx;
    return_int_buf ctx b1; x

let tanh x = 
    let ctx = get_ctx () in let total = x.rows * x.cols in
    let enc = ComputeCommandEncoder.on_buffer (get_cb ctx) in 
    ComputeCommandEncoder.set_compute_pipeline_state enc (Hashtbl.find ctx.pipelines "tanh_fwd"); 
    ComputeCommandEncoder.set_buffer enc x.buffer ~index:0; 
    let b1 = make_int_buf ctx total in
    ComputeCommandEncoder.set_buffer enc b1 ~index:1;
    ComputeCommandEncoder.dispatch_threadgroups enc ~threadgroups_per_grid:{width=(total+1023)/1024; height=1; depth=1} ~threads_per_threadgroup:{width=1024; height=1; depth=1}; 
    ComputeCommandEncoder.end_encoding enc; increment_command_count ctx;
    return_int_buf ctx b1; x

let zero_tensor x =
    let ctx = get_ctx () in let total = x.rows * x.cols in
    let enc = ComputeCommandEncoder.on_buffer (get_cb ctx) in 
    ComputeCommandEncoder.set_compute_pipeline_state enc (Hashtbl.find ctx.pipelines "zero_buf"); 
    ComputeCommandEncoder.set_buffer enc x.buffer ~index:0; 
    let b1 = make_int_buf ctx total in
    ComputeCommandEncoder.set_buffer enc b1 ~index:1;
    ComputeCommandEncoder.dispatch_threadgroups enc ~threadgroups_per_grid:{width=(total+1023)/1024; height=1; depth=1} ~threads_per_threadgroup:{width=1024; height=1; depth=1}; 
    ComputeCommandEncoder.end_encoding enc; increment_command_count ctx;
    return_int_buf ctx b1

let transpose a =
    let ctx = get_ctx () in 
    let out_b = get_buffer ctx (a.rows * a.cols * 4) in
    let enc = ComputeCommandEncoder.on_buffer (get_cb ctx) in
    ComputeCommandEncoder.set_compute_pipeline_state enc (Hashtbl.find ctx.pipelines "mat_transpose");
    ComputeCommandEncoder.set_buffer enc a.buffer ~index:0; 
    ComputeCommandEncoder.set_buffer enc out_b ~index:1;
    let b1 = make_int_buf ctx a.rows in let b2 = make_int_buf ctx a.cols in
    ComputeCommandEncoder.set_buffer enc b1 ~index:2; ComputeCommandEncoder.set_buffer enc b2 ~index:3;
    ComputeCommandEncoder.dispatch_threadgroups enc ~threadgroups_per_grid:{width=(a.cols+31)/32; height=(a.rows+31)/32; depth=1} ~threads_per_threadgroup:{width=32; height=32; depth=1};
    ComputeCommandEncoder.end_encoding enc; increment_command_count ctx;
    List.iter (return_int_buf ctx) [b1;b2];
    { buffer = out_b; rows = a.cols; cols = a.rows; released = false }

let add_bias x bias =
    let ctx = get_ctx () in 
    let enc = ComputeCommandEncoder.on_buffer (get_cb ctx) in
    ComputeCommandEncoder.set_compute_pipeline_state enc (Hashtbl.find ctx.pipelines "add_bias");
    ComputeCommandEncoder.set_buffer enc x.buffer ~index:0; 
    ComputeCommandEncoder.set_buffer enc bias.buffer ~index:1;
    let b1 = make_int_buf ctx x.cols in
    ComputeCommandEncoder.set_buffer enc b1 ~index:2;
    ComputeCommandEncoder.dispatch_threadgroups enc ~threadgroups_per_grid:{width=(x.cols+31)/32; height=(x.rows+31)/32; depth=1} ~threads_per_threadgroup:{width=32; height=32; depth=1};
    ComputeCommandEncoder.end_encoding enc; increment_command_count ctx;
    return_int_buf ctx b1; x

let activation_bwd act_name z upstream =
    let ctx = get_ctx () in let total = z.rows * z.cols in
    let out_b = get_buffer ctx (total * 4) in
    let enc = ComputeCommandEncoder.on_buffer (get_cb ctx) in
    let pipe_name = match act_name with "relu" -> "relu_bwd" | "sigmoid" -> "sigmoid_bwd" | "tanh" -> "tanh_bwd" | _ -> failwith ("Unsupported activation for GPU bwd: " ^ act_name) in
    ComputeCommandEncoder.set_compute_pipeline_state enc (Hashtbl.find ctx.pipelines pipe_name);
    ComputeCommandEncoder.set_buffer enc z.buffer ~index:0;
    ComputeCommandEncoder.set_buffer enc upstream.buffer ~index:1;
    ComputeCommandEncoder.set_buffer enc out_b ~index:2;
    let b1 = make_int_buf ctx total in ComputeCommandEncoder.set_buffer enc b1 ~index:3;
    ComputeCommandEncoder.dispatch_threadgroups enc ~threadgroups_per_grid:{width=(total+1023)/1024; height=1; depth=1} ~threads_per_threadgroup:{width=1024; height=1; depth=1};
    ComputeCommandEncoder.end_encoding enc; increment_command_count ctx;
    return_int_buf ctx b1;
    { buffer = out_b; rows = z.rows; cols = z.cols; released = false }

let conv2d_bias_bwd grad_out grad_b n out_h out_w out_depth =
    let ctx = get_ctx () in
    let enc = ComputeCommandEncoder.on_buffer (get_cb ctx) in
    ComputeCommandEncoder.set_compute_pipeline_state enc (Hashtbl.find ctx.pipelines "conv2d_bias_bwd");
    ComputeCommandEncoder.set_buffer enc grad_out.buffer ~index:0;
    ComputeCommandEncoder.set_buffer enc grad_b.buffer ~index:1;
    let b1 = make_int_buf ctx n in let b2 = make_int_buf ctx out_h in let b3 = make_int_buf ctx out_w in
    let b4 = make_int_buf ctx out_depth in
    ComputeCommandEncoder.set_buffer enc b1 ~index:2; ComputeCommandEncoder.set_buffer enc b2 ~index:3;
    ComputeCommandEncoder.set_buffer enc b3 ~index:4; ComputeCommandEncoder.set_buffer enc b4 ~index:5;
    ComputeCommandEncoder.dispatch_threadgroups enc ~threadgroups_per_grid:{width=(out_depth+63)/64; height=1; depth=1} ~threads_per_threadgroup:{width=64; height=1; depth=1};
    ComputeCommandEncoder.end_encoding enc; increment_command_count ctx;
    List.iter (return_int_buf ctx) [b1;b2;b3;b4]

let im2col img cols n c h w out_h out_w kh kw =
    let ctx = get_ctx () in
    let enc = ComputeCommandEncoder.on_buffer (get_cb ctx) in
    ComputeCommandEncoder.set_compute_pipeline_state enc (Hashtbl.find ctx.pipelines "im2col");
    ComputeCommandEncoder.set_buffer enc img.buffer ~index:0;
    ComputeCommandEncoder.set_buffer enc cols.buffer ~index:1;
    let b_params = make_int_array_buf ctx [|n; c; h; w; out_h; out_w; kh; kw|] in
    ComputeCommandEncoder.set_buffer enc b_params ~index:2;
    let total = n * out_h * out_w * c * kh * kw in
    ComputeCommandEncoder.dispatch_threadgroups enc ~threadgroups_per_grid:{width=(total+1023)/1024; height=1; depth=1} ~threads_per_threadgroup:{width=1024; height=1; depth=1};
    ComputeCommandEncoder.end_encoding enc; increment_command_count ctx; ()

let col2im cols img n c h w out_h out_w kh kw =
    let ctx = get_ctx () in
    let enc = ComputeCommandEncoder.on_buffer (get_cb ctx) in
    ComputeCommandEncoder.set_compute_pipeline_state enc (Hashtbl.find ctx.pipelines "col2im_optimized");
    ComputeCommandEncoder.set_buffer enc cols.buffer ~index:0;
    ComputeCommandEncoder.set_buffer enc img.buffer ~index:1;
    let b_params = make_int_array_buf ctx [|n; c; h; w; out_h; out_w; kh; kw|] in
    ComputeCommandEncoder.set_buffer enc b_params ~index:2;
    let total = n * c * h * w in
    ComputeCommandEncoder.dispatch_threadgroups enc ~threadgroups_per_grid:{width=(total+1023)/1024; height=1; depth=1} ~threads_per_threadgroup:{width=1024; height=1; depth=1};
    ComputeCommandEncoder.end_encoding enc; increment_command_count ctx; ()

let permute_nhwc_nchw input output n h w c =
    let ctx = get_ctx () in
    let enc = ComputeCommandEncoder.on_buffer (get_cb ctx) in
    ComputeCommandEncoder.set_compute_pipeline_state enc (Hashtbl.find ctx.pipelines "permute_nhwc_nchw");
    ComputeCommandEncoder.set_buffer enc input.buffer ~index:0;
    ComputeCommandEncoder.set_buffer enc output.buffer ~index:1;
    let b_params = make_int_array_buf ctx [|n; h; w; c|] in
    ComputeCommandEncoder.set_buffer enc b_params ~index:2;
    ComputeCommandEncoder.dispatch_threadgroups enc ~threadgroups_per_grid:{width=(w+7)/8; height=(h+7)/8; depth=(n*c+7)/8} ~threads_per_threadgroup:{width=8; height=8; depth=8};
    ComputeCommandEncoder.end_encoding enc; increment_command_count ctx; ()

let permute_nchw_nhwc input output n h w c =
    let ctx = get_ctx () in
    let enc = ComputeCommandEncoder.on_buffer (get_cb ctx) in
    ComputeCommandEncoder.set_compute_pipeline_state enc (Hashtbl.find ctx.pipelines "permute_nchw_nhwc");
    ComputeCommandEncoder.set_buffer enc input.buffer ~index:0;
    ComputeCommandEncoder.set_buffer enc output.buffer ~index:1;
    let b_params = make_int_array_buf ctx [|n; h; w; c|] in
    ComputeCommandEncoder.set_buffer enc b_params ~index:2;
    ComputeCommandEncoder.dispatch_threadgroups enc ~threadgroups_per_grid:{width=(w+7)/8; height=(h+7)/8; depth=(n*c+7)/8} ~threads_per_threadgroup:{width=8; height=8; depth=8};
    ComputeCommandEncoder.end_encoding enc; increment_command_count ctx; ()

let conv2d_direct_fwd input weights bias output n in_h in_w out_h out_w kh kw in_d out_d _act =
    let ctx = get_ctx () in
    let col_len = n * out_h * out_w * in_d * kh * kw in
    let cols = get_buffer ctx (col_len * 4) in
    let cols_t = {buffer=cols; rows=n*out_h*out_w; cols=in_d*kh*kw; released=false} in
    im2col input cols_t n in_d in_h in_w out_h out_w kh kw;
    
    let w_t = transpose {buffer=weights.buffer; rows=out_d; cols=in_d*kh*kw; released=false} in
    let res = matmul cols_t w_t in
    
    let out_nhwc = {buffer=res.buffer; rows=n; cols=out_h*out_w*out_d; released=false} in
    permute_nhwc_nchw out_nhwc output n out_h out_w out_d;
    
    let enc = ComputeCommandEncoder.on_buffer (get_cb ctx) in
    ComputeCommandEncoder.set_compute_pipeline_state enc (Hashtbl.find ctx.pipelines "add_bias");
    ComputeCommandEncoder.set_buffer enc output.buffer ~index:0;
    ComputeCommandEncoder.set_buffer enc bias.buffer ~index:1;
    let b1 = make_int_buf ctx out_d in
    ComputeCommandEncoder.set_buffer enc b1 ~index:2;
    ComputeCommandEncoder.dispatch_threadgroups enc ~threadgroups_per_grid:{width=(out_h*out_w+31)/32; height=n*out_d; depth=1} ~threads_per_threadgroup:{width=32; height=1; depth=1};
    ComputeCommandEncoder.end_encoding enc; increment_command_count ctx;
    
    return_int_buf ctx b1;
    release w_t; release res; release cols_t

let conv2d_direct_bwd_weights input grad_out grad_w n in_h in_w out_h out_w kh kw in_d out_d =
    let ctx = get_ctx () in
    let col_len = n * out_h * out_w * in_d * kh * kw in
    let cols = get_buffer ctx (col_len * 4) in
    let cols_t = {buffer=cols; rows=n*out_h*out_w; cols=in_d*kh*kw; released=false} in
    im2col input cols_t n in_d in_h in_w out_h out_w kh kw;
    
    let dz_nhwc_b = get_buffer ctx (n * out_h * out_w * out_d * 4) in
    let dz_nhwc = {buffer=dz_nhwc_b; rows=n*out_h*out_w; cols=out_d; released=false} in
    permute_nchw_nhwc grad_out dz_nhwc n out_h out_w out_d;
    
    let dz_t = transpose dz_nhwc in
    let gw_mat = matmul dz_t cols_t in
    
    let total = out_d * in_d * kh * kw in
    let enc = ComputeCommandEncoder.on_buffer (get_cb ctx) in
    ComputeCommandEncoder.set_compute_pipeline_state enc (Hashtbl.find ctx.pipelines "mat_add");
    ComputeCommandEncoder.set_buffer enc grad_w.buffer ~index:0;
    ComputeCommandEncoder.set_buffer enc gw_mat.buffer ~index:1;
    ComputeCommandEncoder.set_buffer enc grad_w.buffer ~index:2;
    let b1 = make_int_buf ctx total in
    ComputeCommandEncoder.set_buffer enc b1 ~index:3;
    ComputeCommandEncoder.dispatch_threadgroups enc ~threadgroups_per_grid:{width=(total+1023)/1024; height=1; depth=1} ~threads_per_threadgroup:{width=1024; height=1; depth=1};
    ComputeCommandEncoder.end_encoding enc; increment_command_count ctx;
    
    return_int_buf ctx b1;
    release dz_t; release dz_nhwc; release gw_mat; release cols_t

let conv2d_direct_bwd_input grad_out weights grad_in n in_h in_w out_h out_w kh kw in_d out_d =
    let ctx = get_ctx () in
    let dz_nhwc_b = get_buffer ctx (n * out_h * out_w * out_d * 4) in
    let dz_nhwc = {buffer=dz_nhwc_b; rows=n*out_h*out_w; cols=out_d; released=false} in
    permute_nchw_nhwc grad_out dz_nhwc n out_h out_w out_d;
    
    let d_cols = matmul dz_nhwc {buffer=weights.buffer; rows=out_d; cols=in_d*kh*kw; released=false} in
    
    col2im d_cols grad_in n in_d in_h in_w out_h out_w kh kw;
    
    release dz_nhwc; release d_cols

let update_confusion_matrix preds targets cm =
    let ctx = get_ctx () in
    let n = preds.rows in let c = preds.cols in
    if targets.rows <> n || targets.cols <> c then failwith "Gpu.update_confusion_matrix: Dimension mismatch";
    if cm.rows <> c || cm.cols <> c then failwith "Gpu.update_confusion_matrix: CM dimension mismatch";
    let enc = ComputeCommandEncoder.on_buffer (get_cb ctx) in
    ComputeCommandEncoder.set_compute_pipeline_state enc (Hashtbl.find ctx.pipelines "confusion_matrix_update");
    ComputeCommandEncoder.set_buffer enc preds.buffer ~index:0;
    ComputeCommandEncoder.set_buffer enc targets.buffer ~index:1;
    ComputeCommandEncoder.set_buffer enc cm.buffer ~index:2;
    let b1 = make_int_buf ctx n in let b2 = make_int_buf ctx c in
    ComputeCommandEncoder.set_buffer enc b1 ~index:3; ComputeCommandEncoder.set_buffer enc b2 ~index:4;
    ComputeCommandEncoder.dispatch_threadgroups enc ~threadgroups_per_grid:{width=(n+1023)/1024; height=1; depth=1} ~threads_per_threadgroup:{width=1024; height=1; depth=1};
    ComputeCommandEncoder.end_encoding enc; increment_command_count ctx;
    return_int_buf ctx b1; return_int_buf ctx b2

let cm_to_float_tensor cm =
    let ctx = get_ctx () in let len = cm.rows * cm.cols in
    let out_b = get_buffer ctx (len * 4) in
    let enc = ComputeCommandEncoder.on_buffer (get_cb ctx) in
    ComputeCommandEncoder.set_compute_pipeline_state enc (Hashtbl.find ctx.pipelines "cm_to_float");
    ComputeCommandEncoder.set_buffer enc cm.buffer ~index:0;
    ComputeCommandEncoder.set_buffer enc out_b ~index:1;
    let b1 = make_int_buf ctx len in ComputeCommandEncoder.set_buffer enc b1 ~index:2;
    ComputeCommandEncoder.dispatch_threadgroups enc ~threadgroups_per_grid:{width=(len+1023)/1024; height=1; depth=1} ~threads_per_threadgroup:{width=1024; height=1; depth=1};
    ComputeCommandEncoder.end_encoding enc; increment_command_count ctx;
    return_int_buf ctx b1; { buffer = out_b; rows = cm.rows; cols = cm.cols; released = false }

let cleanup () =
  sync (); match !ctx_ref with Some ctx -> Queue.clear ctx.int_buffer_pool; Queue.clear ctx.float_buffer_pool; Hashtbl.clear ctx.buffer_cache; ctx.commands_in_buffer <- 0 | None -> ()
