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
    for (int k = 0; k < K; k++) sum += A[id.y * K + k] * B[k * N + id.x];
    C[id.y * N + id.x] = sum;
}

kernel void mat_add(device const float *A [[buffer(0)]], device const float *B [[buffer(1)]], device float *C [[buffer(2)]], uint id [[thread_position_in_grid]]) {
    C[id] = A[id] + B[id];
}

kernel void relu_fwd(device float *X [[buffer(0)]], uint id [[thread_position_in_grid]]) { X[id] = max(0.0, X[id]); }
kernel void sigmoid_fwd(device float *X [[buffer(0)]], uint id [[thread_position_in_grid]]) { X[id] = 1.0/(1.0+exp(-X[id])); }
kernel void tanh_fwd(device float *X [[buffer(0)]], uint id [[thread_position_in_grid]]) { X[id] = tanh(X[id]); }

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
    grad_w[id.y * K + id.x] += dw / M; if (id.x == 0) grad_b[id.y] += db / M;
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
    constant float &wd [[buffer(10)]], uint id [[thread_position_in_grid]])
{
    float g = G[id]; float m = b1 * M_t[id] + (1.0 - b1) * g; float v = b2 * V_t[id] + (1.0 - b2) * g * g;
    M_t[id] = m; V_t[id] = v;
    W[id] = W[id] - lr * wd * W[id] - lr * (m / (1.0 - b1p)) / (sqrt(v / (1.0 - b2p)) + eps);
}

kernel void mse_grad(device const float *p [[buffer(0)]], device const float *t [[buffer(1)]], device float *g [[buffer(2)]],
                     constant float &scale [[buffer(3)]], uint id [[thread_position_in_grid]]) {
    g[id] = scale * (p[id] - t[id]);
}
|}

type context = { device : Device.t; queue : CommandQueue.t; pipelines : (string, ComputePipelineState.t) Hashtbl.t }
let ctx_ref = ref None
let active_cb = ref None

let get_ctx () =
  match !ctx_ref with Some ctx -> ctx | None ->
      let device = Device.create_system_default () in
      let queue = CommandQueue.on_device device in
      let library = Library.on_device device ~source:shader_source (CompileOptions.init ()) in
      let pipelines = Hashtbl.create 16 in
      let names = ["matmul";"mat_add";"relu_fwd";"sigmoid_fwd";"tanh_fwd";"linear_fwd";"linear_bwd_weights";"linear_bwd_input";"adam_step";"mse_grad"] in
      List.iter (fun n -> Hashtbl.add pipelines n (let f = Library.new_function_with_name library n in fst (ComputePipelineState.on_device_with_function device f))) names;
      let ctx = { device; queue; pipelines } in ctx_ref := Some ctx; ctx

let get_cb ctx = match !active_cb with Some cb -> cb | None -> let cb = CommandBuffer.on_queue ctx.queue in active_cb := Some cb; cb
let commit_batch () = match !active_cb with Some cb -> CommandBuffer.commit cb; active_cb := None | None -> ()
let sync () = match !active_cb with Some cb -> CommandBuffer.commit cb; CommandBuffer.wait_until_completed cb; active_cb := None | None -> ()

type tensor = { buffer : Buffer.t; rows : int; cols : int }

let of_cpu (t : float array array) =
    let ctx = get_ctx () in
    let r, c = Array.length t, if Array.length t > 0 then Array.length t.(0) else 0 in
    let b = Buffer.on_device ctx.device ~length:(max 4 (r * c * 4)) ResourceOptions.storage_mode_shared in
    let p = Buffer.contents b |> coerce (ptr void) (ptr float) in
    let ca = CArray.from_ptr p (r * c) in
    Array.iteri (fun i row -> Array.iteri (fun j v -> CArray.set ca (i * c + j) v) row) t;
    { buffer = b; rows = r; cols = c }

let to_cpu (gt : tensor) =
    let r, c = gt.rows, gt.cols in if r = 0 then [||] else
    let t = Array.make_matrix r c 0.0 in
    let p = Buffer.contents gt.buffer |> coerce (ptr void) (ptr float) in
    let ca = CArray.from_ptr p (r * c) in
    for i = 0 to r - 1 do for j = 0 to c - 1 do t.(i).(j) <- CArray.get ca (i * c + j) done done; t

let make_int_buf ctx v = let b = Buffer.on_device ctx.device ~length:4 ResourceOptions.storage_mode_shared in let p = Buffer.contents b |> coerce (ptr void) (ptr int) in (<-@) p v; b
let make_float_buf ctx v = let b = Buffer.on_device ctx.device ~length:4 ResourceOptions.storage_mode_shared in let p = Buffer.contents b |> coerce (ptr void) (ptr float) in (<-@) p v; b

let linear_fwd input weights bias batch out_dim in_dim act_type =
    let ctx = get_ctx () in let out_b = Buffer.on_device ctx.device ~length:(batch * out_dim * 4) ResourceOptions.storage_mode_shared in
    let enc = ComputeCommandEncoder.on_buffer (get_cb ctx) in
    ComputeCommandEncoder.set_compute_pipeline_state enc (Hashtbl.find ctx.pipelines "linear_fwd");
    ComputeCommandEncoder.set_buffer enc input.buffer ~index:0; ComputeCommandEncoder.set_buffer enc weights.buffer ~index:1; ComputeCommandEncoder.set_buffer enc bias.buffer ~index:2; ComputeCommandEncoder.set_buffer enc out_b ~index:3;
    ComputeCommandEncoder.set_buffer enc (make_int_buf ctx batch) ~index:4; ComputeCommandEncoder.set_buffer enc (make_int_buf ctx out_dim) ~index:5; ComputeCommandEncoder.set_buffer enc (make_int_buf ctx in_dim) ~index:6; ComputeCommandEncoder.set_buffer enc (make_int_buf ctx act_type) ~index:7;
    ComputeCommandEncoder.dispatch_threadgroups enc ~threadgroups_per_grid:{width=(out_dim+31)/32; height=(batch+31)/32; depth=1} ~threads_per_threadgroup:{width=32; height=32; depth=1};
    ComputeCommandEncoder.end_encoding enc; { buffer = out_b; rows = batch; cols = out_dim }

let linear_bwd upstream in_cache weights out_cache grad_w grad_b batch out_dim in_dim act_type =
    let ctx = get_ctx () in let grad_in_b = Buffer.on_device ctx.device ~length:(batch * in_dim * 4) ResourceOptions.storage_mode_shared in
    let enc = ComputeCommandEncoder.on_buffer (get_cb ctx) in
    ComputeCommandEncoder.set_compute_pipeline_state enc (Hashtbl.find ctx.pipelines "linear_bwd_weights");
    ComputeCommandEncoder.set_buffer enc upstream.buffer ~index:0; ComputeCommandEncoder.set_buffer enc in_cache.buffer ~index:1; ComputeCommandEncoder.set_buffer enc out_cache.buffer ~index:2; ComputeCommandEncoder.set_buffer enc grad_w.buffer ~index:3; ComputeCommandEncoder.set_buffer enc grad_b.buffer ~index:4;
    ComputeCommandEncoder.set_buffer enc (make_int_buf ctx batch) ~index:5; ComputeCommandEncoder.set_buffer enc (make_int_buf ctx out_dim) ~index:6; ComputeCommandEncoder.set_buffer enc (make_int_buf ctx in_dim) ~index:7; ComputeCommandEncoder.set_buffer enc (make_int_buf ctx act_type) ~index:8;
    ComputeCommandEncoder.dispatch_threadgroups enc ~threadgroups_per_grid:{width=(in_dim+31)/32; height=(out_dim+31)/32; depth=1} ~threads_per_threadgroup:{width=32; height=32; depth=1};
    ComputeCommandEncoder.set_compute_pipeline_state enc (Hashtbl.find ctx.pipelines "linear_bwd_input");
    ComputeCommandEncoder.set_buffer enc upstream.buffer ~index:0; ComputeCommandEncoder.set_buffer enc weights.buffer ~index:1; ComputeCommandEncoder.set_buffer enc out_cache.buffer ~index:2; ComputeCommandEncoder.set_buffer enc grad_in_b ~index:3;
    ComputeCommandEncoder.set_buffer enc (make_int_buf ctx batch) ~index:4; ComputeCommandEncoder.set_buffer enc (make_int_buf ctx out_dim) ~index:5; ComputeCommandEncoder.set_buffer enc (make_int_buf ctx in_dim) ~index:6; ComputeCommandEncoder.set_buffer enc (make_int_buf ctx act_type) ~index:7;
    ComputeCommandEncoder.dispatch_threadgroups enc ~threadgroups_per_grid:{width=(in_dim+31)/32; height=(batch+31)/32; depth=1} ~threads_per_threadgroup:{width=32; height=32; depth=1};
    ComputeCommandEncoder.end_encoding enc; { buffer = grad_in_b; rows = batch; cols = in_dim }

let adam_step w g m v lr b1 b2 b1p b2p eps wd =
    let ctx = get_ctx () in let total = w.rows * w.cols in let enc = ComputeCommandEncoder.on_buffer (get_cb ctx) in
    ComputeCommandEncoder.set_compute_pipeline_state enc (Hashtbl.find ctx.pipelines "adam_step");
    ComputeCommandEncoder.set_buffer enc w.buffer ~index:0; ComputeCommandEncoder.set_buffer enc g.buffer ~index:1; ComputeCommandEncoder.set_buffer enc m.buffer ~index:2; ComputeCommandEncoder.set_buffer enc v.buffer ~index:3;
    ComputeCommandEncoder.set_buffer enc (make_float_buf ctx lr) ~index:4; ComputeCommandEncoder.set_buffer enc (make_float_buf ctx b1) ~index:5; ComputeCommandEncoder.set_buffer enc (make_float_buf ctx b2) ~index:6; ComputeCommandEncoder.set_buffer enc (make_float_buf ctx b1p) ~index:7; ComputeCommandEncoder.set_buffer enc (make_float_buf ctx b2p) ~index:8; ComputeCommandEncoder.set_buffer enc (make_float_buf ctx eps) ~index:9; ComputeCommandEncoder.set_buffer enc (make_float_buf ctx wd) ~index:10;
    ComputeCommandEncoder.dispatch_threadgroups enc ~threadgroups_per_grid:{width=(total+1023)/1024; height=1; depth=1} ~threads_per_threadgroup:{width=1024; height=1; depth=1};
    ComputeCommandEncoder.end_encoding enc

let mse_grad p t scale =
    let ctx = get_ctx () in let res_b = Buffer.on_device ctx.device ~length:(p.rows * p.cols * 4) ResourceOptions.storage_mode_shared in
    let enc = ComputeCommandEncoder.on_buffer (get_cb ctx) in
    ComputeCommandEncoder.set_compute_pipeline_state enc (Hashtbl.find ctx.pipelines "mse_grad");
    ComputeCommandEncoder.set_buffer enc p.buffer ~index:0; ComputeCommandEncoder.set_buffer enc t.buffer ~index:1; ComputeCommandEncoder.set_buffer enc res_b ~index:2; ComputeCommandEncoder.set_buffer enc (make_float_buf ctx scale) ~index:3;
    ComputeCommandEncoder.dispatch_threadgroups enc ~threadgroups_per_grid:{width=(p.rows*p.cols+1023)/1024; height=1; depth=1} ~threads_per_threadgroup:{width=1024; height=1; depth=1};
    ComputeCommandEncoder.end_encoding enc; { buffer = res_b; rows = p.rows; cols = p.cols }

let copy_inplace src dst = let ctx = get_ctx () in let enc = BlitCommandEncoder.on_buffer (get_cb ctx) in BlitCommandEncoder.copy_from_buffer enc ~source_buffer:src.buffer ~source_offset:0 ~destination_buffer:dst.buffer ~destination_offset:0 ~size:(src.rows*src.cols*4); BlitCommandEncoder.end_encoding enc

let matmul a b =
    let ctx = get_ctx () in let m, k, n = a.rows, a.cols, b.cols in let out_b = Buffer.on_device ctx.device ~length:(m * n * 4) ResourceOptions.storage_mode_shared in
    let enc = ComputeCommandEncoder.on_buffer (get_cb ctx) in
    ComputeCommandEncoder.set_compute_pipeline_state enc (Hashtbl.find ctx.pipelines "matmul");
    ComputeCommandEncoder.set_buffer enc a.buffer ~index:0; ComputeCommandEncoder.set_buffer enc b.buffer ~index:1; ComputeCommandEncoder.set_buffer enc out_b ~index:2;
    ComputeCommandEncoder.set_buffer enc (make_int_buf ctx m) ~index:3; ComputeCommandEncoder.set_buffer enc (make_int_buf ctx n) ~index:4; ComputeCommandEncoder.set_buffer enc (make_int_buf ctx k) ~index:5;
    ComputeCommandEncoder.dispatch_threadgroups enc ~threadgroups_per_grid:{width=(n+31)/32; height=(m+31)/32; depth=1} ~threads_per_threadgroup:{width=32; height=32; depth=1};
    ComputeCommandEncoder.end_encoding enc; { buffer = out_b; rows = m; cols = n }

let add a b =
    let ctx = get_ctx () in let m, n = a.rows, a.cols in let out_b = Buffer.on_device ctx.device ~length:(m * n * 4) ResourceOptions.storage_mode_shared in
    let enc = ComputeCommandEncoder.on_buffer (get_cb ctx) in
    ComputeCommandEncoder.set_compute_pipeline_state enc (Hashtbl.find ctx.pipelines "mat_add");
    ComputeCommandEncoder.set_buffer enc a.buffer ~index:0; ComputeCommandEncoder.set_buffer enc b.buffer ~index:1; ComputeCommandEncoder.set_buffer enc out_b ~index:2;
    ComputeCommandEncoder.dispatch_threadgroups enc ~threadgroups_per_grid:{width=(m*n+1023)/1024; height=1; depth=1} ~threads_per_threadgroup:{width=1024; height=1; depth=1};
    ComputeCommandEncoder.end_encoding enc; { buffer = out_b; rows = m; cols = n }

let relu x = let ctx = get_ctx () in let enc = ComputeCommandEncoder.on_buffer (get_cb ctx) in ComputeCommandEncoder.set_compute_pipeline_state enc (Hashtbl.find ctx.pipelines "relu_fwd"); ComputeCommandEncoder.set_buffer enc x.buffer ~index:0; ComputeCommandEncoder.dispatch_threadgroups enc ~threadgroups_per_grid:{width=(x.rows*x.cols+1023)/1024; height=1; depth=1} ~threads_per_threadgroup:{width=1024; height=1; depth=1}; ComputeCommandEncoder.end_encoding enc; x
let sigmoid x = let ctx = get_ctx () in let enc = ComputeCommandEncoder.on_buffer (get_cb ctx) in ComputeCommandEncoder.set_compute_pipeline_state enc (Hashtbl.find ctx.pipelines "sigmoid_fwd"); ComputeCommandEncoder.set_buffer enc x.buffer ~index:0; ComputeCommandEncoder.dispatch_threadgroups enc ~threadgroups_per_grid:{width=(x.rows*x.cols+1023)/1024; height=1; depth=1} ~threads_per_threadgroup:{width=1024; height=1; depth=1}; ComputeCommandEncoder.end_encoding enc; x
let tanh x = let ctx = get_ctx () in let enc = ComputeCommandEncoder.on_buffer (get_cb ctx) in ComputeCommandEncoder.set_compute_pipeline_state enc (Hashtbl.find ctx.pipelines "tanh_fwd"); ComputeCommandEncoder.set_buffer enc x.buffer ~index:0; ComputeCommandEncoder.dispatch_threadgroups enc ~threadgroups_per_grid:{width=(x.rows*x.cols+1023)/1024; height=1; depth=1} ~threads_per_threadgroup:{width=1024; height=1; depth=1}; ComputeCommandEncoder.end_encoding enc; x
let transpose a =
    let ctx = get_ctx () in let out_b = Buffer.on_device ctx.device ~length:(a.rows * a.cols * 4) ResourceOptions.storage_mode_shared in
    let enc = ComputeCommandEncoder.on_buffer (get_cb ctx) in
    ComputeCommandEncoder.set_compute_pipeline_state enc (Hashtbl.find ctx.pipelines "mat_transpose");
    ComputeCommandEncoder.set_buffer enc a.buffer ~index:0; ComputeCommandEncoder.set_buffer enc out_b ~index:1;
    ComputeCommandEncoder.set_buffer enc (make_int_buf ctx a.rows) ~index:2; ComputeCommandEncoder.set_buffer enc (make_int_buf ctx a.cols) ~index:3;
    ComputeCommandEncoder.dispatch_threadgroups enc ~threadgroups_per_grid:{width=(a.cols+31)/32; height=(a.rows+31)/32; depth=1} ~threads_per_threadgroup:{width=32; height=32; depth=1};
    ComputeCommandEncoder.end_encoding enc; { buffer = out_b; rows = a.cols; cols = a.rows }

let add_bias x bias =
    let ctx = get_ctx () in let enc = ComputeCommandEncoder.on_buffer (get_cb ctx) in
    ComputeCommandEncoder.set_compute_pipeline_state enc (Hashtbl.find ctx.pipelines "add_bias");
    ComputeCommandEncoder.set_buffer enc x.buffer ~index:0; ComputeCommandEncoder.set_buffer enc bias.buffer ~index:1;
    ComputeCommandEncoder.set_buffer enc (make_int_buf ctx x.cols) ~index:2;
    ComputeCommandEncoder.dispatch_threadgroups enc ~threadgroups_per_grid:{width=(x.cols+31)/32; height=(x.rows+31)/32; depth=1} ~threads_per_threadgroup:{width=32; height=32; depth=1};
    ComputeCommandEncoder.end_encoding enc; x

let conv2d _ _ _ = failwith "Use fused kernels"
