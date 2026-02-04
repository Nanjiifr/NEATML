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

(* Maximum commands before auto-commit to prevent memory buildup *)
let max_commands_per_buffer = 100
(* Maximum pool size for temporary buffers *)
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
      let names = ["matmul";"mat_add";"relu_fwd";"sigmoid_fwd";"tanh_fwd";"linear_fwd";"linear_bwd_weights";"linear_bwd_input";"adam_step";"mse_grad";"mat_transpose";"add_bias";"zero_buf";"confusion_matrix_update";"cm_to_float"] in
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

let get_cb ctx = 
  (* Auto-commit if buffer is getting too large *)
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

let increment_command_count ctx =
  ctx.commands_in_buffer <- ctx.commands_in_buffer + 1

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
    sync (); (* Ensure GPU is done before reading *)
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

let return_int_buf ctx b =
  ctx.pending_ints <- b :: ctx.pending_ints

let return_float_buf ctx b =
  ctx.pending_floats <- b :: ctx.pending_floats

let linear_fwd input weights bias batch out_dim in_dim act_type =
    let ctx = get_ctx () in 
    let out_b = get_buffer ctx (batch * out_dim * 4) in
    let enc = ComputeCommandEncoder.on_buffer (get_cb ctx) in
    ComputeCommandEncoder.set_compute_pipeline_state enc (Hashtbl.find ctx.pipelines "linear_fwd");
    ComputeCommandEncoder.set_buffer enc input.buffer ~index:0; 
    ComputeCommandEncoder.set_buffer enc weights.buffer ~index:1; 
    ComputeCommandEncoder.set_buffer enc bias.buffer ~index:2; 
    ComputeCommandEncoder.set_buffer enc out_b ~index:3;
    let buf_batch = make_int_buf ctx batch in
    let buf_out_dim = make_int_buf ctx out_dim in
    let buf_in_dim = make_int_buf ctx in_dim in
    let buf_act_type = make_int_buf ctx act_type in
    ComputeCommandEncoder.set_buffer enc buf_batch ~index:4; 
    ComputeCommandEncoder.set_buffer enc buf_out_dim ~index:5; 
    ComputeCommandEncoder.set_buffer enc buf_in_dim ~index:6; 
    ComputeCommandEncoder.set_buffer enc buf_act_type ~index:7;
    ComputeCommandEncoder.dispatch_threadgroups enc ~threadgroups_per_grid:{width=(out_dim+31)/32; height=(batch+31)/32; depth=1} ~threads_per_threadgroup:{width=32; height=32; depth=1};
    ComputeCommandEncoder.end_encoding enc; 
    increment_command_count ctx;
    (* Return buffers to pool after encoding *)
    return_int_buf ctx buf_batch;
    return_int_buf ctx buf_out_dim;
    return_int_buf ctx buf_in_dim;
    return_int_buf ctx buf_act_type;
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
    let buf_batch = make_int_buf ctx batch in
    let buf_out_dim = make_int_buf ctx out_dim in
    let buf_in_dim = make_int_buf ctx in_dim in
    let buf_act_type = make_int_buf ctx act_type in
    ComputeCommandEncoder.set_buffer enc buf_batch ~index:5; 
    ComputeCommandEncoder.set_buffer enc buf_out_dim ~index:6; 
    ComputeCommandEncoder.set_buffer enc buf_in_dim ~index:7; 
    ComputeCommandEncoder.set_buffer enc buf_act_type ~index:8;
    ComputeCommandEncoder.dispatch_threadgroups enc ~threadgroups_per_grid:{width=(in_dim+31)/32; height=(out_dim+31)/32; depth=1} ~threads_per_threadgroup:{width=32; height=32; depth=1};
    ComputeCommandEncoder.set_compute_pipeline_state enc (Hashtbl.find ctx.pipelines "linear_bwd_input");
    ComputeCommandEncoder.set_buffer enc upstream.buffer ~index:0; 
    ComputeCommandEncoder.set_buffer enc weights.buffer ~index:1; 
    ComputeCommandEncoder.set_buffer enc out_cache.buffer ~index:2; 
    ComputeCommandEncoder.set_buffer enc grad_in_b ~index:3;
    ComputeCommandEncoder.set_buffer enc buf_batch ~index:4; 
    ComputeCommandEncoder.set_buffer enc buf_out_dim ~index:5; 
    ComputeCommandEncoder.set_buffer enc buf_in_dim ~index:6; 
    ComputeCommandEncoder.set_buffer enc buf_act_type ~index:7;
    ComputeCommandEncoder.dispatch_threadgroups enc ~threadgroups_per_grid:{width=(in_dim+31)/32; height=(batch+31)/32; depth=1} ~threads_per_threadgroup:{width=32; height=32; depth=1};
    ComputeCommandEncoder.end_encoding enc; 
    increment_command_count ctx;
    (* Return buffers to pool after encoding *)
    return_int_buf ctx buf_batch;
    return_int_buf ctx buf_out_dim;
    return_int_buf ctx buf_in_dim;
    return_int_buf ctx buf_act_type;
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
    let buf_lr = make_float_buf ctx lr in
    let buf_b1 = make_float_buf ctx b1 in
    let buf_b2 = make_float_buf ctx b2 in
    let buf_b1p = make_float_buf ctx b1p in
    let buf_b2p = make_float_buf ctx b2p in
    let buf_eps = make_float_buf ctx eps in
    let buf_wd = make_float_buf ctx wd in
    let buf_total = make_int_buf ctx total in
    ComputeCommandEncoder.set_buffer enc buf_lr ~index:4; 
    ComputeCommandEncoder.set_buffer enc buf_b1 ~index:5; 
    ComputeCommandEncoder.set_buffer enc buf_b2 ~index:6; 
    ComputeCommandEncoder.set_buffer enc buf_b1p ~index:7; 
    ComputeCommandEncoder.set_buffer enc buf_b2p ~index:8; 
    ComputeCommandEncoder.set_buffer enc buf_eps ~index:9; 
    ComputeCommandEncoder.set_buffer enc buf_wd ~index:10;
    ComputeCommandEncoder.set_buffer enc buf_total ~index:11;
    ComputeCommandEncoder.dispatch_threadgroups enc ~threadgroups_per_grid:{width=(total+1023)/1024; height=1; depth=1} ~threads_per_threadgroup:{width=1024; height=1; depth=1};
    ComputeCommandEncoder.end_encoding enc;
    increment_command_count ctx;
    return_float_buf ctx buf_lr;
    return_float_buf ctx buf_b1;
    return_float_buf ctx buf_b2;
    return_float_buf ctx buf_b1p;
    return_float_buf ctx buf_b2p;
    return_float_buf ctx buf_eps;
    return_float_buf ctx buf_wd;
    return_int_buf ctx buf_total

let mse_grad p t scale =
    let ctx = get_ctx () in 
    let total = p.rows * p.cols in
    let res_b = get_buffer ctx (total * 4) in
    let enc = ComputeCommandEncoder.on_buffer (get_cb ctx) in
    ComputeCommandEncoder.set_compute_pipeline_state enc (Hashtbl.find ctx.pipelines "mse_grad");
    ComputeCommandEncoder.set_buffer enc p.buffer ~index:0; 
    ComputeCommandEncoder.set_buffer enc t.buffer ~index:1; 
    ComputeCommandEncoder.set_buffer enc res_b ~index:2;
    let buf_scale = make_float_buf ctx scale in
    let buf_total = make_int_buf ctx total in
    ComputeCommandEncoder.set_buffer enc buf_scale ~index:3;
    ComputeCommandEncoder.set_buffer enc buf_total ~index:4;
    ComputeCommandEncoder.dispatch_threadgroups enc ~threadgroups_per_grid:{width=(total+1023)/1024; height=1; depth=1} ~threads_per_threadgroup:{width=1024; height=1; depth=1};
    ComputeCommandEncoder.end_encoding enc; 
    increment_command_count ctx;
    return_float_buf ctx buf_scale;
    return_int_buf ctx buf_total;
    { buffer = res_b; rows = p.rows; cols = p.cols; released = false }

let copy_inplace src dst = 
    let ctx = get_ctx () in 
    let enc = BlitCommandEncoder.on_buffer (get_cb ctx) in 
    BlitCommandEncoder.copy_from_buffer enc ~source_buffer:src.buffer ~source_offset:0 ~destination_buffer:dst.buffer ~destination_offset:0 ~size:(src.rows*src.cols*4); 
    BlitCommandEncoder.end_encoding enc;
    increment_command_count ctx

let matmul a b =
    let ctx = get_ctx () in 
    let m, k, n = a.rows, a.cols, b.cols in 
    let out_b = get_buffer ctx (m * n * 4) in
    let enc = ComputeCommandEncoder.on_buffer (get_cb ctx) in
    ComputeCommandEncoder.set_compute_pipeline_state enc (Hashtbl.find ctx.pipelines "matmul");
    ComputeCommandEncoder.set_buffer enc a.buffer ~index:0; 
    ComputeCommandEncoder.set_buffer enc b.buffer ~index:1; 
    ComputeCommandEncoder.set_buffer enc out_b ~index:2;
    let buf_m = make_int_buf ctx m in
    let buf_n = make_int_buf ctx n in
    let buf_k = make_int_buf ctx k in
    ComputeCommandEncoder.set_buffer enc buf_m ~index:3; 
    ComputeCommandEncoder.set_buffer enc buf_n ~index:4; 
    ComputeCommandEncoder.set_buffer enc buf_k ~index:5;
    ComputeCommandEncoder.dispatch_threadgroups enc ~threadgroups_per_grid:{width=(n+31)/32; height=(m+31)/32; depth=1} ~threads_per_threadgroup:{width=32; height=32; depth=1};
    ComputeCommandEncoder.end_encoding enc; 
    increment_command_count ctx;
    return_int_buf ctx buf_m;
    return_int_buf ctx buf_n;
    return_int_buf ctx buf_k;
    { buffer = out_b; rows = m; cols = n; released = false }

let add a b =
    let ctx = get_ctx () in 
    let m, n = a.rows, a.cols in 
    let total = m * n in
    let out_b = get_buffer ctx (total * 4) in
    let enc = ComputeCommandEncoder.on_buffer (get_cb ctx) in
    ComputeCommandEncoder.set_compute_pipeline_state enc (Hashtbl.find ctx.pipelines "mat_add");
    ComputeCommandEncoder.set_buffer enc a.buffer ~index:0; 
    ComputeCommandEncoder.set_buffer enc b.buffer ~index:1; 
    ComputeCommandEncoder.set_buffer enc out_b ~index:2;
    let buf_total = make_int_buf ctx total in
    ComputeCommandEncoder.set_buffer enc buf_total ~index:3;
    ComputeCommandEncoder.dispatch_threadgroups enc ~threadgroups_per_grid:{width=(total+1023)/1024; height=1; depth=1} ~threads_per_threadgroup:{width=1024; height=1; depth=1};
    ComputeCommandEncoder.end_encoding enc; 
    increment_command_count ctx;
    return_int_buf ctx buf_total;
    { buffer = out_b; rows = m; cols = n; released = false }

let relu x = 
    let ctx = get_ctx () in 
    let total = x.rows * x.cols in
    let enc = ComputeCommandEncoder.on_buffer (get_cb ctx) in 
    ComputeCommandEncoder.set_compute_pipeline_state enc (Hashtbl.find ctx.pipelines "relu_fwd"); 
    ComputeCommandEncoder.set_buffer enc x.buffer ~index:0; 
    let buf_total = make_int_buf ctx total in
    ComputeCommandEncoder.set_buffer enc buf_total ~index:1;
    ComputeCommandEncoder.dispatch_threadgroups enc ~threadgroups_per_grid:{width=(total+1023)/1024; height=1; depth=1} ~threads_per_threadgroup:{width=1024; height=1; depth=1}; 
    ComputeCommandEncoder.end_encoding enc; 
    increment_command_count ctx;
    return_int_buf ctx buf_total;
    x

let sigmoid x = 
    let ctx = get_ctx () in 
    let total = x.rows * x.cols in
    let enc = ComputeCommandEncoder.on_buffer (get_cb ctx) in 
    ComputeCommandEncoder.set_compute_pipeline_state enc (Hashtbl.find ctx.pipelines "sigmoid_fwd"); 
    ComputeCommandEncoder.set_buffer enc x.buffer ~index:0; 
    let buf_total = make_int_buf ctx total in
    ComputeCommandEncoder.set_buffer enc buf_total ~index:1;
    ComputeCommandEncoder.dispatch_threadgroups enc ~threadgroups_per_grid:{width=(total+1023)/1024; height=1; depth=1} ~threads_per_threadgroup:{width=1024; height=1; depth=1}; 
    ComputeCommandEncoder.end_encoding enc; 
    increment_command_count ctx;
    return_int_buf ctx buf_total;
    x

let tanh x = 
    let ctx = get_ctx () in 
    let total = x.rows * x.cols in
    let enc = ComputeCommandEncoder.on_buffer (get_cb ctx) in 
    ComputeCommandEncoder.set_compute_pipeline_state enc (Hashtbl.find ctx.pipelines "tanh_fwd"); 
    ComputeCommandEncoder.set_buffer enc x.buffer ~index:0; 
    let buf_total = make_int_buf ctx total in
    ComputeCommandEncoder.set_buffer enc buf_total ~index:1;
    ComputeCommandEncoder.dispatch_threadgroups enc ~threadgroups_per_grid:{width=(total+1023)/1024; height=1; depth=1} ~threads_per_threadgroup:{width=1024; height=1; depth=1}; 
    ComputeCommandEncoder.end_encoding enc; 
    increment_command_count ctx;
    return_int_buf ctx buf_total;
    x

let zero_tensor x =
    let ctx = get_ctx () in 
    let total = x.rows * x.cols in
    let enc = ComputeCommandEncoder.on_buffer (get_cb ctx) in 
    ComputeCommandEncoder.set_compute_pipeline_state enc (Hashtbl.find ctx.pipelines "zero_buf"); 
    ComputeCommandEncoder.set_buffer enc x.buffer ~index:0; 
    let buf_total = make_int_buf ctx total in
    ComputeCommandEncoder.set_buffer enc buf_total ~index:1;
    ComputeCommandEncoder.dispatch_threadgroups enc ~threadgroups_per_grid:{width=(total+1023)/1024; height=1; depth=1} ~threads_per_threadgroup:{width=1024; height=1; depth=1}; 
    ComputeCommandEncoder.end_encoding enc; 
    increment_command_count ctx;
    return_int_buf ctx buf_total

let transpose a =
    let ctx = get_ctx () in 
    let out_b = get_buffer ctx (a.rows * a.cols * 4) in
    let enc = ComputeCommandEncoder.on_buffer (get_cb ctx) in
    ComputeCommandEncoder.set_compute_pipeline_state enc (Hashtbl.find ctx.pipelines "mat_transpose");
    ComputeCommandEncoder.set_buffer enc a.buffer ~index:0; 
    ComputeCommandEncoder.set_buffer enc out_b ~index:1;
    let buf_rows = make_int_buf ctx a.rows in
    let buf_cols = make_int_buf ctx a.cols in
    ComputeCommandEncoder.set_buffer enc buf_rows ~index:2; 
    ComputeCommandEncoder.set_buffer enc buf_cols ~index:3;
    ComputeCommandEncoder.dispatch_threadgroups enc ~threadgroups_per_grid:{width=(a.cols+31)/32; height=(a.rows+31)/32; depth=1} ~threads_per_threadgroup:{width=32; height=32; depth=1};
    ComputeCommandEncoder.end_encoding enc; 
    increment_command_count ctx;
    return_int_buf ctx buf_rows;
    return_int_buf ctx buf_cols;
    { buffer = out_b; rows = a.cols; cols = a.rows; released = false }

let add_bias x bias =
    let ctx = get_ctx () in 
    let enc = ComputeCommandEncoder.on_buffer (get_cb ctx) in
    ComputeCommandEncoder.set_compute_pipeline_state enc (Hashtbl.find ctx.pipelines "add_bias");
    ComputeCommandEncoder.set_buffer enc x.buffer ~index:0; 
    ComputeCommandEncoder.set_buffer enc bias.buffer ~index:1;
    let buf_cols = make_int_buf ctx x.cols in
    ComputeCommandEncoder.set_buffer enc buf_cols ~index:2;
    ComputeCommandEncoder.dispatch_threadgroups enc ~threadgroups_per_grid:{width=(x.cols+31)/32; height=(x.rows+31)/32; depth=1} ~threads_per_threadgroup:{width=32; height=32; depth=1};
    ComputeCommandEncoder.end_encoding enc; 
    increment_command_count ctx;
    return_int_buf ctx buf_cols;
    x

let conv2d _ _ _ = failwith "Use fused kernels"

let update_confusion_matrix preds targets cm =
    let ctx = get_ctx () in
    let n = preds.rows in
    let c = preds.cols in
    (* Validate dimensions *)
    if targets.rows <> n || targets.cols <> c then failwith "Gpu.update_confusion_matrix: Dimension mismatch";
    if cm.rows <> c || cm.cols <> c then failwith "Gpu.update_confusion_matrix: CM dimension mismatch";
    
    let enc = ComputeCommandEncoder.on_buffer (get_cb ctx) in
    ComputeCommandEncoder.set_compute_pipeline_state enc (Hashtbl.find ctx.pipelines "confusion_matrix_update");
    ComputeCommandEncoder.set_buffer enc preds.buffer ~index:0;
    ComputeCommandEncoder.set_buffer enc targets.buffer ~index:1;
    ComputeCommandEncoder.set_buffer enc cm.buffer ~index:2;
    let buf_n = make_int_buf ctx n in
    let buf_c = make_int_buf ctx c in
    ComputeCommandEncoder.set_buffer enc buf_n ~index:3;
    ComputeCommandEncoder.set_buffer enc buf_c ~index:4;
    ComputeCommandEncoder.dispatch_threadgroups enc 
        ~threadgroups_per_grid:{width=(n+1023)/1024; height=1; depth=1} 
        ~threads_per_threadgroup:{width=1024; height=1; depth=1};
    ComputeCommandEncoder.end_encoding enc;
    increment_command_count ctx;
    return_int_buf ctx buf_n;
    return_int_buf ctx buf_c

let cm_to_float_tensor cm =
    let ctx = get_ctx () in
    let len = cm.rows * cm.cols in
    let out_b = get_buffer ctx (len * 4) in
    let enc = ComputeCommandEncoder.on_buffer (get_cb ctx) in
    ComputeCommandEncoder.set_compute_pipeline_state enc (Hashtbl.find ctx.pipelines "cm_to_float");
    ComputeCommandEncoder.set_buffer enc cm.buffer ~index:0;
    ComputeCommandEncoder.set_buffer enc out_b ~index:1;
    let buf_len = make_int_buf ctx len in
    ComputeCommandEncoder.set_buffer enc buf_len ~index:2;
    ComputeCommandEncoder.dispatch_threadgroups enc
        ~threadgroups_per_grid:{width=(len+1023)/1024; height=1; depth=1}
        ~threads_per_threadgroup:{width=1024; height=1; depth=1};
    ComputeCommandEncoder.end_encoding enc;
    increment_command_count ctx;
    return_int_buf ctx buf_len;
    { buffer = out_b; rows = cm.rows; cols = cm.cols; released = false }

let release (t : tensor) =
  if not t.released then begin
    t.released <- true;
    match !ctx_ref with
    | Some ctx -> release_buffer ctx t.buffer
    | None -> ()
  end

(* Cleanup function to release all resources and clear buffer pools *)
let cleanup () =
  (* Sync any pending commands *)
  sync ();
  (* Clear buffer pools *)
  match !ctx_ref with
  | Some ctx ->
      Queue.clear ctx.int_buffer_pool;
      Queue.clear ctx.float_buffer_pool;
      Hashtbl.clear ctx.buffer_cache;
      ctx.commands_in_buffer <- 0
  | None -> ()