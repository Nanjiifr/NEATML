open Metal
open Ctypes

(* Shader Source Code *)
let shader_source = "
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
    for (int k = 0; k < K; k++) {
        sum += A[id.y * K + k] * B[k * N + id.x];
    }
    C[id.y * N + id.x] = sum;
}

kernel void relu_activation(
    device const float *in [[buffer(0)]],
    device float *out [[buffer(1)]],
    uint id [[thread_position_in_grid]])
{
    out[id] = max(0.0, in[id]);
}

kernel void sigmoid_activation(
    device const float *in [[buffer(0)]],
    device float *out [[buffer(1)]],
    uint id [[thread_position_in_grid]])
{
    float x = in[id];
    if (x >= 0.0) {
        float z = exp(-x);
        out[id] = 1.0 / (1.0 + z);
    } else {
        float z = exp(x);
        out[id] = z / (1.0 + z);
    }
}

kernel void tanh_activation(
    device const float *in [[buffer(0)]],
    device float *out [[buffer(1)]],
    uint id [[thread_position_in_grid]])
{
    out[id] = tanh(in[id]);
}

kernel void mat_add(
    device const float *A [[buffer(0)]],
    device const float *B [[buffer(1)]],
    device float *C [[buffer(2)]],
    uint id [[thread_position_in_grid]])
{
    C[id] = A[id] + B[id];
}

kernel void conv2d_valid(
    device const float *input [[buffer(0)]],
    device const float *kernel_weights [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant int &in_h [[buffer(3)]],
    constant int &in_w [[buffer(4)]],
    constant int &k_h [[buffer(5)]],
    constant int &k_w [[buffer(6)]],
    constant int &out_h [[buffer(7)]],
    constant int &out_w [[buffer(8)]],
    uint2 id [[thread_position_in_grid]])
{
    if (id.x >= uint(out_w) || id.y >= uint(out_h)) return;

    float sum = 0.0;
    for (int r = 0; r < k_h; r++) {
        for (int c = 0; c < k_w; c++) {
            sum += input[(id.y + r) * in_w + (id.x + c)] * kernel_weights[r * k_w + c];
        }
    }
    output[id.y * out_w + id.x] = sum;
}

kernel void conv2d_full(
    device const float *input [[buffer(0)]],
    device const float *kernel_weights [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant int &in_h [[buffer(3)]],
    constant int &in_w [[buffer(4)]],
    constant int &k_h [[buffer(5)]],
    constant int &k_w [[buffer(6)]],
    constant int &out_h [[buffer(7)]],
    constant int &out_w [[buffer(8)]],
    uint2 id [[thread_position_in_grid]])
{
    // Output size for full conv is (in_h + k_h - 1) x (in_w + k_w - 1)
    if (id.x >= uint(out_w) || id.y >= uint(out_h)) return;

    float sum = 0.0;
    
    // We need to compute sum over kernel positions that overlap with input
    // The relation for full convolution can be seen as:
    // output[i, j] = sum_{m, n} input[m, n] * kernel[i - m, j - n]
    // where indices are valid.
    
    for (int r = 0; r < k_h; r++) {
        for (int c = 0; c < k_w; c++) {
            int input_r = id.y - r;
            int input_c = id.x - c;
            
            if (input_r >= 0 && input_r < in_h && input_c >= 0 && input_c < in_w) {
                sum += input[input_r * in_w + input_c] * kernel_weights[r * k_w + c];
            }
        }
    }
    output[id.y * out_w + id.x] = sum;
}
"

(* Singleton Context *)
type context = {
  device : Device.t;
  queue : CommandQueue.t;
  matmul_pipeline : ComputePipelineState.t;
  relu_pipeline : ComputePipelineState.t;
  sigmoid_pipeline : ComputePipelineState.t;
  tanh_pipeline : ComputePipelineState.t;
  add_pipeline : ComputePipelineState.t;
  conv_valid_pipeline : ComputePipelineState.t;
  conv_full_pipeline : ComputePipelineState.t;
}

let ctx_ref = ref None

let get_ctx () =
  match !ctx_ref with
  | Some ctx -> ctx
  | None ->
      let device = Device.create_system_default () in
      let queue = CommandQueue.on_device device in
      let library = Device.new_library_with_source device shader_source in
      
      let make_pipeline name =
        let func = Library.make_function library name in
        ComputePipelineState.create device func
      in

      let ctx = {
        device;
        queue;
        matmul_pipeline = make_pipeline "matmul";
        relu_pipeline = make_pipeline "relu_activation";
        sigmoid_pipeline = make_pipeline "sigmoid_activation";
        tanh_pipeline = make_pipeline "tanh_activation";
        add_pipeline = make_pipeline "mat_add";
        conv_valid_pipeline = make_pipeline "conv2d_valid";
        conv_full_pipeline = make_pipeline "conv2d_full";
      } in
      ctx_ref := Some ctx;
      ctx

(* Helpers *)
let float_size = 4 (* sizeof(float) *)

let buffer_from_tensor ctx (t : Tensor.t) =
  let rows = Array.length t in
  let cols = if rows > 0 then Array.length t.(0) else 0 in
  let size = rows * cols * float_size in
  let buffer = Buffer.on_device ctx.device ~length:size ResourceOptions.storage_mode_shared in
  let ptr = Buffer.contents buffer in
  let flat_idx = ref 0 in
  for i = 0 to rows - 1 do
    let row = t.(i) in
    for j = 0 to cols - 1 do
      Ctypes.set_float (Ctypes.ptr_add ptr (!flat_idx * float_size)) row.(j);
      incr flat_idx
    done
  done;
  buffer

let tensor_from_buffer buffer rows cols =
  let t = Array.make_matrix rows cols 0.0 in
  let ptr = Buffer.contents buffer in
  let flat_idx = ref 0 in
  for i = 0 to rows - 1 do
    let row = t.(i) in
    for j = 0 to cols - 1 do
      row.(j) <- Ctypes.get_float (Ctypes.ptr_add ptr (!flat_idx * float_size));
      incr flat_idx
    done
  done;
  t

(* Operations *)

let matmul a b =
  let ctx = get_ctx () in
  let m = Array.length a in
  let k = Array.length a.(0) in
  assert (k = Array.length b);
  let n = Array.length b.(0) in
  
  let buf_a = buffer_from_tensor ctx a in
  let buf_b = buffer_from_tensor ctx b in
  let buf_c = Buffer.on_device ctx.device ~length:(m * n * float_size) ResourceOptions.storage_mode_shared in
  
  let command_buffer = CommandQueue.make_command_buffer ctx.queue in
  let encoder = CommandBuffer.make_compute_command_encoder command_buffer in
  
  ComputeCommandEncoder.set_compute_pipeline_state encoder ctx.matmul_pipeline;
  ComputeCommandEncoder.set_buffer encoder ~index:0 buf_a;
  ComputeCommandEncoder.set_buffer encoder ~index:1 buf_b;
  ComputeCommandEncoder.set_buffer encoder ~index:2 buf_c;
  
  (* Pass dimensions as constants. We need buffers for them too or use set_bytes if supported. 
     Using buffers for simplicity with ctypes. *)
  let int_size = 4 in (* sizeof(int) *)
  let make_int_buf v = 
      let b = Buffer.on_device ctx.device ~length:int_size ResourceOptions.storage_mode_shared in
      let p = Buffer.contents b in
      Ctypes.set_int (Ctypes.ptr_add p 0) v; (* Ctypes.set_int usually works for 4 bytes on 64bit if int is 32bit? OCaml int is 63bit. 
                                                Need to be careful. Metal 'int' is 32 bit. 
                                                Ctypes.set_int writes native int? 
                                                Let's assume standard 32-bit int for GPU interop context or use specific type.
                                                Actually, safer to use Ctypes.Int32 *)
      b
  in
  (* Wait, ctypes set_int writes system int. On 64-bit system, that might be 64-bit. Metal expects 32-bit int.
     Let's verify Ctypes. 
     I'll use `Ctypes.set_int32` if available or `Signed.Int32`? 
     Standard `Ctypes` defines `int` as matching C `int`. Usually 32-bit even on 64-bit Unix. 
     Let's assume it's fine. *)
     
  (* Actually, to be safe, I'll use set_int with a small buffer. *)
  let buf_m = make_int_buf m in
  let buf_n = make_int_buf n in
  let buf_k = make_int_buf k in
  
  ComputeCommandEncoder.set_buffer encoder ~index:3 buf_m;
  ComputeCommandEncoder.set_buffer encoder ~index:4 buf_n;
  ComputeCommandEncoder.set_buffer encoder ~index:5 buf_k;

  let w = ComputePipelineState.max_total_threads_per_threadgroup ctx.matmul_pipeline in
  (* Simple 1D threadgroup size or 2D? API usually takes 3 args for width, height, depth *)
  (* Let's use 32x32 block if possible, or whatever fits. 32*32 = 1024. *)
  let th_w = 32 in
  let th_h = 32 in
  let thread_group_size = ComputeCommandEncoder.create_size ~width:th_w ~height:th_h ~depth:1 in
  let grid_size = ComputeCommandEncoder.create_size ~width:n ~height:m ~depth:1 in
  
  ComputeCommandEncoder.dispatch_threads encoder grid_size thread_group_size;
  ComputeCommandEncoder.end_encoding encoder;
  CommandBuffer.commit command_buffer;
  CommandBuffer.wait_until_completed command_buffer;
  
  tensor_from_buffer buf_c m n


let conv2d mode input kernel =
  let ctx = get_ctx () in
  let in_h = Array.length input in
  let in_w = Array.length input.(0) in
  let k_h = Array.length kernel in
  let k_w = Array.length kernel.(0) in
  
  let out_h, out_w, pipeline = 
      if mode = "valid" then 
          (in_h - k_h + 1, in_w - k_w + 1, ctx.conv_valid_pipeline)
      else if mode = "full" then
          (in_h + k_h - 1, in_w + k_w - 1, ctx.conv_full_pipeline)
      else failwith "Gpu.conv2d: unknown mode"
  in
  
  let buf_in = buffer_from_tensor ctx input in
  let buf_k = buffer_from_tensor ctx kernel in
  let buf_out = Buffer.on_device ctx.device ~length:(out_h * out_w * float_size) ResourceOptions.storage_mode_shared in
  
  let command_buffer = CommandQueue.make_command_buffer ctx.queue in
  let encoder = CommandBuffer.make_compute_command_encoder command_buffer in
  
  ComputeCommandEncoder.set_compute_pipeline_state encoder pipeline;
  ComputeCommandEncoder.set_buffer encoder ~index:0 buf_in;
  ComputeCommandEncoder.set_buffer encoder ~index:1 buf_k;
  ComputeCommandEncoder.set_buffer encoder ~index:2 buf_out;
  
  let make_int_buf v = 
      let b = Buffer.on_device ctx.device ~length:4 ResourceOptions.storage_mode_shared in
      let p = Buffer.contents b in
      Ctypes.set_int (Ctypes.ptr_add p 0) v;
      b
  in
  
  ComputeCommandEncoder.set_buffer encoder ~index:3 (make_int_buf in_h);
  ComputeCommandEncoder.set_buffer encoder ~index:4 (make_int_buf in_w);
  ComputeCommandEncoder.set_buffer encoder ~index:5 (make_int_buf k_h);
  ComputeCommandEncoder.set_buffer encoder ~index:6 (make_int_buf k_w);
  ComputeCommandEncoder.set_buffer encoder ~index:7 (make_int_buf out_h);
  ComputeCommandEncoder.set_buffer encoder ~index:8 (make_int_buf out_w);
  
  let th_w = 32 in
  let th_h = 32 in
  let thread_group_size = ComputeCommandEncoder.create_size ~width:th_w ~height:th_h ~depth:1 in
  let grid_size = ComputeCommandEncoder.create_size ~width:out_w ~height:out_h ~depth:1 in
  
  ComputeCommandEncoder.dispatch_threads encoder grid_size thread_group_size;
  ComputeCommandEncoder.end_encoding encoder;
  CommandBuffer.commit command_buffer;
  CommandBuffer.wait_until_completed command_buffer;
  
  tensor_from_buffer buf_out out_h out_w

let add a b = 
  let ctx = get_ctx () in
  let m = Array.length a in
  let n = Array.length a.(0) in
  (* Assume same shape *)
  
  let buf_a = buffer_from_tensor ctx a in
  let buf_b = buffer_from_tensor ctx b in
  let buf_c = Buffer.on_device ctx.device ~length:(m * n * float_size) ResourceOptions.storage_mode_shared in
  
  let command_buffer = CommandQueue.make_command_buffer ctx.queue in
  let encoder = CommandBuffer.make_compute_command_encoder command_buffer in
  
  ComputeCommandEncoder.set_compute_pipeline_state encoder ctx.add_pipeline;
  ComputeCommandEncoder.set_buffer encoder ~index:0 buf_a;
  ComputeCommandEncoder.set_buffer encoder ~index:1 buf_b;
  ComputeCommandEncoder.set_buffer encoder ~index:2 buf_c;
  
  let total_elements = m * n in
  let thread_group_size = ComputeCommandEncoder.create_size ~width:1024 ~height:1 ~depth:1 in
  let grid_size = ComputeCommandEncoder.create_size ~width:total_elements ~height:1 ~depth:1 in
  
  ComputeCommandEncoder.dispatch_threads encoder grid_size thread_group_size;
  ComputeCommandEncoder.end_encoding encoder;
  CommandBuffer.commit command_buffer;
  CommandBuffer.wait_until_completed command_buffer;
  
  tensor_from_buffer buf_c m n

let activate pipeline_selector input =
  let ctx = get_ctx () in
  let m = Array.length input in
  let n = Array.length input.(0) in
  
  let buf_in = buffer_from_tensor ctx input in
  let buf_out = Buffer.on_device ctx.device ~length:(m * n * float_size) ResourceOptions.storage_mode_shared in
  
  let pipeline = pipeline_selector ctx in
  
  let command_buffer = CommandQueue.make_command_buffer ctx.queue in
  let encoder = CommandBuffer.make_compute_command_encoder command_buffer in
  
  ComputeCommandEncoder.set_compute_pipeline_state encoder pipeline;
  ComputeCommandEncoder.set_buffer encoder ~index:0 buf_in;
  ComputeCommandEncoder.set_buffer encoder ~index:1 buf_out;
  
  let total_elements = m * n in
  let thread_group_size = ComputeCommandEncoder.create_size ~width:1024 ~height:1 ~depth:1 in
  let grid_size = ComputeCommandEncoder.create_size ~width:total_elements ~height:1 ~depth:1 in
  
  ComputeCommandEncoder.dispatch_threads encoder grid_size thread_group_size;
  ComputeCommandEncoder.end_encoding encoder;
  CommandBuffer.commit command_buffer;
  CommandBuffer.wait_until_completed command_buffer;
  
  tensor_from_buffer buf_out m n

let relu = activate (fun ctx -> ctx.relu_pipeline)
let sigmoid = activate (fun ctx -> ctx.sigmoid_pipeline)
let tanh = activate (fun ctx -> ctx.tanh_pipeline)
