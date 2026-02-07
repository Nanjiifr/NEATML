(* 
 * OCaml bindings for MPS (Metal Performance Shaders) stubs
 * Uses Ctypes.Foreign to bind to C functions
 *)

open Ctypes
open Foreign

(* ============================================================================
   Type Definitions
   ============================================================================ *)

type mps_device = unit ptr
let mps_device : mps_device typ = ptr void

type mps_command_buffer = unit ptr
let mps_command_buffer : mps_command_buffer typ = ptr void

type mps_matrix = unit ptr
let mps_matrix : mps_matrix typ = ptr void

type mps_conv_descriptor = unit ptr
let mps_conv_descriptor : mps_conv_descriptor typ = ptr void

type mps_pool_descriptor = unit ptr
let mps_pool_descriptor : mps_pool_descriptor typ = ptr void

(* Our tensor type matching gpu.mli *)
type tensor = {
  matrix : mps_matrix;
  rows : int;
  cols : int;
  mutable released : bool;
}

(* ============================================================================
   Foreign Function Bindings
   ============================================================================ *)

(* Device management *)
let mps_device_create = 
  foreign "mps_device_create" (void @-> returning mps_device)

let mps_device_destroy = 
  foreign "mps_device_destroy" (mps_device @-> returning void)

(* Command buffer management *)
let mps_command_buffer_create = 
  foreign "mps_command_buffer_create" (mps_device @-> returning mps_command_buffer)

let mps_command_buffer_commit = 
  foreign "mps_command_buffer_commit" (mps_command_buffer @-> returning void)

let mps_command_buffer_wait_until_completed = 
  foreign "mps_command_buffer_wait_until_completed" (mps_command_buffer @-> returning void)

let mps_command_buffer_destroy = 
  foreign "mps_command_buffer_destroy" (mps_command_buffer @-> returning void)

(* Matrix operations *)
let mps_matrix_create = 
  foreign "mps_matrix_create" (mps_device @-> size_t @-> size_t @-> returning mps_matrix)

let mps_matrix_destroy = 
  foreign "mps_matrix_destroy" (mps_matrix @-> returning void)

let mps_matrix_set_data = 
  foreign "mps_matrix_set_data" (mps_matrix @-> ptr float @-> size_t @-> returning void)

let mps_matrix_get_data = 
  foreign "mps_matrix_get_data" (mps_matrix @-> ptr float @-> size_t @-> returning void)

let mps_matrix_rows = 
  foreign "mps_matrix_rows" (mps_matrix @-> returning size_t)

let mps_matrix_cols = 
  foreign "mps_matrix_cols" (mps_matrix @-> returning size_t)

(* Core operations *)
let mps_matmul = 
  foreign "mps_matmul" 
    (mps_command_buffer @-> mps_matrix @-> mps_matrix @-> mps_matrix @-> 
     float @-> float @-> returning void)

let mps_matrix_add = 
  foreign "mps_matrix_add"
    (mps_command_buffer @-> mps_matrix @-> mps_matrix @-> mps_matrix @-> returning void)

let mps_matrix_mul_elementwise = 
  foreign "mps_matrix_mul_elementwise"
    (mps_command_buffer @-> mps_matrix @-> mps_matrix @-> mps_matrix @-> returning void)

let mps_matrix_transpose = 
  foreign "mps_matrix_transpose"
    (mps_command_buffer @-> mps_matrix @-> mps_matrix @-> returning void)

(* Activation functions *)
let mps_relu_forward = 
  foreign "mps_relu_forward" (mps_command_buffer @-> mps_matrix @-> returning void)

let mps_sigmoid_forward = 
  foreign "mps_sigmoid_forward" (mps_command_buffer @-> mps_matrix @-> returning void)

let mps_tanh_forward = 
  foreign "mps_tanh_forward" (mps_command_buffer @-> mps_matrix @-> returning void)

(* Linear layer operations *)
let mps_linear_forward = 
  foreign "mps_linear_forward"
    (mps_command_buffer @-> mps_matrix @-> mps_matrix @-> mps_matrix @->
     mps_matrix @-> int @-> returning void)

let mps_linear_backward_weights = 
  foreign "mps_linear_backward_weights"
    (mps_command_buffer @-> mps_matrix @-> mps_matrix @->
     mps_matrix @-> mps_matrix @-> returning void)

let mps_linear_backward_input = 
  foreign "mps_linear_backward_input"
    (mps_command_buffer @-> mps_matrix @-> mps_matrix @->
     mps_matrix @-> returning void)

(* Convolution operations *)
let mps_conv_descriptor_create = 
  foreign "mps_conv_descriptor_create"
    (int @-> int @-> int @-> int @-> int @-> int @-> returning mps_conv_descriptor)

let mps_conv_descriptor_destroy = 
  foreign "mps_conv_descriptor_destroy" (mps_conv_descriptor @-> returning void)

let mps_conv2d_forward = 
  foreign "mps_conv2d_forward"
    (mps_command_buffer @-> mps_conv_descriptor @-> mps_matrix @-> mps_matrix @->
     mps_matrix @-> mps_matrix @-> int @-> returning void)

let mps_conv2d_backward_input = 
  foreign "mps_conv2d_backward_input"
    (mps_command_buffer @-> mps_conv_descriptor @-> mps_matrix @-> mps_matrix @->
     mps_matrix @-> int @-> returning void)

let mps_conv2d_backward_weights = 
  foreign "mps_conv2d_backward_weights"
    (mps_command_buffer @-> mps_conv_descriptor @-> mps_matrix @-> mps_matrix @->
     mps_matrix @-> mps_matrix @-> int @-> returning void)

(* Pooling operations *)
let mps_maxpool_descriptor_create = 
  foreign "mps_maxpool_descriptor_create"
    (int @-> int @-> int @-> int @-> returning mps_pool_descriptor)

let mps_pool_descriptor_destroy = 
  foreign "mps_pool_descriptor_destroy" (mps_pool_descriptor @-> returning void)

let mps_maxpool_forward = 
  foreign "mps_maxpool_forward"
    (mps_command_buffer @-> mps_pool_descriptor @-> mps_matrix @-> mps_matrix @->
     mps_matrix @-> int @-> returning void)

let mps_maxpool_backward = 
  foreign "mps_maxpool_backward"
    (mps_command_buffer @-> mps_pool_descriptor @-> mps_matrix @-> mps_matrix @->
     mps_matrix @-> int @-> returning void)

(* Optimizer operations *)
let mps_adam_step = 
  foreign "mps_adam_step"
    (mps_command_buffer @-> mps_matrix @-> mps_matrix @-> mps_matrix @-> mps_matrix @->
     float @-> float @-> float @-> float @-> float @-> float @-> float @-> returning void)

(* Utility operations *)
let mps_matrix_zero = 
  foreign "mps_matrix_zero" (mps_command_buffer @-> mps_matrix @-> returning void)

let mps_matrix_copy = 
  foreign "mps_matrix_copy" (mps_command_buffer @-> mps_matrix @-> mps_matrix @-> returning void)

let mps_mse_gradient = 
  foreign "mps_mse_gradient"
    (mps_command_buffer @-> mps_matrix @-> mps_matrix @-> mps_matrix @-> float @-> returning void)

(* Synchronization *)
let mps_synchronize = 
  foreign "mps_synchronize" (mps_device @-> returning void)

(* ============================================================================
   Global State
   ============================================================================ *)

let device = ref None
let current_cmd_buffer = ref None

let get_device () =
  match !device with
  | Some d -> d
  | None ->
      let d = mps_device_create () in
      device := Some d;
      d

let get_cmd_buffer () =
  match !current_cmd_buffer with
  | Some cb -> cb
  | None ->
      let cb = mps_command_buffer_create (get_device ()) in
      current_cmd_buffer := Some cb;
      cb

let commit_batch () =
  match !current_cmd_buffer with
  | Some cb ->
      mps_command_buffer_commit cb;
      mps_command_buffer_wait_until_completed cb;
      mps_command_buffer_destroy cb;
      current_cmd_buffer := None
  | None -> ()

let sync () = commit_batch ()

(* ============================================================================
   Tensor Operations - Matching gpu.mli Interface
   ============================================================================ *)

let of_cpu (data : float array array) : tensor =
  let rows = Array.length data in
  if rows = 0 then failwith "Empty array";
  let cols = Array.length data.(0) in
  
  let mat = mps_matrix_create (get_device ()) (Unsigned.Size_t.of_int rows) (Unsigned.Size_t.of_int cols) in
  
  (* Flatten data *)
  let flat_data = Array.init (rows * cols) (fun i ->
    let r = i / cols in
    let c = i mod cols in
    data.(r).(c)
  ) in
  
  let flat_ptr = CArray.of_list float (Array.to_list flat_data) in
  mps_matrix_set_data mat (CArray.start flat_ptr) (Unsigned.Size_t.of_int (rows * cols));
  
  { matrix = mat; rows; cols; released = false }

let to_cpu (t : tensor) : float array array =
  if t.released then failwith "Tensor already released";
  
  let size = t.rows * t.cols in
  let flat_data = CArray.make float size in
  mps_matrix_get_data t.matrix (CArray.start flat_data) (Unsigned.Size_t.of_int size);
  
  Array.init t.rows (fun r ->
    Array.init t.cols (fun c ->
      CArray.get flat_data (r * t.cols + c)
    )
  )

let copy_inplace (src : tensor) (dst : tensor) : unit =
  if src.released || dst.released then failwith "Tensor already released";
  if src.rows <> dst.rows || src.cols <> dst.cols then
    failwith "Dimension mismatch in copy_inplace";
  mps_matrix_copy (get_cmd_buffer ()) src.matrix dst.matrix

let zero_tensor (t : tensor) : unit =
  if t.released then failwith "Tensor already released";
  mps_matrix_zero (get_cmd_buffer ()) t.matrix

let release (t : tensor) : unit =
  if not t.released then begin
    mps_matrix_destroy t.matrix;
    t.released <- true
  end

let cleanup () =
  commit_batch ();
  match !device with
  | Some d ->
      mps_device_destroy d;
      device := None
  | None -> ()

(* ============================================================================
   Matrix Operations
   ============================================================================ *)

let matmul (a : tensor) (b : tensor) : tensor =
  if a.released || b.released then failwith "Tensor already released";
  if a.cols <> b.rows then failwith "Dimension mismatch in matmul";
  
  let c = mps_matrix_create (get_device ()) 
    (Unsigned.Size_t.of_int a.rows) 
    (Unsigned.Size_t.of_int b.cols) in
  
  mps_matmul (get_cmd_buffer ()) a.matrix b.matrix c 1.0 0.0;
  
  { matrix = c; rows = a.rows; cols = b.cols; released = false }

let add (a : tensor) (b : tensor) : tensor =
  if a.released || b.released then failwith "Tensor already released";
  if a.rows <> b.rows || a.cols <> b.cols then
    failwith "Dimension mismatch in add";
  
  let result = mps_matrix_create (get_device ())
    (Unsigned.Size_t.of_int a.rows)
    (Unsigned.Size_t.of_int a.cols) in
  
  mps_matrix_add (get_cmd_buffer ()) a.matrix b.matrix result;
  
  { matrix = result; rows = a.rows; cols = a.cols; released = false }

let mul (a : tensor) (b : tensor) : tensor =
  if a.released || b.released then failwith "Tensor already released";
  if a.rows <> b.rows || a.cols <> b.cols then
    failwith "Dimension mismatch in mul";
  
  let result = mps_matrix_create (get_device ())
    (Unsigned.Size_t.of_int a.rows)
    (Unsigned.Size_t.of_int a.cols) in
  
  mps_matrix_mul_elementwise (get_cmd_buffer ()) a.matrix b.matrix result;
  
  { matrix = result; rows = a.rows; cols = a.cols; released = false }

let transpose (t : tensor) : tensor =
  if t.released then failwith "Tensor already released";
  
  let result = mps_matrix_create (get_device ())
    (Unsigned.Size_t.of_int t.cols)
    (Unsigned.Size_t.of_int t.rows) in
  
  mps_matrix_transpose (get_cmd_buffer ()) t.matrix result;
  
  { matrix = result; rows = t.cols; cols = t.rows; released = false }

(* ============================================================================
   Activation Functions
   ============================================================================ *)

let relu (x : tensor) : tensor =
  if x.released then failwith "Tensor already released";
  
  let result = mps_matrix_create (get_device ())
    (Unsigned.Size_t.of_int x.rows)
    (Unsigned.Size_t.of_int x.cols) in
  
  mps_matrix_copy (get_cmd_buffer ()) x.matrix result;
  mps_relu_forward (get_cmd_buffer ()) result;
  
  { matrix = result; rows = x.rows; cols = x.cols; released = false }

let sigmoid (x : tensor) : tensor =
  if x.released then failwith "Tensor already released";
  
  let result = mps_matrix_create (get_device ())
    (Unsigned.Size_t.of_int x.rows)
    (Unsigned.Size_t.of_int x.cols) in
  
  mps_matrix_copy (get_cmd_buffer ()) x.matrix result;
  mps_sigmoid_forward (get_cmd_buffer ()) result;
  
  { matrix = result; rows = x.rows; cols = x.cols; released = false }

let tanh (x : tensor) : tensor =
  if x.released then failwith "Tensor already released";
  
  let result = mps_matrix_create (get_device ())
    (Unsigned.Size_t.of_int x.rows)
    (Unsigned.Size_t.of_int x.cols) in
  
  mps_matrix_copy (get_cmd_buffer ()) x.matrix result;
  mps_tanh_forward (get_cmd_buffer ()) result;
  
  { matrix = result; rows = x.rows; cols = x.cols; released = false }

(* ============================================================================
   Linear Layer Operations
   ============================================================================ *)

let linear_fwd input weights bias batch in_dim out_dim activation =
  if input.released || weights.released then failwith "Tensor already released";
  
  let output = mps_matrix_create (get_device ())
    (Unsigned.Size_t.of_int batch)
    (Unsigned.Size_t.of_int out_dim) in
  
  let bias_mat = if bias.cols > 0 && not bias.released then bias.matrix 
                 else coerce (ptr void) mps_matrix null in
  
  mps_linear_forward (get_cmd_buffer ()) input.matrix weights.matrix 
    bias_mat output activation;
  
  { matrix = output; rows = batch; cols = out_dim; released = false }

let linear_bwd grad_output input weights grad_weights grad_bias grad_input 
               batch in_dim out_dim activation =
  if grad_output.released || input.released || weights.released then 
    failwith "Tensor already released";
  
  (* Backward for weights and bias *)
  mps_linear_backward_weights (get_cmd_buffer ()) 
    grad_output.matrix input.matrix grad_weights.matrix grad_bias.matrix;
  
  (* Backward for input *)
  let gi = mps_matrix_create (get_device ())
    (Unsigned.Size_t.of_int batch)
    (Unsigned.Size_t.of_int in_dim) in
  
  mps_linear_backward_input (get_cmd_buffer ()) 
    grad_output.matrix weights.matrix gi;
  
  { matrix = gi; rows = batch; cols = in_dim; released = false }

(* ============================================================================
   Optimizer Operations
   ============================================================================ *)

let adam_step weights grads m v lr beta1 beta2 beta1_power beta2_power epsilon weight_decay =
  if weights.released || grads.released || m.released || v.released then
    failwith "Tensor already released";
  
  mps_adam_step (get_cmd_buffer ()) weights.matrix grads.matrix m.matrix v.matrix
    lr beta1 beta2 beta1_power beta2_power epsilon weight_decay

(* ============================================================================
   Loss Functions
   ============================================================================ *)

let mse_grad predictions targets scale =
  if predictions.released || targets.released then failwith "Tensor already released";
  if predictions.rows <> targets.rows || predictions.cols <> targets.cols then
    failwith "Dimension mismatch in mse_grad";
  
  let grad = mps_matrix_create (get_device ())
    (Unsigned.Size_t.of_int predictions.rows)
    (Unsigned.Size_t.of_int predictions.cols) in
  
  mps_mse_gradient (get_cmd_buffer ()) predictions.matrix targets.matrix grad scale;
  
  { matrix = grad; rows = predictions.rows; cols = predictions.cols; released = false }

(* ============================================================================
   Convolution Operations (Simplified stubs for API compatibility)
   ============================================================================ *)

let conv2d_direct_fwd input weights bias output n in_h in_w out_h out_w kh kw in_d out_d act =
  (* Conv2D requires MPSImage not MPSMatrix - not yet implemented *)
  failwith "conv2d_direct_fwd: Not implemented in MPS backend (requires MPSImage support)"

let conv2d_direct_bwd_input grad_output weights grad_input n in_h in_w out_h out_w kh kw in_d out_d =
  failwith "conv2d_direct_bwd_input: Not implemented in MPS backend (requires MPSImage support)"

let conv2d_direct_bwd_weights input grad_output grad_weights n in_h in_w out_h out_w kh kw in_d out_d =
  failwith "conv2d_direct_bwd_weights: Not implemented in MPS backend (requires MPSImage support)"

let conv2d_bias_bwd grad_output grad_bias n out_h out_w out_d =
  Printf.fprintf stderr "Warning: conv2d_bias_bwd not fully implemented in MPS backend\n%!"

let im2col input col_output n c h w out_h out_w kh kw =
  Printf.fprintf stderr "Warning: im2col not fully implemented in MPS backend\n%!"

let col2im col_output input n c h w out_h out_w kh kw =
  Printf.fprintf stderr "Warning: col2im not fully implemented in MPS backend\n%!"

let permute_nhwc_nchw input output n h w c =
  Printf.fprintf stderr "Warning: permute_nhwc_nchw not fully implemented in MPS backend\n%!"

let permute_nchw_nhwc input output n h w c =
  Printf.fprintf stderr "Warning: permute_nchw_nhwc not fully implemented in MPS backend\n%!"

(* ============================================================================
   Pooling Operations
   ============================================================================ *)

let maxpool_fwd input n c in_h in_w out_h out_w kh kw stride =
  if input.released then failwith "Tensor already released";
  
  let output = mps_matrix_create (get_device ())
    (Unsigned.Size_t.of_int n)
    (Unsigned.Size_t.of_int (c * out_h * out_w)) in
  
  let indices = mps_matrix_create (get_device ())
    (Unsigned.Size_t.of_int n)
    (Unsigned.Size_t.of_int (c * out_h * out_w)) in
  
  let desc = mps_maxpool_descriptor_create kh kw stride stride in
  
  mps_maxpool_forward (get_cmd_buffer ()) desc input.matrix output indices n;
  
  mps_pool_descriptor_destroy desc;
  
  ({ matrix = output; rows = n; cols = c * out_h * out_w; released = false },
   { matrix = indices; rows = n; cols = c * out_h * out_w; released = false })

let maxpool_bwd grad_output indices n c in_h in_w out_h out_w =
  if grad_output.released || indices.released then failwith "Tensor already released";
  
  let grad_input = mps_matrix_create (get_device ())
    (Unsigned.Size_t.of_int n)
    (Unsigned.Size_t.of_int (c * in_h * in_w)) in
  
  (* Simplified - full implementation needs proper descriptor *)
  Printf.fprintf stderr "Warning: maxpool_bwd uses simplified implementation\n%!";
  
  { matrix = grad_input; rows = n; cols = c * in_h * in_w; released = false }

(* ============================================================================
   Utility Operations  
   ============================================================================ *)

let add_bias output bias =
  if output.released || bias.released then failwith "Tensor already released";
  
  (* Simplified - should broadcast bias across batch *)
  let result = add output bias in
  result

let activation_bwd act_type grad_output activation_output =
  if grad_output.released || activation_output.released then 
    failwith "Tensor already released";
  
  (* Simplified stub *)
  Printf.fprintf stderr "Warning: activation_bwd not fully implemented in MPS backend\n%!";
  
  let result = mps_matrix_create (get_device ())
    (Unsigned.Size_t.of_int grad_output.rows)
    (Unsigned.Size_t.of_int grad_output.cols) in
  
  mps_matrix_copy (get_cmd_buffer ()) grad_output.matrix result;
  
  { matrix = result; rows = grad_output.rows; cols = grad_output.cols; released = false }

(* ============================================================================
   Confusion Matrix Operations (Stubs for compatibility)
   ============================================================================ *)

let update_confusion_matrix preds targets cm =
  Printf.fprintf stderr "Warning: update_confusion_matrix not implemented in MPS backend\n%!"

let cm_to_float_tensor cm =
  if cm.released then failwith "Tensor already released";
  
  (* Return a copy for now *)
  let result = mps_matrix_create (get_device ())
    (Unsigned.Size_t.of_int cm.rows)
    (Unsigned.Size_t.of_int cm.cols) in
  
  mps_matrix_copy (get_cmd_buffer ()) cm.matrix result;
  
  { matrix = result; rows = cm.rows; cols = cm.cols; released = false }
