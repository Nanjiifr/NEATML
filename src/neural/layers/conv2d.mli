type t = {
  kernels : Tensor.t array array;
  mutable bias : float array;
  mutable grad_weights : Tensor.t array array;
  mutable grad_bias : float array;
  mutable input_cache : Tensor.t option;
  activation : Activations.t;
  mutable preact_cache : Tensor.t option;
  input_depth : int;
  input_height : int;
  input_width : int;
  kernel_size : int;
  output_depth : int;
}

val create : int -> int -> int -> int -> int -> Activations.t -> t
val zero_grad : t -> unit
val forward : t -> Tensor.t -> Tensor.t
val backward : t -> Tensor.t -> Gradients.t
