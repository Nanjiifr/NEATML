type t = {
  mutable kernels : Tensor.t;
  mutable bias : Tensor.t;
  mutable grad_weights : Tensor.t;
  mutable grad_bias : Tensor.t;
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
val forward : t -> Tensor.t -> Tensor.t
val zero_grad : t -> unit
val backward : t -> Tensor.t -> Gradients.t