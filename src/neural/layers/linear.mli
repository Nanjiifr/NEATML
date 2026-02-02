type t = {
  weights : Tensor.t;
  bias : Tensor.t;
  mutable grad_weights : Tensor.t;
  mutable grad_bias : Tensor.t;
  mutable input_cache : Tensor.t option;
  activation : Activations.t;
  mutable preact_cache : Tensor.t option;
}

val create_bias_tensor : t -> int -> Tensor.t
val create : int -> int -> int -> Activations.t -> t
val forward : t -> Tensor.t -> Tensor.t
val zero_grad : t -> unit
val backward : t -> Tensor.t -> Gradients.t