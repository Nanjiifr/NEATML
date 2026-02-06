

type t = {
  kernel_size : int;
  stride : int;
  input_depth : int;
  input_height : int;
  input_width : int;
  mutable input_cache : Tensor.t option;
  mutable max_indices : (int array array) option;
}

val create : int -> int -> int -> int -> int -> t
val forward : t -> Tensor.t -> Tensor.t
val backward : t -> Tensor.t -> Gradients.t
val get_output_dims : t -> int * int * int
