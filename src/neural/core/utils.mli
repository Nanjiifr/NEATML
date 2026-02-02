val use_gpu : bool ref
val enable_gpu : unit -> unit
val disable_gpu : unit -> unit

val rows : Tensor.t -> int
val cols : Tensor.t -> int
val zeros : int -> int -> Tensor.t
val copy_mat : Tensor.t -> Tensor.t
val copy_mat_inplace : Tensor.t -> Tensor.t -> unit
val map_mat : (float -> int -> int -> float) -> Tensor.t -> Tensor.t
val sum_column : Tensor.t -> int -> float
val map2_mat : (float -> float -> float) -> Tensor.t -> Tensor.t -> Tensor.t
val transpose : Tensor.t -> Tensor.t
val scalar : float -> Tensor.t -> Tensor.t
val multiply_matrix : Tensor.t -> Tensor.t -> Tensor.t
val add_matrices : Tensor.t -> Tensor.t -> Tensor.t
val iter_matrix : (float -> int -> int -> unit) -> Tensor.t -> unit
val list_iter3 : ('a -> 'b -> 'c -> unit) -> 'a list -> 'b list -> 'c list -> unit
val conv : string -> Tensor.t -> Tensor.t -> Tensor.t

val to_cpu : Tensor.t -> float array array
val to_gpu : Tensor.t -> Tensor.t