open Metal

type tensor = { buffer : Buffer.t; rows : int; cols : int; mutable released : bool }

val of_cpu : float array array -> tensor
val to_cpu : tensor -> float array array
val copy_inplace : tensor -> tensor -> unit
val zero_tensor : tensor -> unit
val release : tensor -> unit
val sync : unit -> unit
val commit_batch : unit -> unit
val cleanup : unit -> unit

val linear_fwd : tensor -> tensor -> tensor -> int -> int -> int -> int -> tensor
val linear_bwd : tensor -> tensor -> tensor -> tensor -> tensor -> tensor -> int -> int -> int -> int -> tensor
val adam_step : tensor -> tensor -> tensor -> tensor -> float -> float -> float -> float -> float -> float -> float -> unit
val mse_grad : tensor -> tensor -> float -> tensor

val matmul : tensor -> tensor -> tensor
val add : tensor -> tensor -> tensor
val relu : tensor -> tensor
val sigmoid : tensor -> tensor
val tanh : tensor -> tensor
val conv2d : string -> tensor -> tensor -> tensor
val add_bias : tensor -> tensor -> tensor
val transpose : tensor -> tensor
