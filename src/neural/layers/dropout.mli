

type t

val create : float -> t
val forward : t -> Tensor.t -> Tensor.t
val backward : t -> Tensor.t -> Gradients.t
val set_training_mode : t -> bool -> unit
