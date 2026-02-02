type t = ReLU | Tanh | Linear | Sigmoid | Softmax

val relu : float -> float
val softmax : Tensor.t -> Tensor.t
val sigmoid : float -> float
val activate : t -> Tensor.t -> Tensor.t
val derivative_pre : t -> Tensor.t -> float array array array