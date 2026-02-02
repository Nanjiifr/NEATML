(** Activation function types supported by the neural network *)
type t = ReLU | Tanh | Linear | Sigmoid | Softmax

(** [activate acti inputs] applies the specified activation function to a tensor.
    @param acti The activation function type (Linear, Tanh, Sigmoid, ReLU, or Softmax)
    @param inputs A tensor of shape (batch_size, output_dim)
    @return The tensor after applying the activation function *)
val activate : t -> Tensor.t -> Tensor.t

(** [derivative_pre acti z] computes the Jacobian matrix of the activation function derivative.
    The derivative is evaluated on the pre-activation values z (before applying activation).
    @param acti The activation function type
    @param z The pre-activation tensor of shape (batch_size, output_dim)
    @return A 3D array of shape (batch_size, output_dim, output_dim) representing the Jacobian for each sample *)
val derivative_pre : t -> Tensor.t -> Tensor.t array
