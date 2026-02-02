type t = {
  weights : Tensor.t; (* Shape: (output_dim, input_dim) *)
  bias : float array; (* Shape: (output_dim) *)
  grad_weights : Tensor.t; (* Accumulated gradients for weights W *)
  grad_bias : float array; (* Accumulated gradients for bias b *)
  mutable input_cache : Tensor.t option; (* Cached inputs for backward pass *)
  activation : Activations.t;
  mutable preact_cache : Tensor.t option;
      (* Cached pre-activations z for backward pass *)
}
(** Linear (fully connected) layer type *)

val create : int -> int -> int -> Activations.t -> t
(** [create in_dim out_dim batch activation] initializes a new linear layer with
    Xavier/Glorot initialization. Weights are randomly initialized with values
    in [-limit, limit] where limit = sqrt(6 / (in_dim + out_dim)). Biases are
    initialized to zero.
    @param in_dim The input dimension (number of input features)
    @param out_dim The output dimension (number of neurons)
    @param batch The batch size (for cache initialization)
    @param activation
      The activation function to apply after the linear transformation
    @return A new linear layer *)

val forward : t -> Tensor.t -> Tensor.t
(** [forward layer inputs] performs a forward pass through the linear layer.
    Computes: output = activation(inputs @ weights^T + bias)
    Caches both inputs and pre-activations for the backward pass.
    @param layer The linear layer
    @param inputs A tensor of shape (batch_size, input_dim)
    @return A tensor of shape (batch_size, output_dim) after activation *)

val zero_grad : t -> unit
(** [zero_grad layer] resets all accumulated gradients to zero. Should be called
    before each new backward pass.
    @param layer The linear layer to reset *)

val backward : t -> Tensor.t -> Gradients.t
(** [backward layer upstream_grad] performs backpropagation through the linear
    layer. Computes gradients with respect to inputs, weights, and biases.
    Includes the derivative of the activation function in the computation.
    Accumulates gradients into layer.grad_weights and layer.grad_bias.
    @param layer The linear layer
    @param upstream_grad
      The gradient from the next layer, shape (batch_size, output_dim)
    @return A Gradients.t record containing d_input, d_weights, and d_bias *)
