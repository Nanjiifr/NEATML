type weight_grad =
  | Dense of Tensor.t
  | Conv of Tensor.t array array

(** Gradient record containing all computed gradients from backpropagation *)
type t = {
  d_input : Tensor.t; (* Gradient with respect to inputs *)
  d_weights : weight_grad; (* Gradient with respect to weights *)
  d_bias : float array; (* Gradient with respect to biases *)
}