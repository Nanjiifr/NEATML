(** Sequential model type containing an ordered list of layers *)
type t = { layers : Layer.t list }

(** [zero_grad_seq network] resets all accumulated gradients in the network to zero.
    Should be called before each backward pass.
    @param network The sequential model *)
val zero_grad_seq : t -> unit

(** [forward_seq network inputs] performs a forward pass through all layers.
    Propagates the input through each layer sequentially.
    @param network The sequential model
    @param inputs The input tensor of shape (batch_size, input_dim)
    @return The output tensor after passing through all layers *)
val forward_seq : t -> Tensor.t -> Tensor.t

(** [backward_seq network grad_output] performs backpropagation through all layers.
    Computes gradients by propagating backward from the output to the input.
    @param network The sequential model
    @param grad_output The gradient from the loss function, shape (batch_size, output_dim)
    @return A list of Gradients.t, one for each layer *)
val backward_seq : t -> Tensor.t -> Gradients.t list

(** [update_seq network update_fn grads_list] applies an update function to each layer.
    Allows custom update logic (e.g., different optimizers per layer).
    @param network The sequential model
    @param update_fn Function that takes a layer and its gradients and updates it
    @param grads_list List of gradients for each layer *)
val update_seq :
  t -> (Layer.t -> Gradients.t -> unit) -> Gradients.t list -> unit

(** [print_weights net] prints the weights and biases of all layers.
    Useful for debugging and inspecting trained parameters.
    @param net The sequential model *)
val print_weights : t -> unit

(** [get_out_dim model] returns the output dimension of the model.
    @param model The sequential model
    @return The number of output features from the last layer *)
val get_out_dim : t -> int

(** [get_in_dim model] returns the input dimension of the model.
    @param model The sequential model
    @return The number of input features expected by the first layer *)
val get_in_dim : t -> int
