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

(** [create_bias_tensor layer dim] creates a bias tensor by replicating the
    layer's bias vector.
    @param layer The linear layer containing the bias
    @param dim The number of times to replicate the bias (typically batch_size)
    @return A 2D array where each row is a copy of the bias *)
let create_bias_tensor (layer : t) (dim : int) =
  Array.init dim (fun _ -> Array.copy layer.bias)

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
let create in_dim out_dim batch activation =
  let limit = sqrt (6.0 /. float (in_dim + out_dim)) in
  let weights = Array.make_matrix out_dim in_dim 0.0 in
  for i = 0 to out_dim - 1 do
    for j = 0 to in_dim - 1 do
      weights.(i).(j) <- Random.float (2.0 *. limit) -. limit
    done
  done;
  let bias = Array.init out_dim (fun _ -> 0.0) in
  let grad_weights = Utils.zeros out_dim in_dim in
  let grad_bias = Array.init out_dim (fun _ -> 0.0) in
  let input_cache = Some (Utils.zeros batch in_dim) in
  let preact_cache = Some (Utils.zeros batch out_dim) in
  {
    weights;
    bias;
    grad_weights;
    grad_bias;
    input_cache;
    activation;
    preact_cache;
  }

(** [forward layer inputs] performs a forward pass through the linear layer.
    Computes: output = activation(inputs @ weights^T + bias)
    Caches both inputs and pre-activations for the backward pass.
    @param layer The linear layer
    @param inputs A tensor of shape (batch_size, input_dim)
    @return A tensor of shape (batch_size, output_dim) after activation *)
let forward layer inputs =
  layer.input_cache <- Some (Utils.copy_mat inputs);
  let z =
    Utils.add_matrices
      (Utils.multiply_matrix inputs (Utils.transpose layer.weights))
      (create_bias_tensor layer (Array.length inputs))
  in
  layer.preact_cache <- Some (Utils.copy_mat z);
  Activations.activate layer.activation z

(** [zero_grad layer] resets all accumulated gradients to zero. Should be called
    before each new backward pass.
    @param layer The linear layer to reset *)
let zero_grad layer =
  Array.iteri
    (fun i row ->
      Array.iteri (fun j _ -> layer.grad_weights.(i).(j) <- 0.0) row)
    layer.grad_weights;
  Array.iteri (fun i _ -> layer.grad_bias.(i) <- 0.0) layer.grad_bias

(** [backward layer upstream_grad] performs backpropagation through the linear
    layer. Computes gradients with respect to inputs, weights, and biases.
    Includes the derivative of the activation function in the computation.
    Accumulates gradients into layer.grad_weights and layer.grad_bias.
    @param layer The linear layer
    @param upstream_grad
      The gradient from the next layer, shape (batch_size, output_dim)
    @return A Gradients.t record containing d_input, d_weights, and d_bias *)
let backward layer upstream_grad =
  let batch_size = float (Array.length upstream_grad) in
  (* Include the derivative of the activation function *)
  let d_act =
    Activations.derivative_pre layer.activation (Option.get layer.preact_cache)
  in
  let output_dim = Array.length upstream_grad.(0) in
  let upstream =
    Array.init (Array.length upstream_grad) (fun b ->
        let jacobian = d_act.(b) in
        let grad = upstream_grad.(b) in
        (* Multiply Jacobian (output_dim, output_dim) by gradient vector (output_dim) *)
        let res = Array.make output_dim 0.0 in
        for i = 0 to output_dim - 1 do
          for j = 0 to output_dim - 1 do
            res.(i) <- res.(i) +. (jacobian.(i).(j) *. grad.(j))
          done
        done;
        res)
  in
  (* Compute new gradients *)
  let d_weights_new =
    Utils.scalar (1.0 /. batch_size)
      (Utils.multiply_matrix (Utils.transpose upstream)
         (Option.get layer.input_cache))
  in
  let d_bias_new =
    let output_dim = Array.length upstream_grad.(0) in
    Array.init output_dim (fun j ->
        Array.fold_left (fun acc row -> acc +. row.(j)) 0.0 upstream
        /. batch_size)
  in
  Array.iteri
    (fun i row ->
      Array.iteri
        (fun j g ->
          layer.grad_weights.(i).(j) <- layer.grad_weights.(i).(j) +. g)
        row)
    d_weights_new;

  Array.iteri
    (fun j v -> layer.grad_bias.(j) <- layer.grad_bias.(j) +. v)
    d_bias_new;
  let d_input_v = Utils.multiply_matrix upstream layer.weights in
  ({
     d_input = d_input_v;
     d_weights = Gradients.Dense layer.grad_weights;
     d_bias = layer.grad_bias;
   }
    : Gradients.t)
