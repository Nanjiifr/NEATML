type t = ReLU | Tanh | Linear | Sigmoid | Softmax

(** [relu x] applies the ReLU (Rectified Linear Unit) activation function.
    Returns x if x > 0, otherwise returns 0.
    @param x The input value
    @return The activated value *)
let relu x = match x with x when x > 0.0 -> x | _ -> 0.0

(** [softmax inputs] applies the softmax activation function to a batch of inputs.
    Computes exponentials with numerical stability (subtracting max value).
    @param inputs A tensor of shape (batch_size, n_features)
    @return A tensor with softmax applied, same shape as inputs *)
let softmax inputs =
  let exps =
    Array.map
      (fun row ->
        let maxv = Array.fold_left max neg_infinity row in
        Array.map (fun x -> exp (x -. maxv)) row)
      inputs
  in
  let sums = Array.map (Array.fold_left ( +. ) 0.0) exps in
  Array.mapi (fun i row -> Array.mapi (fun _ x -> x /. sums.(i)) row) exps

(** [sigmoid x] applies the sigmoid activation function with numerical stability.
    Uses different formulations for positive and negative inputs to avoid overflow.
    @param x The input value
    @return A value between 0 and 1 *)
let sigmoid x =
  if x >= 0.0 then
    let z = exp (-.x) in
    1. /. (1. +. z)
  else
    let z = exp x in
    z /. (1. +. z)

(** [activate acti inputs] applies the specified activation function to a tensor.
    @param acti The activation function type (Linear, Tanh, Sigmoid, ReLU, or Softmax)
    @param inputs A tensor of shape (batch_size, output_dim)
    @return The tensor after applying the activation function *)
let activate (acti : t) (inputs : Tensor.t) =
  if !Utils.use_gpu then
    match acti with
    | Linear -> inputs
    | Tanh -> Gpu.tanh inputs
    | Sigmoid -> Gpu.sigmoid inputs
    | ReLU -> Gpu.relu inputs
    | Softmax -> softmax inputs (* Softmax usually requires reduction, keeping on CPU for now or implement in GPU *)
  else
  match acti with
  | Linear -> inputs
  | Tanh -> Utils.map_mat (fun a _ _ -> tanh a) inputs
  | Sigmoid -> Utils.map_mat (fun a _ _ -> sigmoid a) inputs
  | ReLU -> Utils.map_mat (fun a _ _ -> relu a) inputs
  | Softmax -> softmax inputs

(** [derivative_pre acti z] computes the Jacobian matrix of the activation function derivative.
    The derivative is evaluated on the pre-activation values z (before applying activation).
    @param acti The activation function type
    @param z The pre-activation tensor of shape (batch_size, output_dim)
    @return A 3D array of shape (batch_size, output_dim, output_dim) representing the Jacobian for each sample *)
let derivative_pre (acti : t) (z : float array array) : float array array array
    =
  let batch_size = Array.length z in
  let output_dim = Array.length z.(0) in
  match acti with
  | Linear ->
      Array.init batch_size (fun _ ->
          Array.init output_dim (fun i ->
              Array.init output_dim (fun j -> if i = j then 1.0 else 0.0)))
  | Tanh ->
      Array.init batch_size (fun b ->
          Array.init output_dim (fun i ->
              let t = tanh z.(b).(i) in
              Array.init output_dim (fun j ->
                  if i = j then 1.0 -. (t *. t) else 0.0)))
  | Sigmoid ->
      Array.init batch_size (fun b ->
          Array.init output_dim (fun i ->
              let s = sigmoid z.(b).(i) in
              Array.init output_dim (fun j ->
                  if i = j then s *. (1.0 -. s) else 0.0)))
  | ReLU ->
      Array.init batch_size (fun b ->
          Array.init output_dim (fun i ->
              Array.init output_dim (fun j ->
                  if i = j then if z.(b).(i) > 0.0 then 1.0 else 0.0 else 0.0)))
  | Softmax ->
      let s = softmax z in
      Array.init batch_size (fun b ->
          Array.init output_dim (fun i ->
              Array.init output_dim (fun j ->
                  if i = j then s.(b).(i) *. (1.0 -. s.(b).(i))
                  else -.(s.(b).(i) *. s.(b).(j)))))
