type t = MSE | MAE | RMSE | CROSS_ENTROPY

(** [grad_error err y_true y_pred] computes the gradient of the loss function.
    Returns the derivative of the loss with respect to predictions.
    @param err The error/loss function type
    @param y_true The true labels, shape (batch_size, n_classes)
    @param y_pred The predicted values, shape (batch_size, n_classes)
    @return The gradient tensor of the same shape as y_pred *)
let grad_error (err : t) (y_true : Tensor.t) (y_pred : Tensor.t) : Tensor.t =
  let n = Array.length y_true in
  match err with
  | MSE ->
      Utils.map2_mat (fun y p ->
          -2.0 *. (y -. p) /. float_of_int n) y_true y_pred
  | MAE ->
      Utils.map2_mat (fun y p ->
          let diff = y -. p in
          let s = if diff > 0. then 1.0 else if diff < 0. then -1.0 else 0.0 in
          -.s /. float_of_int n) y_true y_pred
  | RMSE ->
      Utils.map2_mat (fun y p ->
          -2.0 *. (y -. p) /. float_of_int n) y_true y_pred
  | CROSS_ENTROPY ->
      Utils.map2_mat (fun y p -> (p -. y) /. float_of_int n) y_true y_pred

(** [mae model real] computes the Mean Absolute Error.
    MAE = (1/(m*n)) * sum(|model - real|)
    @param model The predicted values
    @param real The true values
    @return The MAE loss value *)
let mae (model : Tensor.t) (real : Tensor.t) =
  let n = Float.of_int (Array.length model.(0)) in
  let m = Float.of_int (Array.length model) in
  let temp =
    Array.init (Array.length model) (fun i ->
        Array.mapi
          (fun j _ -> Float.abs (model.(i).(j) -. real.(i).(j)) /. n)
          model.(i))
  in
  Array.fold_left
    (fun acc v ->
      acc +. (Array.fold_left (fun acc1 v1 -> acc1 +. v1) 0. v /. m))
    0. temp

(** [mse model real] computes the Mean Squared Error.
    MSE = (1/(m*n)) * sum((model - real)^2)
    @param model The predicted values
    @param real The true values
    @return The MSE loss value *)
let mse (model : Tensor.t) (real : Tensor.t) =
  let n = Float.of_int (Array.length model.(0)) in
  let m = Float.of_int (Array.length model) in
  let temp =
    Array.init (Array.length model) (fun i ->
        Array.mapi
          (fun j _ -> Float.pow (model.(i).(j) -. real.(i).(j)) 2. /. n)
          model.(i))
  in
  Array.fold_left
    (fun acc v ->
      acc +. (Array.fold_left (fun acc1 v1 -> acc1 +. v1) 0. v /. m))
    0. temp

(** [cross_entropy model real] computes the Cross-Entropy loss.
    CE = -(1/batch_size) * sum(real * log(model + epsilon))
    Uses epsilon for numerical stability.
    @param model The predicted probabilities
    @param real The true labels (one-hot encoded)
    @return The cross-entropy loss value *)
let cross_entropy (model : Tensor.t) (real : Tensor.t) =
  let epsilon = 1e-12 in
  let batch_size = Float.of_int (Array.length model) in
  let tmp =
    Array.mapi
      (fun i _ ->
        -.Array.fold_left ( +. ) 0.
            (Array.map2
               (fun label prob -> label *. log (prob +. epsilon))
               real.(i) model.(i)))
      model
  in
  Array.fold_left ( +. ) 0. tmp /. batch_size

(** [compute_error err real model] computes the specified loss function.
    @param err The error/loss function type to use
    @param real The true labels
    @param model The predicted values
    @return The computed loss value *)
let compute_error (err : t) (real : Tensor.t) (model : Tensor.t) =
  match err with
  | MAE -> mae model real
  | MSE -> mse model real
  | CROSS_ENTROPY -> cross_entropy model real
  | _ -> 0.
