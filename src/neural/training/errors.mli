(** Loss/error function types *)
type t = MSE | MAE | RMSE | CROSS_ENTROPY

(** [compute_error err real model] computes the specified loss function.
    @param err The error/loss function type to use
    @param real The true labels
    @param model The predicted values
    @return The computed loss value *)
val compute_error : t -> Tensor.t -> Tensor.t -> float

(** [grad_error err y_true y_pred] computes the gradient of the loss function.
    Returns the derivative of the loss with respect to predictions.
    @param err The error/loss function type
    @param y_true The true labels, shape (batch_size, n_classes)
    @param y_pred The predicted values, shape (batch_size, n_classes)
    @return The gradient tensor of the same shape as y_pred *)
val grad_error : t -> Tensor.t -> Tensor.t -> Tensor.t
