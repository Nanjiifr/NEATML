(** Structure containing classification performance metrics *)
type classification_metrics = {
  accuracy : float;             (** Fraction of correct predictions *)
  precision : float;            (** Macro-average precision *)
  recall : float;               (** Macro-average recall *)
  f1_score : float;             (** Harmonic mean of precision and recall *)
  confusion_matrix : float array array; (** Raw confusion matrix (rows=targets, cols=preds) *)
}

(** [evaluate model x_train y_train x_test y_test batch_size]
    Performs a full evaluation of the model on both training and test sets.
    Computes and prints Accuracy, Precision, Recall, and F1-score using GPU acceleration.
    Requires GPU to be enabled.
    
    @param model The sequential model to evaluate
    @param x_train Training inputs
    @param y_train Training targets (one-hot)
    @param x_test Test inputs
    @param y_test Test targets (one-hot)
    @param batch_size Batch size for evaluation inference
*)
val evaluate : Sequential.t -> Tensor.t -> Tensor.t -> Tensor.t -> Tensor.t -> int -> unit

(** [evaluate_dataset model x y batch_size name]
    Evaluates the model on a single dataset and returns the metrics.
    
    @param model The model
    @param x Inputs
    @param y Targets
    @param batch_size Batch size
    @param name Display name for the report
    @return Calculated metrics
*)
val evaluate_dataset : Sequential.t -> Tensor.t -> Tensor.t -> int -> string -> classification_metrics