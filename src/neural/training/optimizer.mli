(** Optimizer type discriminator *)
type opt_types = None | Adam

type weight_moments =
  | DenseM of Tensor.t
  | ConvM of Tensor.t array array

(** Adam optimizer state *)
type adam = {
  mutable count : int; (* Number of optimization steps *)
  m_t_weights : weight_moments; (* First moment estimate for weights *)
  v_t_weights : weight_moments; (* Second moment estimate for weights *)
  m_t_bias : Tensor.t; (* First moment estimate for biases *)
  v_t_bias : Tensor.t; (* Second moment estimate for biases *)
  beta1 : float; (* Exponential decay rate for first moment *)
  beta2 : float; (* Exponential decay rate for second moment *)
  mutable beta1_pow : float; (* beta1^t for bias correction *)
  mutable beta2_pow : float; (* beta2^t for bias correction *)
  eps : float; (* Numerical stability constant *)
  lr : float; (* Learning rate *)
  weight_decay : float; (* L2 regularization coefficient *)
}

(** Optimizer state type *)
type t = None of float | Adam of adam

(** [create ?beta1 ?beta2 ?eps ?weight_decay lr opt_type seq] initializes optimizer states for all layers.
    For Adam optimizer, creates momentum and velocity accumulators for each layer.
    @param beta1 First moment decay rate for Adam (default: 0.9)
    @param beta2 Second moment decay rate for Adam (default: 0.999)
    @param eps Numerical stability term for Adam (default: 1e-3)
    @param weight_decay L2 regularization coefficient for AdamW (default: 0.01)
    @param lr Learning rate
    @param opt_type Type of optimizer (None for vanilla SGD, or Adam)
    @param seq The sequential model containing layers to optimize
    @return A list of optimizer states, one per layer *)
val create :
  ?beta1:float ->
  ?beta2:float ->
  ?eps:float ->
  ?weight_decay:float ->
  float ->
  opt_types ->
  Sequential.t ->
  t list

(** [update seq grads opt] updates all layers in a sequential model.
    @param seq The sequential model
    @param grads List of gradients for each layer
    @param opt List of optimizer states for each layer *)
val update : Sequential.t -> Gradients.t list -> t list -> unit

(** [fit model xtrain ytrain xtest ytest batchsize epochs opt err] trains the model.
    Performs mini-batch gradient descent with data shuffling each epoch.
    Prints training and test loss after each epoch.
    @param model The sequential model to train
    @param xtrain Training input data
    @param ytrain Training target labels
    @param xtest Test input data
    @param ytest Test target labels
    @param batchsize The batch size for mini-batch training
    @param epochs Number of training epochs
    @param opt List of optimizer states
    @param err The loss function to use *)
val fit :
  Sequential.t ->
  Tensor.t ->
  Tensor.t ->
  Tensor.t ->
  Tensor.t ->
  int ->
  int ->
  t list ->
  Errors.t ->
  unit
