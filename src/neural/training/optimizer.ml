type opt_types = None | Adam
type weight_moments = DenseM of Tensor.t | ConvM of Tensor.t array array

type adam = {
  mutable count : int;
  m_t_weights : weight_moments; v_t_weights : weight_moments;
  m_t_bias : Tensor.t; v_t_bias : Tensor.t;
  beta1 : float; beta2 : float;
  mutable beta1_pow : float; mutable beta2_pow : float;
  eps : float; lr : float; weight_decay : float;
}

type t = None of float | Adam of adam

let create ?(beta1 = 0.9) ?(beta2 = 0.999) ?(eps = 1e-3) ?(weight_decay = 0.01)
    lr _ (seq : Sequential.t) = 
  List.map (fun layer ->
    let out_dim = match layer with Layer.Linear l -> Utils.rows l.weights | _ -> 0 in
    let in_dim = match layer with Layer.Linear l -> Utils.cols l.weights | _ -> 0 in
    let m_t_bias = let t = Utils.zeros 1 out_dim in if !Utils.use_gpu then Utils.to_gpu t else t in
    let v_t_bias = let t = Utils.zeros 1 out_dim in if !Utils.use_gpu then Utils.to_gpu t else t in
    Adam 
      {
        count = 0;
        m_t_weights = DenseM (let t = Utils.zeros out_dim in_dim in if !Utils.use_gpu then Utils.to_gpu t else t);
        v_t_weights = DenseM (let t = Utils.zeros out_dim in_dim in if !Utils.use_gpu then Utils.to_gpu t else t);
        m_t_bias; v_t_bias;
        beta1; beta2; beta1_pow = 1.; beta2_pow = 1.;
        lr; weight_decay; eps;
      }
  ) seq.Sequential.layers

let optimize_adam (layer : Layer.t) (grads : Gradients.t) (o : adam) = 
  o.count <- o.count + 1;
  o.beta1_pow <- o.beta1_pow *. o.beta1;
  o.beta2_pow <- o.beta2_pow *. o.beta2;
  match layer, grads.d_weights, o.m_t_weights, o.v_t_weights with
  | Layer.Linear l, Gradients.Dense gw, DenseM mw, DenseM vw ->
      (match l.weights, gw, mw, vw with 
       | Tensor.GPU pw, Tensor.GPU gww, Tensor.GPU mww, Tensor.GPU vww -> 
           Gpu.adam_step pw gww mww vww o.lr o.beta1 o.beta2 o.beta1_pow o.beta2_pow o.eps o.weight_decay;
           (* Bias update *)
           (match l.bias, o.m_t_bias, o.v_t_bias with 
            | Tensor.GPU pb, Tensor.GPU mb, Tensor.GPU vb -> 
                let gb_cpu = [| grads.d_bias |] in
                let gb_gpu = Gpu.of_cpu gb_cpu in 
                Gpu.adam_step pb gb_gpu mb vb o.lr o.beta1 o.beta2 o.beta1_pow o.beta2_pow o.eps o.weight_decay
            | _ -> ()); 
           Linear.zero_grad l
       | _ -> failwith "Optimizer: GPU enabled but tensors are CPU")
  | _ -> failwith "Optimizer: Unsupported layer"

let update seq grads opt = 
  Utils.list_iter3 (fun l g o -> match o with Adam a -> optimize_adam l g a | _ -> ()) seq.Sequential.layers grads opt

let fit (model : Sequential.t) (xtrain : Tensor.t) (ytrain : Tensor.t)
    (_xtest : Tensor.t) (_ytest : Tensor.t) (batchsize : int) (epochs : int)
    (opt : t list) (err : Errors.t) = 
  let xt_cpu = Utils.to_cpu xtrain in
  let yt_cpu = Utils.to_cpu ytrain in
  let n = Array.length xt_cpu in
  let n_batches = n / batchsize in
  for epoch = 1 to epochs do
    let start_time = Unix.gettimeofday () in
    for b = 0 to n_batches - 1 do
      let idx = b * batchsize in
      let b_in = if !Utils.use_gpu then Tensor.GPU (Gpu.of_cpu (Array.sub xt_cpu idx batchsize)) else Tensor.CPU (Array.sub xt_cpu idx batchsize) in
      let b_tg = if !Utils.use_gpu then Tensor.GPU (Gpu.of_cpu (Array.sub yt_cpu idx batchsize)) else Tensor.CPU (Array.sub yt_cpu idx batchsize) in
      let preds = Sequential.forward_seq model b_in in
      let grads = Sequential.backward_seq model (Errors.grad_error err b_tg preds) in
      update model grads opt;
      if !Utils.use_gpu then Gpu.commit_batch () 
    done;
    if !Utils.use_gpu then Gpu.sync ();
    let end_time = Unix.gettimeofday () in
    Printf.printf "Epoch %d complete in %.4fs\n" epoch (end_time -. start_time);
    flush stdout
  done
