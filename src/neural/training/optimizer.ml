type opt_types = None | Adam

type weight_moments =
  | DenseM of Tensor.t
  | ConvM of Tensor.t array array

type adam = {
  mutable count : int;
  m_t_weights : weight_moments;
  v_t_weights : weight_moments;
  m_t_bias : float array;
  v_t_bias : float array;
  beta1 : float;
  beta2 : float;
  mutable beta1_pow : float;
  mutable beta2_pow : float;
  eps : float;
  lr : float;
  weight_decay : float;
}

type t = None of float | Adam of adam

let create ?(beta1 = 0.9) ?(beta2 = 0.999) ?(eps = 1e-3) ?(weight_decay = 0.01)
    lr (opt_type : opt_types) (seq : Sequential.t) = 
  let network = seq.layers in 
  match opt_type with 
  | None -> List.init (List.length network) (fun _ -> None lr)
  | Adam ->
      List.map 
        (fun layer -> 
          Adam 
            {
              count = 0;
              m_t_weights =
                (match layer with 
                 | Layer.Linear l -> 
                     DenseM (Utils.zeros (Array.length l.weights) (Array.length l.weights.(0)))
                 | Layer.Conv2d c ->
                     let d = c.output_depth and id = c.input_depth in
                     let k = c.kernel_size in
                     ConvM (Array.init d (fun _ -> Array.init id (fun _ -> Utils.zeros k k)))
                );
              v_t_weights =
                (match layer with 
                 | Layer.Linear l -> 
                     DenseM (Utils.zeros (Array.length l.weights) (Array.length l.weights.(0)))
                 | Layer.Conv2d c ->
                     let d = c.output_depth and id = c.input_depth in
                     let k = c.kernel_size in
                     ConvM (Array.init d (fun _ -> Array.init id (fun _ -> Utils.zeros k k)))
                );
              m_t_bias =
                Array.make
                  (match layer with Layer.Linear l -> Array.length l.bias | Layer.Conv2d c -> Array.length c.bias)
                  0.;
              v_t_bias =
                Array.make
                  (match layer with Layer.Linear l -> Array.length l.bias | Layer.Conv2d c -> Array.length c.bias)
                  0.;
              beta1;
              beta2;
              beta1_pow = 1.;
              beta2_pow = 1.;
              lr;
              weight_decay;
              eps;
            })
        network

let adam_update_matrix param grad m_t v_t o = 
    Utils.iter_matrix (fun x i j -> 
        let g = grad.(i).(j) in 
        let m_new = (o.beta1 *. m_t.(i).(j)) +. ((1. -. o.beta1) *. g) in 
        m_t.(i).(j) <- m_new;
        let v_new = (o.beta2 *. v_t.(i).(j)) +. ((1. -. o.beta2) *. g *. g) in 
        v_t.(i).(j) <- v_new;
        let m_hat = m_new /. (1. -. o.beta1_pow) in 
        let v_hat = v_new /. (1. -. o.beta2_pow) in 
        let w_decayed = x -. (o.lr *. o.weight_decay *. x) in 
        let step = m_hat /. (sqrt v_hat +. o.eps) in 
        param.(i).(j) <- w_decayed -. (o.lr *. step)
    ) param

let optimize_adam (layer : Layer.t) (grads : Gradients.t) (o : adam) : unit = 
  o.count <- o.count + 1;
  o.beta1_pow <- o.beta1_pow *. o.beta1;
  o.beta2_pow <- o.beta2_pow *. o.beta2;
  
  (match layer, grads.d_weights, o.m_t_weights, o.v_t_weights with 
   | Layer.Linear l, Gradients.Dense g, DenseM m, DenseM v -> 
       adam_update_matrix l.weights g m v o
   | Layer.Conv2d c, Gradients.Conv g, ConvM m, ConvM v -> 
       for out_idx = 0 to c.output_depth - 1 do 
         for in_idx = 0 to c.input_depth - 1 do 
            adam_update_matrix c.kernels.(out_idx).(in_idx) g.(out_idx).(in_idx) m.(out_idx).(in_idx) v.(out_idx).(in_idx) o
         done 
       done
   | _ -> failwith "Optimizer: Mismatch between layer type and gradient type"
  );
  
  let bias = match layer with Layer.Linear l -> l.bias | Layer.Conv2d c -> c.bias in
  let temp = 
    Array.mapi 
      (fun i x -> 
        let m_t = 
          (o.beta1 *. o.m_t_bias.(i)) 
          +. ((1. -. o.beta1) *. grads.d_bias.(i)) 
        in 
        o.m_t_bias.(i) <- m_t;
        let v_t = 
          (o.beta2 *. o.v_t_bias.(i)) 
          +. ((1. -. o.beta2) *. grads.d_bias.(i) *. grads.d_bias.(i)) 
        in 
        o.v_t_bias.(i) <- v_t;
        let c_bias_l1 = m_t /. (1. -. o.beta1_pow) in 
        let c_bias_l2 = v_t /. (1. -. o.beta2_pow) in 
        let step_t = c_bias_l1 /. (sqrt c_bias_l2 +. o.eps) in 
        x -. (o.lr *. step_t)) 
      bias 
  in 
  Array.iteri (fun i v -> bias.(i) <- v) temp;
  
  match layer with 
  | Layer.Linear l -> Linear.zero_grad l 
  | Layer.Conv2d c -> Conv2d.zero_grad c

let update_l (layer : Layer.t) (grads : Gradients.t) (opt : t) = 
  match opt with 
  | None learning_rate -> (
      match layer, grads.d_weights with 
      | Layer.Linear l, Gradients.Dense g -> 
          Utils.iter_matrix (fun x i j -> l.weights.(i).(j) <- x -. (learning_rate *. g.(i).(j))) l.weights;
          Array.iteri (fun i b -> l.bias.(i) <- b -. (learning_rate *. grads.d_bias.(i))) l.bias;
          Linear.zero_grad l
      | Layer.Conv2d c, Gradients.Conv g -> 
          for out_idx = 0 to c.output_depth - 1 do 
             for in_idx = 0 to c.input_depth - 1 do 
                Utils.iter_matrix (fun x r col -> 
                    c.kernels.(out_idx).(in_idx).(r).(col) <- x -. (learning_rate *. g.(out_idx).(in_idx).(r).(col))
                ) c.kernels.(out_idx).(in_idx)
             done
          done;
          Array.iteri (fun i b -> c.bias.(i) <- b -. (learning_rate *. grads.d_bias.(i))) c.bias;
          Conv2d.zero_grad c
      | _ -> failwith "Optimizer: Mismatch in SGD update"
      )
  | Adam o -> optimize_adam layer grads o

let update (seq : Sequential.t) (grads : Gradients.t list) (opt : t list) = 
  let network = seq.layers in 
  Utils.list_iter3 (fun layer grad o -> update_l layer grad o) network grads opt

let shuffle_indices n = 
  let a = Array.init n (fun i -> i) in 
  Array.shuffle ~rand:Random.int a;
  a

let compute_test (model : Sequential.t) (xtest : Tensor.t) (ytest : Tensor.t) = 
  let num_samples = Array.length xtest in 
  for i = 0 to num_samples - 1 do 
    let pred = Sequential.forward_seq model [| xtest.(i) |] in 
    Printf.printf "Sample %d:\n" (i + 1);
    Printf.printf "  Predicted: ["; 
    Array.iter (fun v -> Printf.printf " %.4f" v) pred.(0);
    Printf.printf " ]\n";
    Printf.printf "  Expected:  ["; 
    Array.iter (fun v -> Printf.printf " %.4f" v) ytest.(i);
    Printf.printf " ]\n";
    Printf.printf "\n"
  done

let fit (model : Sequential.t) (xtrain : Tensor.t) (ytrain : Tensor.t) 
    (xtest : Tensor.t) (ytest : Tensor.t) (batchsize : int) (epochs : int) 
    (opt : t list) (err : Errors.t) = 
  assert (Array.length xtrain = Array.length ytrain);
  Random.self_init ();
  let num_samples = Array.length xtrain in 
  let num_batches = num_samples / batchsize in 
  for epoch = 1 to epochs do 
    let idx = shuffle_indices num_samples in 
    Printf.printf "Starting epoch n%d ...\n" epoch;
    let epoch_loss = ref 0.0 in 
    for b = 0 to num_batches - 1 do 
      let start_idx = b * batchsize in 
            let batch_inputs =
              Array.init batchsize (fun i -> xtrain.(idx.(i + start_idx)))
            in
            let batch_targets =
              Array.init batchsize (fun i -> ytrain.(idx.(i + start_idx)))
            in      let preds = Sequential.forward_seq model batch_inputs in 
      let batch_loss = Errors.compute_error err batch_targets preds in 
      epoch_loss := !epoch_loss +. batch_loss;
      let grad_outputs = Errors.grad_error err batch_targets preds in 
      let grads = Sequential.backward_seq model grad_outputs in 
      update model grads opt 
    done;
    let avg_loss = !epoch_loss /. float_of_int num_batches in 
    let test_loss_sum = ref 0.0 in 
    let n_test = Array.length xtest in 
    for i = 0 to n_test - 1 do 
      let x = [| xtest.(i) |] in 
      let y = [| ytest.(i) |] in 
      let pred = Sequential.forward_seq model x in 
      let loss = Errors.compute_error err y pred in 
      test_loss_sum := !test_loss_sum +. loss 
    done;
    let avg_test_loss = !test_loss_sum /. float_of_int n_test in 
    Printf.printf "Epoch %d: average train loss = %.6f | test loss = %.6f\n" 
      epoch avg_loss avg_test_loss;
    flush stdout 
  done;
  compute_test model xtest ytest
