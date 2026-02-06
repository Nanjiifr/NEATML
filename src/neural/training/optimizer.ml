type opt_types = None | Adam
type weight_moments = DenseM of Tensor.t | ConvM of Tensor.t array array

type adam = {
  mutable count : int;
  m_t_weights : weight_moments;
  v_t_weights : weight_moments;
  m_t_bias : Tensor.t;
  v_t_bias : Tensor.t;
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
    lr _ (seq : Sequential.t) =
  List.map
    (fun layer ->
          let out_dim = match layer with 
            | Layer.Linear l -> Utils.rows l.weights 
            | Layer.Conv2d c -> c.output_depth
            | _ -> 0 (* Ignored for parameterless layers *)
          in
          let in_dim = match layer with 
            | Layer.Linear l -> Utils.cols l.weights 
            | Layer.Conv2d c -> c.input_depth
            | _ -> 0 (* Ignored *)
          in
          
              (* Moments for weights *)
          
              let m_w, v_w = match layer with
          
              | Layer.Linear _ -> 
          
                  let t1 = Utils.zeros out_dim in_dim in
          
                  let t2 = Utils.zeros out_dim in_dim in
          
                  let m = if !Utils.use_gpu then Utils.to_gpu t1 else t1 in
          
                  let v = if !Utils.use_gpu then Utils.to_gpu t2 else t2 in
          
                  DenseM m, DenseM v
          
                  | Layer.Conv2d c ->
          
                      let t1 = Utils.zeros 1 (c.output_depth * c.input_depth * c.kernel_size * c.kernel_size) in
          
                      let t2 = Utils.zeros 1 (c.output_depth * c.input_depth * c.kernel_size * c.kernel_size) in
          
                      let m = if !Utils.use_gpu then Utils.to_gpu t1 else t1 in
          
                      let v = if !Utils.use_gpu then Utils.to_gpu t2 else t2 in
          
                      DenseM m, DenseM v
          
          | Layer.Dropout _ | Layer.MaxPool2d _ ->
              let t = Utils.zeros 1 1 in
              DenseM t, DenseM t (* Dummy, unused *)
                  in
          
              
          
          (* Moments for bias. Now both Linear and Conv2d use GPU if available *)
          let m_b, v_b = 
              match layer with
              | Layer.Dropout _ | Layer.MaxPool2d _ -> 
                  let t = Utils.zeros 1 1 in t, t
              | Layer.Linear l ->
                  let t = Utils.zeros 1 (Utils.rows l.weights) in
                  let tm = if !Utils.use_gpu then Utils.to_gpu t else t in
                  let tv = if !Utils.use_gpu then Utils.to_gpu t else t in
                  tm, tv
              | Layer.Conv2d c ->
                  let t = Utils.zeros 1 c.output_depth in
                  let tm = if !Utils.use_gpu then Utils.to_gpu t else t in
                  let tv = if !Utils.use_gpu then Utils.to_gpu t else t in
                  tm, tv
          in
      
              Adam 
      
                {
      
                  count = 0;
      
                  m_t_weights = m_w;
      
                  v_t_weights = v_w;
      
                  m_t_bias = m_b; v_t_bias = v_b;
      
          
              beta1; beta2; beta1_pow = 1.; beta2_pow = 1.;
              lr; weight_decay; eps;
            }
      )
    seq.Sequential.layers

let optimize_adam (layer : Layer.t) (grads : Gradients.t) (o : adam) =
  o.count <- o.count + 1;
  o.beta1_pow <- o.beta1_pow *. o.beta1;
  o.beta2_pow <- o.beta2_pow *. o.beta2;
  match (layer, grads.d_weights, o.m_t_weights, o.v_t_weights) with
  | Layer.Linear l, Gradients.Dense gw, DenseM mw, DenseM vw -> (
      match (l.weights, gw, mw, vw) with
      | Tensor.GPU pw, Tensor.GPU gww, Tensor.GPU mww, Tensor.GPU vww ->
          Gpu.adam_step pw gww mww vww o.lr o.beta1 o.beta2 o.beta1_pow
            o.beta2_pow o.eps o.weight_decay;
          (* Bias update *)
          (match (l.bias, grads.d_bias, o.m_t_bias, o.v_t_bias) with
          | Tensor.GPU pb, Tensor.GPU gbb, Tensor.GPU mb, Tensor.GPU vb ->
              Gpu.adam_step pb gbb mb vb o.lr o.beta1 o.beta2 o.beta1_pow
                o.beta2_pow o.eps o.weight_decay
          | _ -> ());
                     Linear.zero_grad l
                 | _ -> failwith "Optimizer: GPU enabled but tensors are CPU")
                | Layer.Conv2d l, Gradients.Dense gc, DenseM mw, DenseM vw ->
                    (match l.kernels, gc, mw, vw with
                     | Tensor.GPU pw, Tensor.GPU pg, Tensor.GPU pm, Tensor.GPU pv ->
                         Gpu.adam_step pw pg pm pv o.lr o.beta1 o.beta2 o.beta1_pow o.beta2_pow o.eps o.weight_decay
                     | _ -> failwith "Optimizer: Conv2d weights not on GPU");
                    
                    (match l.bias, grads.d_bias, o.m_t_bias, o.v_t_bias with
                     | Tensor.GPU pb, Tensor.GPU gb, Tensor.GPU mb, Tensor.GPU vb ->
                         Gpu.adam_step pb gb mb vb o.lr o.beta1 o.beta2 o.beta1_pow o.beta2_pow o.eps o.weight_decay
                     | _ -> failwith "Optimizer: Conv2d bias not on GPU");
                    
                    Conv2d.zero_grad l
            
            | Layer.Dropout _, _, _, _ -> ()
            | Layer.MaxPool2d _, _, _, _ -> ()
            | _, Gradients.Empty, _, _ -> () (* Skip layers with no weight gradients *)
            | _ -> failwith "Optimizer: Unsupported layer"
          let update seq grads opt =
  Utils.list_iter3
    (fun l g o -> match o with Adam a -> optimize_adam l g a | _ -> ())
    seq.Sequential.layers grads opt

let print_status epoch epochs batch n_batches t_loss v_loss start_time =
  let width = 40 in
  let total_steps = epochs * n_batches in
  let current_step = ((epoch - 1) * n_batches) + batch in
  let filled = if total_steps = 0 then width else (current_step * width) / total_steps in
  let bar = 
    let b = Buffer.create (width * 3) in
    for i = 0 to width - 1 do
      if i < filled then Buffer.add_string b "█"
      else if i = filled then Buffer.add_string b "▓"
      else Buffer.add_string b "░"
    done;
    Buffer.contents b
  in
  let percent = if total_steps = 0 then 100 else (current_step * 100) / total_steps in
  let elapsed = Unix.gettimeofday () -. start_time in
  
  (* Move cursor to start of line, then move up 1 line and clear it *)
  Printf.printf "\r\027[K\027[A\027[K"; 
  
  Printf.printf "\027[1;36mOverall Progress\027[0m \027[1;33m[%s]\027[0m %d%% | \027[1;34mTime: %.1fs\027[0m\n"
    bar percent elapsed;
  Printf.printf "\027[1;32mCurrent State\027[0m  | Epoch: \027[1;36m%d/%d\027[0m | Train Loss: \027[1;32m%.6f\027[0m | Val Loss: \027[1;35m%.6f\027[0m%!"
    epoch epochs t_loss v_loss

let fit (model : Sequential.t) (xtrain : Tensor.t) (ytrain : Tensor.t)
    (xtest : Tensor.t) (ytest : Tensor.t) (batchsize : int) (epochs : int)
    (opt : t list) (err : Errors.t) = 
  let xt_cpu = Utils.to_cpu xtrain in
  let yt_cpu = Utils.to_cpu ytrain in
  let n = Array.length xt_cpu in
  let n_batches = n / batchsize in
  
  let global_start = Unix.gettimeofday () in
  Printf.printf "\027[1;35m[Training Start]\027[0m Batches: %d | Samples: %d\n\n\n%!" n_batches n;

  let val_loss = ref 0.0 in
  let running_loss = ref 0.0 in
  let loss_samples = ref 0 in

  let train_history = ref [] in
  let val_history = ref [] in
  
  (* Determine display refresh rate to avoid stalling pipeline with too many syncs *)
  (* For fast CNNs, updating every 10-20 batches is good. *)
  let refresh_rate = max 1 (n_batches / 20) in 

  Sequential.set_training_mode model true;

  for epoch = 1 to epochs do
    for b = 0 to n_batches - 1 do
      let idx = b * batchsize in
      let b_in_cpu = Array.sub xt_cpu idx batchsize in
      let b_tg_cpu = Array.sub yt_cpu idx batchsize in
      let b_in = if !Utils.use_gpu then Tensor.GPU (Gpu.of_cpu b_in_cpu) else Tensor.CPU b_in_cpu in
      let b_tg = if !Utils.use_gpu then Tensor.GPU (Gpu.of_cpu b_tg_cpu) else Tensor.CPU b_tg_cpu in
      
      let preds = Sequential.forward_seq model b_in in
      
      (* Sample loss for display occasionally to avoid killing perf with sync() *)
      if b mod refresh_rate = 0 || b = n_batches - 1 then (
          let l = Errors.compute_error err b_tg preds in
          running_loss := 0.9 *. !running_loss +. 0.1 *. l;
          if !loss_samples = 0 then running_loss := l; (* Init *)
          loss_samples := !loss_samples + 1;
          print_status epoch epochs (b + 1) n_batches !running_loss !val_loss global_start
      );

      let grads = Sequential.backward_seq model (Errors.grad_error err b_tg preds) in
      update model grads opt;
      
      (match (List.hd grads).Gradients.d_input with Tensor.GPU g -> Gpu.release g | _ -> ());
      (match b_tg with Tensor.GPU g -> Gpu.release g | _ -> ());
      (match b_in with Tensor.GPU g -> Gpu.release g | _ -> ());
      
      (* Removed forced commit_batch to let Gpu module batch commands naturally *)
    done;

    if !Utils.use_gpu then begin
      Gpu.sync ();
      Gc.full_major ()
    end;

    (* Validation loss calculation *)
    let val_preds = Sequential.forward_seq model xtest in
    val_loss := Errors.compute_error err ytest val_preds;
    (match val_preds with Tensor.GPU g -> Gpu.release g | _ -> ());
    
    (* Refresh status with final epoch data *)
    print_status epoch epochs n_batches n_batches !running_loss !val_loss global_start;

    train_history := !running_loss :: !train_history;
    val_history := !val_loss :: !val_history;
  done;
  Printf.printf "\n\n\027[1;32m[Training Complete]\027[0m\n%!";
  (List.rev !train_history, List.rev !val_history)
