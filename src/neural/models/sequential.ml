type t = { layers : Layer.t list }

let zero_grad_seq (network : t) : unit = 
  List.iter
    (fun l -> 
       match l with 
       | Layer.Linear layer -> Linear.zero_grad layer
       | Layer.Conv2d layer -> Conv2d.zero_grad layer
       | Layer.Dropout _ -> ()
       | Layer.MaxPool2d _ -> ()
    )
    network.layers

let forward_seq (network : t) (inputs : Tensor.t) = 
  List.fold_left
    (fun inputs l ->
      match l with 
      | Layer.Linear layer -> Linear.forward layer inputs
      | Layer.Conv2d layer -> Conv2d.forward layer inputs
      | Layer.Dropout layer -> Dropout.forward layer inputs
      | Layer.MaxPool2d layer -> Pooling.forward layer inputs
    )
    inputs network.layers

let backward_seq (network : t) (grad_output : Tensor.t) = 
  let upstream_grad = ref grad_output in
  let gradients = ref [] in

  List.iter
    (fun l ->
      let current_grad = !upstream_grad in
      match l with 
      | Layer.Linear layer ->
          let grads_rec = Linear.backward layer current_grad in
          upstream_grad := grads_rec.d_input;
          gradients := grads_rec :: !gradients;
          (match current_grad with Tensor.GPU g_tensor when current_grad <> grads_rec.d_input -> Gpu.release g_tensor | _ -> ())
      | Layer.Conv2d layer ->
          let grads_rec = Conv2d.backward layer current_grad in
          upstream_grad := grads_rec.d_input;
          gradients := grads_rec :: !gradients;
          (match current_grad with Tensor.GPU g_tensor when current_grad <> grads_rec.d_input -> Gpu.release g_tensor | _ -> ())
      | Layer.Dropout layer ->
          let grads_rec = Dropout.backward layer current_grad in
          upstream_grad := grads_rec.d_input;
          gradients := grads_rec :: !gradients;
          (* Dropout backward reuses grad buffer usually, but let's be safe *)
      | Layer.MaxPool2d layer ->
          let grads_rec = Pooling.backward layer current_grad in
          upstream_grad := grads_rec.d_input;
          gradients := grads_rec :: !gradients;
          (* MaxPool allocates new grad input *)
          (match current_grad with Tensor.GPU g_tensor -> Gpu.release g_tensor | _ -> ())
    )
    (List.rev network.layers);

  !gradients

let print_weights (net : t) = 
  List.iteri
    (fun i layer ->
      match layer with 
      | Layer.Linear l ->
          Printf.printf "Layer %d (Linear) weights:\n" (i + 1);
          let weights = Utils.to_cpu l.weights in
          let bias = Utils.to_cpu l.bias in
          Array.iteri
            (fun j row ->
              Printf.printf "  Neuron %d: [%s]\n" j
                (String.concat "; "
                   (Array.to_list (Array.map (fun v -> Printf.sprintf "%.4f" v) row))))
            weights;
          Printf.printf "  Bias: [%s]\n"
            (String.concat "; "
               (Array.to_list (Array.map (fun v -> Printf.sprintf "%.4f" v) bias.(0))))
      | Layer.Conv2d c ->
          Printf.printf "  Layer %d (Conv2d):\n" (i + 1);
          Printf.printf "  Kernels: [%d x %d x %d x %d]\n" c.output_depth c.input_depth c.kernel_size c.kernel_size;
          let bias_cpu = Utils.to_cpu c.bias in
          Printf.printf "  Bias: [%s]\n"
            (String.concat "; "
               (Array.to_list (Array.map (fun v -> Printf.sprintf "%.4f" v) bias_cpu.(0))))
      | _ -> ()
    )
    net.layers

let update_seq (network : t) (update_fn : Layer.t -> Gradients.t -> unit)
    (grads_list : Gradients.t list) : unit = 
  List.iter2 (fun l grads -> update_fn l grads) network.layers grads_list

let get_out_dim (model : t) = 
  let l = List.length model.layers in
  let last = List.nth model.layers (l - 1) in
  match last with 
  | Layer.Linear layer -> Utils.rows layer.weights
  | Layer.Conv2d c -> 
      let out_h = c.input_height - c.kernel_size + 1 in
      let out_w = c.input_width - c.kernel_size + 1 in
      c.output_depth * out_h * out_w
  | Layer.Dropout _ -> failwith "Cannot determine out dim from Dropout (context dependent)"
  | Layer.MaxPool2d p ->
      let d, h, w = Pooling.get_output_dims p in
      d * h * w

let get_in_dim (model : t) = 
  let first = List.hd model.layers in
  match first with 
  | Layer.Linear layer -> Utils.cols layer.weights
  | Layer.Conv2d c -> c.input_depth * c.input_height * c.input_width
  | Layer.Dropout _ -> failwith "Cannot determine in dim from Dropout"
  | Layer.MaxPool2d p -> p.input_depth * p.input_height * p.input_width

let summary (model : t) =
  let format_int n =
    let s = string_of_int n in
    let len = String.length s in
    let rec aux i acc =
      if i <= 0 then acc
      else if i <= 3 then (String.sub s 0 i) ^ acc
      else
        let chunk = String.sub s (i - 3) 3 in
        aux (i - 3) ("," ^ chunk ^ acc)
    in
    aux len ""
  in
  let line = "=================================================================" in
  let thin_line = "_________________________________________________________________" in
  Printf.printf "%-28s %-25s %-15s\n" "Layer (type)" "Output Shape" "Param #";
  Printf.printf "%s\n" line;
  
  let total_params = ref 0 in
  let linear_count = ref 0 in
  let conv2d_count = ref 0 in
  let drop_count = ref 0 in
  let pool_count = ref 0 in
  let current_shape = ref "" in

  List.iter (fun layer ->
    let name, out_shape, params =
      match layer with
      | Layer.Linear l ->
          incr linear_count;
          let in_dim = Utils.cols l.weights in
          let out_dim = Utils.rows l.weights in
          let p = (in_dim * out_dim) + out_dim in
          let n = Printf.sprintf "linear_%d (Linear)" !linear_count in
          let s = Printf.sprintf "(None, %d)" out_dim in
          current_shape := s;
          (n, s, p)
      | Layer.Conv2d c ->
          incr conv2d_count;
          let p = (c.output_depth * c.input_depth * c.kernel_size * c.kernel_size) + c.output_depth in
          let n = Printf.sprintf "conv2d_%d (Conv2d)" !conv2d_count in
          let out_h = c.input_height - c.kernel_size + 1 in
          let out_w = c.input_width - c.kernel_size + 1 in
          let s = Printf.sprintf "(None, %d, %d, %d)" c.output_depth out_h out_w in
          current_shape := s;
          (n, s, p)
      | Layer.Dropout _ ->
          incr drop_count;
          let n = Printf.sprintf "dropout_%d (Dropout)" !drop_count in
          let s = if !current_shape = "" then "Unknown" else !current_shape in
          (n, s, 0)
      | Layer.MaxPool2d p ->
          incr pool_count;
          let d, h, w = Pooling.get_output_dims p in
          let n = Printf.sprintf "maxpool_%d (Pool)" !pool_count in
          let s = Printf.sprintf "(None, %d, %d, %d)" d h w in
          current_shape := s;
          (n, s, 0)
    in
    total_params := !total_params + params;
    Printf.printf "%-28s %-25s %-15s\n" name out_shape (format_int params);
    Printf.printf "%s\n" thin_line;
  ) model.layers;
  
  Printf.printf "%s\n" line;
  Printf.printf "Total params: %s\n" (format_int !total_params);
  Printf.printf "Trainable params: %s\n" (format_int !total_params);
  Printf.printf "Non-trainable params: 0\n"

let set_training_mode (model : t) (active : bool) =
  List.iter (fun l -> Layer.set_training_mode l active) model.layers
