type t = { layers : Layer.t list }

let zero_grad_seq (network : t) : unit =
  List.iter
    (fun l -> 
       match l with 
       | Layer.Linear layer -> Linear.zero_grad layer
       | Layer.Conv2d layer -> Conv2d.zero_grad layer
    )
    network.layers

let forward_seq (network : t) (inputs : Tensor.t) =
  List.fold_left
    (fun inputs l ->
      match l with 
      | Layer.Linear layer -> Linear.forward layer inputs
      | Layer.Conv2d layer -> Conv2d.forward layer inputs
    )
    inputs network.layers

let backward_seq (network : t) (grad_output : Tensor.t) =
  let upstream_grad = ref grad_output in
  let gradients = ref [] in

  List.iter
    (fun l ->
      match l with
      | Layer.Linear layer ->
          let g = Linear.backward layer !upstream_grad in
          upstream_grad := g.d_input;
          gradients := g :: !gradients
      | Layer.Conv2d layer ->
          let g = Conv2d.backward layer !upstream_grad in
          upstream_grad := g.d_input;
          gradients := g :: !gradients
    )
    (List.rev network.layers);

  !gradients

let print_weights (net : t) =
  List.iteri
    (fun i layer ->
      match layer with
      | Layer.Linear l ->
          Printf.printf "Layer %d (Linear) weights:\n" (i + 1);
          Array.iteri
            (fun j row ->
              Printf.printf "  Neuron %d: [%s]\n" j
                (String.concat "; "
                   (Array.to_list (Array.map (Printf.sprintf "%.4f") row))))
            l.weights;
          Printf.printf "  Bias: [%s]\n"
            (String.concat "; "
               (Array.to_list (Array.map (Printf.sprintf "%.4f") l.bias)))
      | Layer.Conv2d c ->
          Printf.printf "Layer %d (Conv2d):\n" (i + 1);
          Printf.printf "  Kernels: [%dx%dx%dx%d]\n" c.output_depth c.input_depth c.kernel_size c.kernel_size;
          Printf.printf "  Bias: [%s]\n"
            (String.concat "; "
               (Array.to_list (Array.map (Printf.sprintf "%.4f") c.bias)))
    )
    net.layers

let update_seq (network : t) (update_fn : Layer.t -> Gradients.t -> unit)
    (grads_list : Gradients.t list) : unit =
  List.iter2 (fun l grads -> update_fn l grads) network.layers grads_list

let get_out_dim (model : t) =
  let l = List.length model.layers in
  let last = List.nth model.layers (l - 1) in
  match last with 
  | Layer.Linear layer -> Array.length layer.weights
  | Layer.Conv2d c -> 
      let out_h = c.input_height - c.kernel_size + 1 in
      let out_w = c.input_width - c.kernel_size + 1 in
      c.output_depth * out_h * out_w

let get_in_dim (model : t) =
  let first = List.hd model.layers in
  match first with 
  | Layer.Linear layer -> Array.length layer.weights.(0)
  | Layer.Conv2d c -> c.input_depth * c.input_height * c.input_width