

type t = {
  rate : float;
  mutable mask : Tensor.t option;
  mutable training : bool;
}

let create rate = { rate; mask = None; training = true }

let forward (l : t) (input : Tensor.t) =
  if l.training then (
    let rows = Utils.rows input in
    let cols = Utils.cols input in
    let mask_cpu = Array.make_matrix rows cols 0.0 in
    let scale = 1.0 /. (1.0 -. l.rate) in
    
    for i = 0 to rows - 1 do
      for j = 0 to cols - 1 do
        if Random.float 1.0 > l.rate then
          mask_cpu.(i).(j) <- scale
        else
          mask_cpu.(i).(j) <- 0.0
      done
    done;
    
    let mask_t = 
      if !Utils.use_gpu then 
        Utils.to_gpu (Tensor.CPU mask_cpu) 
      else 
        Tensor.CPU mask_cpu 
    in
    l.mask <- Some mask_t;
    Utils.multiply_matrix_elementwise input mask_t
  ) else (
    (* During inference, we don't drop anything. 
       Since we used inverted dropout (scaling during train), 
       we pass input as is. *)
    input
  )

let backward (l : t) (grad_output : Tensor.t) =
  let dummy_bias = Utils.zeros 1 1 in
  let d_b = if !Utils.use_gpu then Utils.to_gpu dummy_bias else dummy_bias in
  match l.mask with
  | Some m -> 
      let d_input = Utils.multiply_matrix_elementwise grad_output m in
      { Gradients.d_input; d_weights = Gradients.Empty; d_bias = d_b }
  | None -> 
      (* Should not happen if training, but if so, pass through *)
      { Gradients.d_input = grad_output; d_weights = Gradients.Empty; d_bias = d_b }

let set_training_mode (l : t) (active : bool) =
  l.training <- active
