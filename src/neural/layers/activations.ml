type t = ReLU | Tanh | Linear | Sigmoid | Softmax

let relu x = match x with x when x > 0.0 -> x | _ -> 0.0

let softmax inputs =
  (* Softmax on CPU for now *)
  let mat = Utils.to_cpu inputs in
  let exps =
    Array.map
      (fun row ->
        let maxv = Array.fold_left max neg_infinity row in
        Array.map (fun x -> exp (x -. maxv)) row)
      mat
  in
  let sums = Array.map (Array.fold_left ( +. ) 0.0) exps in
  let res = Array.mapi (fun i row -> Array.mapi (fun _ x -> x /. sums.(i)) row) exps in
  Tensor.CPU res

let sigmoid x =
  if x >= 0.0 then
    let z = exp (-.x) in
    1. /. (1. +. z)
  else
    let z = exp x in
    z /. (1. +. z)

let activate (acti : t) (inputs : Tensor.t) =
  match inputs with
  | Tensor.GPU g ->
     (match acti with
      | Linear -> Tensor.GPU g
      | Tanh -> Tensor.GPU (Gpu.tanh g)
      | Sigmoid -> Tensor.GPU (Gpu.sigmoid g)
      | ReLU -> Tensor.GPU (Gpu.relu g)
      | Softmax -> softmax (Tensor.GPU g))
  | Tensor.CPU _ ->
     (match acti with
      | Linear -> inputs
      | Tanh -> Utils.map_mat (fun a _ _ -> tanh a) inputs
      | Sigmoid -> Utils.map_mat (fun a _ _ -> sigmoid a) inputs
      | ReLU -> Utils.map_mat (fun a _ _ -> relu a) inputs
      | Softmax -> softmax inputs)

let derivative (acti : t) (z_tensor : Tensor.t) : Tensor.t =
  match acti with
  | Linear -> Utils.map_mat (fun _ _ _ -> 1.0) z_tensor
  | ReLU -> Utils.map_mat (fun z _ _ -> if z > 0.0 then 1.0 else 0.0) z_tensor
  | Tanh -> 
      Utils.map_mat (fun z _ _ -> 
          let t = tanh z in 
          1.0 -. t *. t) z_tensor
  | Sigmoid ->
      Utils.map_mat (fun z _ _ -> 
          let s = sigmoid z in 
          s *. (1.0 -. s)) z_tensor
  | Softmax -> failwith "Activations.derivative: Softmax requires full Jacobian"

(* Derivatives are currently computed on CPU returning 3D array *)
(* This is a bottleneck for GPU training. *)
let derivative_pre (acti : t) (z_tensor : Tensor.t) : float array array array
    =
  let z = Utils.to_cpu z_tensor in
  let batch_size = Array.length z in
  let output_dim = Array.length z.(0) in
  match acti with
  | Linear ->
      Array.init batch_size (fun _ ->
          Array.init output_dim (fun i ->
              Array.init output_dim (fun j -> if i = j then 1.0 else 0.0)))
  | Tanh ->
      Array.init batch_size (fun b ->
          Array.init output_dim (fun i ->
              let t = tanh z.(b).(i) in
              Array.init output_dim (fun j ->
                  if i = j then 1.0 -. (t *. t) else 0.0)))
  | Sigmoid ->
      Array.init batch_size (fun b ->
          Array.init output_dim (fun i ->
              let s = sigmoid z.(b).(i) in
              Array.init output_dim (fun j ->
                  if i = j then s *. (1.0 -. s) else 0.0)))
  | ReLU ->
      Array.init batch_size (fun b ->
          Array.init output_dim (fun i ->
              Array.init output_dim (fun j ->
                  if i = j then if z.(b).(i) > 0.0 then 1.0 else 0.0 else 0.0)))
  | Softmax ->
      let s_tensor = softmax (Tensor.CPU z) in
      let s = Utils.to_cpu s_tensor in
      Array.init batch_size (fun b ->
          Array.init output_dim (fun i ->
              Array.init output_dim (fun j ->
                  if i = j then s.(b).(i) *. (1.0 -. s.(b).(i))
                  else -.(s.(b).(i) *. s.(b).(j)))))