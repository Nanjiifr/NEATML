open Tensor
open Utils

type t = {
  weights : Tensor.t;
  bias : Tensor.t;
  mutable grad_weights : Tensor.t; 
  mutable grad_bias : Tensor.t; 
  mutable input_cache : Tensor.t option; 
  activation : Activations.t;
  mutable preact_cache : Tensor.t option;
}

let act_to_int = function
  | Activations.ReLU -> 1
  | Tanh -> 2
  | Sigmoid -> 3
  | _ -> 0

let create_bias_tensor layer dim =
  let b_cpu = to_cpu layer.bias in
  let res = Array.init dim (fun _ -> Array.copy b_cpu.(0)) in
  let t = CPU res in
  if !use_gpu then to_gpu t else t

let create in_dim out_dim _batch activation =
  let limit = sqrt (6.0 /. float (in_dim + out_dim)) in
  let w_data = Array.init out_dim (fun _ -> Array.init in_dim (fun _ -> Random.float (2.0 *. limit) -. limit)) in
  let b_data = [| Array.make out_dim 0.0 |] in
  let weights = if !use_gpu then GPU (Gpu.of_cpu w_data) else CPU w_data in
  let bias = if !use_gpu then GPU (Gpu.of_cpu b_data) else CPU b_data in
  let gw = if !use_gpu then GPU (Gpu.of_cpu (Array.make_matrix out_dim in_dim 0.0)) else CPU (Array.make_matrix out_dim in_dim 0.0) in
  let gb = if !use_gpu then GPU (Gpu.of_cpu [| Array.make out_dim 0.0 |]) else CPU [| Array.make out_dim 0.0 |] in
  { weights; bias; grad_weights = gw; grad_bias = gb; input_cache = None; activation; preact_cache = None }

let forward layer inputs =
  let inputs = if !use_gpu then to_gpu inputs else inputs in
  layer.input_cache <- Some inputs;
  match inputs, layer.weights, layer.bias with
  | GPU in_g, GPU w_g, GPU b_g when !use_gpu ->
      let out_g = Gpu.linear_fwd in_g w_g b_g (rows inputs) (rows layer.weights) (cols layer.weights) (act_to_int layer.activation) in
      layer.preact_cache <- Some (GPU out_g);
      GPU out_g
  | _ ->
      let inputs_cpu = to_cpu inputs in
      let w_cpu = to_cpu layer.weights in
      let b_cpu = to_cpu layer.bias in
      let z = Array.init (Array.length inputs_cpu) (fun b ->
          let row = Array.make (Array.length w_cpu) 0.0 in
          for i = 0 to Array.length w_cpu - 1 do
              let sum = ref b_cpu.(0).(i) in
              for k = 0 to Array.length inputs_cpu.(0) - 1 do sum := !sum +. inputs_cpu.(b).(k) *. w_cpu.(i).(k) done;
              row.(i) <- !sum
          done; row) in
      let z_t = CPU z in
      layer.preact_cache <- Some z_t;
      Activations.activate layer.activation z_t

let backward layer upstream =
  match !use_gpu, upstream, layer.input_cache, layer.weights, layer.preact_cache, layer.grad_weights, layer.grad_bias with
  | true, GPU up_g, Some (GPU in_g), GPU w_g, Some (GPU out_g), GPU gw_g, GPU gb_g ->
      let batch = rows upstream in
      let out_dim = rows layer.weights in
      let in_dim = cols layer.weights in
      let act = act_to_int layer.activation in
      let d_in_g = Gpu.linear_bwd up_g in_g w_g out_g gw_g gb_g batch out_dim in_dim act in
      { Gradients.d_input = GPU d_in_g; d_weights = Gradients.Dense layer.grad_weights; d_bias = to_cpu layer.grad_bias |> fun m -> m.(0) }
  | _ -> failwith "Linear.backward: CPU fallback not implemented in full GPU mode"

let zero_grad layer =
  match layer.grad_weights, layer.grad_bias with
  | GPU gw, GPU gb ->
      let zw = Gpu.of_cpu (Array.make_matrix (rows layer.grad_weights) (cols layer.grad_weights) 0.0) in
      let zb = Gpu.of_cpu [| Array.make (cols layer.grad_bias) 0.0 |] in
      Gpu.copy_inplace zw gw;
      Gpu.copy_inplace zb gb
  | _ -> 
      layer.grad_weights <- zeros (rows layer.grad_weights) (cols layer.grad_weights);
      layer.grad_bias <- zeros (rows layer.grad_bias) (cols layer.grad_bias)