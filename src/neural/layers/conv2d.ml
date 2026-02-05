open Tensor
open Utils

type t = {
  mutable kernels : Tensor.t;         (* 4D: OutDepth x InDepth x KH x KW *)
  mutable bias : Tensor.t;
  mutable grad_weights : Tensor.t;    (* 4D: Same as kernels *)
  mutable grad_bias : Tensor.t;
  mutable input_cache : Tensor.t option;
  activation : Activations.t;
  mutable preact_cache : Tensor.t option;
  input_depth : int;
  input_height : int;
  input_width : int;
  kernel_size : int;
  output_depth : int;
}

let create input_depth input_height input_width kernel_size output_depth
    activation =
  let limit = sqrt (6.0 /. float ((input_depth * kernel_size * kernel_size) + (output_depth * kernel_size * kernel_size))) in
  let total_w = output_depth * input_depth * kernel_size * kernel_size in
  let w_data = [| Array.init total_w (fun _ -> Random.float (2.0 *. limit) -. limit) |] in
  let kernels = if !use_gpu then to_gpu (CPU w_data) else CPU w_data in
  let b_data = [| Array.make output_depth 0.0 |] in
  let bias = if !use_gpu then to_gpu (CPU b_data) else CPU b_data in
  let gw_data = [| Array.make total_w 0.0 |] in
  let grad_weights = if !use_gpu then to_gpu (CPU gw_data) else CPU gw_data in
  let gb_data = [| Array.make output_depth 0.0 |] in
  let grad_bias = if !use_gpu then to_gpu (CPU gb_data) else CPU gb_data in
  { kernels; bias; grad_weights; grad_bias; input_cache = None; activation; preact_cache = None;
    input_depth; input_height; input_width; kernel_size; output_depth }

let zero_grad layer =
  (match layer.input_cache with Some (GPU g) -> Gpu.release g | _ -> ());
  (match layer.preact_cache with Some (GPU g) -> Gpu.release g | _ -> ());
  layer.input_cache <- None;
  layer.preact_cache <- None;
  (match layer.grad_weights with
   | GPU g -> Gpu.zero_tensor g
   | CPU m -> Array.fill m.(0) 0 (Array.length m.(0)) 0.0);
  (match layer.grad_bias with
   | GPU g -> Gpu.zero_tensor g
   | CPU m -> Array.fill m.(0) 0 (Array.length m.(0)) 0.0)

let forward layer inputs =
  let inputs = if !use_gpu then to_gpu inputs else inputs in
  layer.input_cache <- Some (copy_mat inputs) ;
  let batch_size = rows inputs in
  let out_h = layer.input_height - layer.kernel_size + 1 in
  let out_w = layer.input_width - layer.kernel_size + 1 in
  let output_dim = layer.output_depth * out_h * out_w in
  if !use_gpu then begin
      let out_gpu = match to_gpu (zeros batch_size output_dim) with GPU g -> g | _ -> failwith "Alloc" in
      match inputs, layer.kernels, layer.bias with
      | GPU in_gpu, GPU k_gpu, GPU b_gpu ->
          Gpu.conv2d_direct_fwd in_gpu k_gpu b_gpu out_gpu 
             batch_size layer.input_height layer.input_width out_h out_w 
             layer.kernel_size layer.kernel_size layer.input_depth layer.output_depth 0;
          let z_copy = match to_gpu (zeros batch_size output_dim) with GPU g -> g | _ -> failwith "Alloc" in
          Gpu.copy_inplace out_gpu z_copy;
          layer.preact_cache <- Some (GPU z_copy);
          Activations.activate layer.activation (GPU out_gpu)
      | _ -> failwith "Conv2d: GPU error"
  end else begin
    let inputs_cpu = to_cpu inputs in
    let kernels_cpu = to_cpu layer.kernels in
    let bias_cpu = to_cpu layer.bias in
    let outputs = Array.make_matrix batch_size output_dim 0.0 in
    for b = 0 to batch_size - 1 do
      let input_row = inputs_cpu.(b) in
      for o = 0 to layer.output_depth - 1 do
        for i = 0 to layer.input_depth - 1 do
          let weight_offset = (o * layer.input_depth * layer.kernel_size * layer.kernel_size) + (i * layer.kernel_size * layer.kernel_size) in
          let start_idx = i * layer.input_height * layer.input_width in
          for r = 0 to out_h - 1 do
            for c = 0 to out_w - 1 do
              let sum = ref 0.0 in
              for kr = 0 to layer.kernel_size - 1 do
                for kc = 0 to layer.kernel_size - 1 do
                  sum := !sum +. input_row.(start_idx + (r + kr) * layer.input_width + (c + kc)) *. kernels_cpu.(0).(weight_offset + kr * layer.kernel_size + kc)
                done
              done;
              outputs.(b).((o * out_h * out_w) + (r * out_w) + c) <- outputs.(b).((o * out_h * out_w) + (r * out_w) + c) +. !sum
            done
          done
        done;
        let b_val = bias_cpu.(0).(o) in
        for r = 0 to out_h - 1 do
          for c = 0 to out_w - 1 do outputs.(b).((o * out_h * out_w) + (r * out_w) + c) <- outputs.(b).((o * out_h * out_w) + (r * out_w) + c) +. b_val done
        done
      done
    done;
    let res = CPU outputs in
    layer.preact_cache <- Some res;
    Activations.activate layer.activation res
  end

let backward layer upstream_grad =
  if !use_gpu then begin
      let inputs_gpu = match layer.input_cache with Some (GPU g) -> g | _ -> failwith "Conv2d bwd: Cache missing" in
      let preact_gpu = match layer.preact_cache with Some (GPU g) -> g | _ -> failwith "Conv2d bwd: Preact missing" in
      let up_gpu = match upstream_grad with GPU g -> g | CPU c -> (match to_gpu (CPU c) with GPU g -> g | _ -> failwith "Error") in
      let act_name = match layer.activation with Activations.ReLU -> "relu" | Tanh -> "tanh" | Sigmoid -> "sigmoid" | _ -> failwith "Unsupported" in
      let dZ_gpu = Gpu.activation_bwd act_name preact_gpu up_gpu in
      let batch_size = rows (GPU inputs_gpu) in
      let out_h = layer.input_height - layer.kernel_size + 1 in
      let out_w = layer.input_width - layer.kernel_size + 1 in
      (match layer.grad_bias with
       | GPU gb -> Gpu.conv2d_bias_bwd dZ_gpu gb batch_size out_h out_w layer.output_depth
       | _ -> failwith "Error");
      (match layer.grad_weights with
       | GPU gw -> Gpu.conv2d_direct_bwd_weights inputs_gpu dZ_gpu gw batch_size layer.input_height layer.input_width out_h out_w layer.kernel_size layer.kernel_size layer.input_depth layer.output_depth
       | _ -> failwith "Error");
      let d_inputs_gpu = match to_gpu (zeros batch_size (layer.input_depth * layer.input_height * layer.input_width)) with GPU g -> g | _ -> failwith "Alloc" in
      (match layer.kernels with
       | GPU k -> Gpu.conv2d_direct_bwd_input dZ_gpu k d_inputs_gpu batch_size layer.input_height layer.input_width out_h out_w layer.kernel_size layer.kernel_size layer.input_depth layer.output_depth
       | _ -> ());
      Gpu.release dZ_gpu;
      { Gradients.d_input = GPU d_inputs_gpu; d_weights = Gradients.Dense layer.grad_weights; d_bias = layer.grad_bias }
  end else begin
      let upstream_grad_cpu = to_cpu upstream_grad in
      let d_act = Activations.derivative_pre layer.activation (Option.get layer.preact_cache) in
      let batch_size = Array.length upstream_grad_cpu in
      let output_dim = Array.length upstream_grad_cpu.(0) in
      let dl_dz = Array.make_matrix batch_size output_dim 0.0 in
      for b=0 to batch_size-1 do
         for i=0 to output_dim-1 do
           let sum = ref 0.0 in
           for j=0 to output_dim-1 do sum := !sum +. (d_act.(b).(i).(j) *. upstream_grad_cpu.(b).(j)) done;
           dl_dz.(b).(i) <- !sum
         done
      done;
      let out_h = layer.input_height - layer.kernel_size + 1 in
      let out_w = layer.input_width - layer.kernel_size + 1 in
      let d_inputs = Array.make_matrix batch_size (layer.input_depth * layer.input_height * layer.input_width) 0.0 in
            let inputs = to_cpu (Option.get layer.input_cache) in
            let kernels_cpu = to_cpu layer.kernels in
            let gw_cpu = match layer.grad_weights with CPU m -> m | _ -> to_cpu layer.grad_weights in
            let gb_cpu = match layer.grad_bias with CPU m -> m | _ -> to_cpu layer.grad_bias in
            for b=0 to batch_size-1 do
                for o=0 to layer.output_depth-1 do
                    let out_start = o * out_h * out_w in
                    let sum_bias = ref 0.0 in
                    for r=0 to out_h-1 do
                        for c=0 to out_w-1 do
                            let dz = dl_dz.(b).(out_start + r * out_w + c) in
                            sum_bias := !sum_bias +. dz;
                            for i=0 to layer.input_depth-1 do
                                let weight_offset = (o * layer.input_depth * layer.kernel_size * layer.kernel_size) + (i * layer.kernel_size * layer.kernel_size) in
                                let in_start = i * layer.input_height * layer.input_width in
                                for kr=0 to layer.kernel_size-1 do
                                    for kc=0 to layer.kernel_size-1 do
                                        gw_cpu.(0).(weight_offset + kr * layer.kernel_size + kc) <- gw_cpu.(0).(weight_offset + kr * layer.kernel_size + kc) +. dz *. inputs.(b).(in_start + (r+kr)*layer.input_width + (c+kc)) /. float batch_size;
                                        d_inputs.(b).(in_start + (r+kr)*layer.input_width + (c+kc)) <- d_inputs.(b).(in_start + (r+kr)*layer.input_width + (c+kc)) +. dz *. kernels_cpu.(0).(weight_offset + kr * layer.kernel_size + kc)
                                    done
                                done
                            done
                        done
                    done;
                    gb_cpu.(0).(o) <- gb_cpu.(0).(o) +. (!sum_bias /. float batch_size)
                done
            done;
            { Gradients.d_input = CPU d_inputs; d_weights = Gradients.Dense layer.grad_weights; d_bias = layer.grad_bias }
        end
      