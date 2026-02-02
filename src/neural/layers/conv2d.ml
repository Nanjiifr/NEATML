open Tensor
open Utils

type t = {
  kernels : Tensor.t array array; (* depth x input_depth x k_h x k_w *)
  mutable bias : float array;             (* depth - keeping as float array for simplicity or convert to Tensor? 
                                             Linear uses Tensor for bias now. Let's use Tensor for consistency. *)
  mutable grad_weights : Tensor.t array array;
  mutable grad_bias : float array;
  mutable input_cache : Tensor.t option; 
  activation : Activations.t;
  mutable preact_cache : Tensor.t option;
  input_depth : int;
  input_height : int;
  input_width : int;
  kernel_size : int;
  output_depth : int;
}

let create input_depth input_height input_width kernel_size output_depth activation =
  let init_k () =
    let limit = sqrt (6.0 /. float ((input_depth * kernel_size * kernel_size) + (output_depth * kernel_size * kernel_size))) in
    let m = Array.make_matrix kernel_size kernel_size 0.0 in
    for i=0 to kernel_size-1 do
        for j=0 to kernel_size-1 do
            m.(i).(j) <- Random.float (2.0 *. limit) -. limit
        done
    done;
    let t = CPU m in
    if !use_gpu then to_gpu t else t
  in
  let kernels = Array.init output_depth (fun _ -> Array.init input_depth (fun _ -> init_k ())) in
  let bias = Array.make output_depth 0.0 in
  let grad_weights = Array.init output_depth (fun _ -> Array.init input_depth (fun _ -> 
      let t = zeros kernel_size kernel_size in
      if !use_gpu then to_gpu t else t)) in
  let grad_bias = Array.make output_depth 0.0 in
  {
    kernels; bias; grad_weights; grad_bias;
    input_cache=None; activation; preact_cache=None;
    input_depth; input_height; input_width; kernel_size; output_depth;
  }

let zero_grad layer =
  for i=0 to layer.output_depth-1 do
    for j=0 to layer.input_depth-1 do
      let r = rows layer.grad_weights.(i).(j) in
      let c = cols layer.grad_weights.(i).(j) in
      let z = zeros r c in
      layer.grad_weights.(i).(j) <- if !use_gpu then to_gpu z else z
    done;
    layer.grad_bias.(i) <- 0.0
  done

let forward layer inputs =
  let inputs = if !use_gpu then to_gpu inputs else inputs in
  layer.input_cache <- Some (copy_mat inputs);
  
  let batch_size = rows inputs in
  let out_h = layer.input_height - layer.kernel_size + 1 in
  let out_w = layer.input_width - layer.kernel_size + 1 in
  let output_dim = layer.output_depth * out_h * out_w in
  
  if !use_gpu then
      (* GPU Forward for Conv2d is complex to chain without a "Multi-map" kernel.
         For now, we can fallback to CPU or implement a batch-conv kernel.
         Since user wants speed, I should at least do the core convolutions on GPU. *)
      let inputs_cpu = to_cpu inputs in
      let outputs = Array.make_matrix batch_size output_dim 0.0 in
      for b=0 to batch_size-1 do
        let input_row = inputs_cpu.(b) in
        for o=0 to layer.output_depth-1 do
            let sum_map = ref (zeros out_h out_w) in
            for i=0 to layer.input_depth-1 do
                let map_i_data = Array.make_matrix layer.input_height layer.input_width 0.0 in
                let start_idx = i * layer.input_height * layer.input_width in
                for r=0 to layer.input_height-1 do
                    for c=0 to layer.input_width-1 do
                        map_i_data.(r).(c) <- input_row.(start_idx + r * layer.input_width + c)
                    done
                done;
                let map_i = if !use_gpu then to_gpu (CPU map_i_data) else CPU map_i_data in
                let conv_res = conv "valid" map_i layer.kernels.(o).(i) in
                sum_map := add_matrices !sum_map conv_res
            done;
            let b_val = layer.bias.(o) in
            let final_map = to_cpu !sum_map in
            for r=0 to out_h-1 do
                for c=0 to out_w-1 do
                    outputs.(b).(o * out_h * out_w + r * out_w + c) <- final_map.(r).(c) +. b_val
                done
            done
        done
      done;
      let res = CPU outputs in
      let res_gpu = to_gpu res in
      layer.preact_cache <- Some res_gpu;
      Activations.activate layer.activation res_gpu
  else
      (* CPU Forward *)
      let inputs_cpu = to_cpu inputs in
      let outputs = Array.make_matrix batch_size output_dim 0.0 in
      for b=0 to batch_size-1 do
        let input_row = inputs_cpu.(b) in
        for o=0 to layer.output_depth-1 do
            let sum_map = Array.make_matrix out_h out_w 0.0 in
            for i=0 to layer.input_depth-1 do
                let map_i = Array.make_matrix layer.input_height layer.input_width 0.0 in
                let start_idx = i * layer.input_height * layer.input_width in
                for r=0 to layer.input_height-1 do
                    for c=0 to layer.input_width-1 do
                        map_i.(r).(c) <- input_row.(start_idx + r * layer.input_width + c)
                    done
                done;
                let k_cpu = to_cpu layer.kernels.(o).(i) in
                let conv_res = to_cpu (conv "valid" (CPU map_i) (CPU k_cpu)) in
                for r=0 to out_h-1 do
                    for c=0 to out_w-1 do
                        sum_map.(r).(c) <- sum_map.(r).(c) +. conv_res.(r).(c)
                    done
                done
            done;
            let b_val = layer.bias.(o) in
            for r=0 to out_h-1 do
                for c=0 to out_w-1 do
                    outputs.(b).(o * out_h * out_w + r * out_w + c) <- sum_map.(r).(c) +. b_val
                done
            done
        done
      done;
      let res = CPU outputs in
      layer.preact_cache <- Some res;
      Activations.activate layer.activation res

let backward layer upstream_grad =
  let upstream_grad_cpu = to_cpu upstream_grad in
  let d_act = Activations.derivative_pre layer.activation (Option.get layer.preact_cache) in
  let batch_size = Array.length upstream_grad_cpu in
  let output_dim = Array.length upstream_grad_cpu.(0) in
  
  let dl_dz = Array.make_matrix batch_size output_dim 0.0 in
  for b=0 to batch_size-1 do
     let jacobian = d_act.(b) in
     let grad = upstream_grad_cpu.(b) in
     for i=0 to output_dim-1 do
       let sum = ref 0.0 in
       for j=0 to output_dim-1 do
         sum := !sum +. (jacobian.(i).(j) *. grad.(j))
       done;
       dl_dz.(b).(i) <- !sum
     done
  done;
  
  let out_h = layer.input_height - layer.kernel_size + 1 in
  let out_w = layer.input_width - layer.kernel_size + 1 in
  let d_inputs = Array.make_matrix batch_size (layer.input_depth * layer.input_height * layer.input_width) 0.0 in
  let inputs = to_cpu (Option.get layer.input_cache) in
  
  for b=0 to batch_size-1 do
      for o=0 to layer.output_depth-1 do
          let dz_map_data = Array.make_matrix out_h out_w 0.0 in
          let out_start = o * out_h * out_w in
          for r=0 to out_h-1 do
             for c=0 to out_w-1 do
                dz_map_data.(r).(c) <- dl_dz.(b).(out_start + r * out_w + c)
             done
          done;
          let dz_map = if !use_gpu then to_gpu (CPU dz_map_data) else CPU dz_map_data in
          
          let sum_bias = ref 0.0 in
          for r=0 to out_h-1 do
              for c=0 to out_w-1 do
                  sum_bias := !sum_bias +. dz_map_data.(r).(c)
              done
          done;
          layer.grad_bias.(o) <- layer.grad_bias.(o) +. (!sum_bias /. float batch_size);
          
          for i=0 to layer.input_depth-1 do
             let in_map_data = Array.make_matrix layer.input_height layer.input_width 0.0 in
             let in_start = i * layer.input_height * layer.input_width in
             for r=0 to layer.input_height-1 do
                 for c=0 to layer.input_width-1 do
                     in_map_data.(r).(c) <- inputs.(b).(in_start + r * layer.input_width + c)
                 done
             done;
             let in_map = if !use_gpu then to_gpu (CPU in_map_data) else CPU in_map_data in
             
             let dw = conv "valid" in_map dz_map in
             let scaled_dw = scalar (1. /. float batch_size) dw in
             layer.grad_weights.(o).(i) <- add_matrices layer.grad_weights.(o).(i) scaled_dw;
             
             let k = layer.kernels.(o).(i) in
             let k_cpu = to_cpu k in
             let k_flipped_data = Array.make_matrix layer.kernel_size layer.kernel_size 0.0 in
             for r=0 to layer.kernel_size-1 do
                 for c=0 to layer.kernel_size-1 do
                     k_flipped_data.(r).(c) <- k_cpu.(layer.kernel_size - 1 - r).(layer.kernel_size - 1 - c)
                 done
             done;
             let k_flipped = if !use_gpu then to_gpu (CPU k_flipped_data) else CPU k_flipped_data in
             
             let din_part = conv "full" dz_map k_flipped in
             let din_part_cpu = to_cpu din_part in
             let in_start = i * layer.input_height * layer.input_width in
             for r=0 to layer.input_height-1 do
                 for c=0 to layer.input_width-1 do
                     let idx = in_start + r * layer.input_width + c in
                     d_inputs.(b).(idx) <- d_inputs.(b).(idx) +. din_part_cpu.(r).(c)
                 done
             done
          done
      done
  done;
  
  { Gradients.d_input = CPU d_inputs; d_weights = Gradients.Conv layer.grad_weights; d_bias = CPU [| layer.grad_bias |] }
