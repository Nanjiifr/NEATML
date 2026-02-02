open Utils

type t = {
  kernels : Tensor.t array array; (* depth x input_depth x k_h x k_w *)
  bias : float array;             (* depth *)
  grad_weights : Tensor.t array array;
  grad_bias : float array;
  mutable input_cache : Tensor.t option; (* (batch, flatten_input) *)
  activation : Activations.t;
  mutable preact_cache : Tensor.t option; (* (batch, flatten_output) *)
  input_depth : int;
  input_height : int;
  input_width : int;
  kernel_size : int;
  output_depth : int;
}

let create input_depth input_height input_width kernel_size output_depth activation =
  let init_k _ _ =
    let limit = sqrt (6.0 /. float ((input_depth * kernel_size * kernel_size) + (output_depth * kernel_size * kernel_size))) in
    let m = zeros kernel_size kernel_size in
    iter_matrix (fun _ i j -> m.(i).(j) <- Random.float (2.0 *. limit) -. limit) m;
    m
  in
  let kernels = Array.make_matrix output_depth input_depth (zeros kernel_size kernel_size) in
  for i=0 to output_depth-1 do
    for j=0 to input_depth-1 do
       kernels.(i).(j) <- init_k 0 0
    done
  done;
  
  let bias = Array.make output_depth 0.0 in
  let grad_weights = Array.make_matrix output_depth input_depth (zeros kernel_size kernel_size) in
  for i=0 to output_depth-1 do
    for j=0 to input_depth-1 do
       grad_weights.(i).(j) <- zeros kernel_size kernel_size
    done
  done;
  
  let grad_bias = Array.make output_depth 0.0 in
  {
    kernels; bias; grad_weights; grad_bias;
    input_cache=None; activation; preact_cache=None;
    input_depth; input_height; input_width; kernel_size; output_depth;
  }

let zero_grad layer =
  for i=0 to layer.output_depth-1 do
    for j=0 to layer.input_depth-1 do
      iter_matrix (fun _ r c -> layer.grad_weights.(i).(j).(r).(c) <- 0.0) layer.grad_weights.(i).(j)
    done;
    layer.grad_bias.(i) <- 0.0
  done

let forward layer inputs =
  layer.input_cache <- Some (copy_mat inputs);
  let batch_size = Array.length inputs in
  let out_h = layer.input_height - layer.kernel_size + 1 in
  let out_w = layer.input_width - layer.kernel_size + 1 in
  let output_dim = layer.output_depth * out_h * out_w in
  let outputs = Array.make_matrix batch_size output_dim 0.0 in
  
  for b=0 to batch_size-1 do
    let input_row = inputs.(b) in
    
    for o=0 to layer.output_depth-1 do
        let sum_map = zeros out_h out_w in
        for i=0 to layer.input_depth-1 do
            let map_i = zeros layer.input_height layer.input_width in
            let start_idx = i * layer.input_height * layer.input_width in
            for r=0 to layer.input_height-1 do
                for c=0 to layer.input_width-1 do
                    map_i.(r).(c) <- input_row.(start_idx + r * layer.input_width + c)
                done
            done;
            let conv_res = conv "valid" map_i layer.kernels.(o).(i) in
            copy_mat_inplace (add_matrices sum_map conv_res) sum_map
        done;
        
        let b_val = layer.bias.(o) in
        iter_matrix (fun x r c -> sum_map.(r).(c) <- x +. b_val) sum_map;
        
        let out_start = o * out_h * out_w in
        iter_matrix (fun x r c -> outputs.(b).(out_start + r * out_w + c) <- x) sum_map
    done
  done;
  layer.preact_cache <- Some (copy_mat outputs);
  Activations.activate layer.activation outputs

let backward layer upstream_grad =
  let d_act = Activations.derivative_pre layer.activation (Option.get layer.preact_cache) in
  let batch_size = Array.length upstream_grad in
  let output_dim = Array.length upstream_grad.(0) in
  
  let dl_dz = Array.make_matrix batch_size output_dim 0.0 in
  
  for b=0 to batch_size-1 do
     let jacobian = d_act.(b) in
     let grad = upstream_grad.(b) in
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
  
  let inputs = Option.get layer.input_cache in
  
  for b=0 to batch_size-1 do
      for o=0 to layer.output_depth-1 do
          let dz_map = zeros out_h out_w in
          let out_start = o * out_h * out_w in
          for r=0 to out_h-1 do
             for c=0 to out_w-1 do
                dz_map.(r).(c) <- dl_dz.(b).(out_start + r * out_w + c)
             done
          done;
          
          let sum_bias = ref 0.0 in
          iter_matrix (fun x _ _ -> sum_bias := !sum_bias +. x) dz_map;
          layer.grad_bias.(o) <- layer.grad_bias.(o) +. (!sum_bias /. float batch_size);
          
          for i=0 to layer.input_depth-1 do
             let in_map = zeros layer.input_height layer.input_width in
             let in_start = i * layer.input_height * layer.input_width in
             for r=0 to layer.input_height-1 do
                 for c=0 to layer.input_width-1 do
                     in_map.(r).(c) <- inputs.(b).(in_start + r * layer.input_width + c)
                 done
             done;
             
             let dw = conv "valid" in_map dz_map in
             let scaled_dw = scalar (1. /. float batch_size) dw in
             copy_mat_inplace (add_matrices layer.grad_weights.(o).(i) scaled_dw) layer.grad_weights.(o).(i);
             
             let k = layer.kernels.(o).(i) in
             let k_flipped = zeros layer.kernel_size layer.kernel_size in
             for r=0 to layer.kernel_size-1 do
                 for c=0 to layer.kernel_size-1 do
                     k_flipped.(r).(c) <- k.(layer.kernel_size - 1 - r).(layer.kernel_size - 1 - c)
                 done
             done;
             
             let din_part = conv "full" dz_map k_flipped in
             let in_start = i * layer.input_height * layer.input_width in
             iter_matrix (fun x r c -> 
                 let idx = in_start + r * layer.input_width + c in
                 d_inputs.(b).(idx) <- d_inputs.(b).(idx) +. x
             ) din_part
          done
      done
  done;
  
  { Gradients.d_input = d_inputs; d_weights = Gradients.Conv layer.grad_weights; d_bias = layer.grad_bias }