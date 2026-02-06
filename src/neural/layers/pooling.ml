

type t = {
  kernel_size : int;
  stride : int;
  input_depth : int;
  input_height : int;
  input_width : int;
  mutable input_cache : Tensor.t option;
  mutable gpu_indices : Tensor.t option;
  mutable max_indices : (int array array) option; 
}

let create kernel_size stride input_depth input_height input_width =
  {
    kernel_size;
    stride;
    input_depth;
    input_height;
    input_width;
    input_cache = None;
    gpu_indices = None;
    max_indices = None;
  }

let get_output_dims l =
  let out_h = (l.input_height - l.kernel_size) / l.stride + 1 in
  let out_w = (l.input_width - l.kernel_size) / l.stride + 1 in
  (l.input_depth, out_h, out_w)

let forward (l : t) (input : Tensor.t) =
  match input with
  | Tensor.GPU g ->
      let n = g.Gpu.rows in
      let c = l.input_depth in
      let _, out_h, out_w = get_output_dims l in
      let out_t, idx_t = Gpu.maxpool_fwd g n c l.input_height l.input_width out_h out_w l.kernel_size l.kernel_size l.stride in
      l.gpu_indices <- Some (Tensor.GPU idx_t);
      Tensor.GPU out_t
  | Tensor.CPU input_cpu ->
      let batch_size = Array.length input_cpu in
      let channels, out_h, out_w = get_output_dims l in
      let out_dim = channels * out_h * out_w in
      
      let output = Array.make_matrix batch_size out_dim 0.0 in
      let indices = Array.make_matrix batch_size out_dim 0 in
      
      for b = 0 to batch_size - 1 do
        let sample = input_cpu.(b) in
        for c = 0 to channels - 1 do
          for i = 0 to out_h - 1 do
            for j = 0 to out_w - 1 do
              let h_start = i * l.stride in
              let w_start = j * l.stride in
              let h_end = min (h_start + l.kernel_size) l.input_height in
              let w_end = min (w_start + l.kernel_size) l.input_width in
              
              let max_val = ref neg_infinity in
              let max_idx = ref (-1) in
              
              for r = h_start to h_end - 1 do
                for k = w_start to w_end - 1 do
                  let idx = (c * l.input_height * l.input_width) + (r * l.input_width) + k in
                  let v = sample.(idx) in
                  if v > !max_val then (
                    max_val := v;
                    max_idx := idx
                  )
                done
              done;
              
              let out_idx = (c * out_h * out_w) + (i * out_w) + j in
              output.(b).(out_idx) <- !max_val;
              indices.(b).(out_idx) <- !max_idx
            done
          done
        done
      done;
      
      l.input_cache <- Some input; 
      l.max_indices <- Some indices;
      Tensor.CPU output

let backward (l : t) (grad_output : Tensor.t) =
  let dummy_bias = Utils.zeros 1 1 in
  let d_b = if !Utils.use_gpu then Utils.to_gpu dummy_bias else dummy_bias in
  
  match grad_output with
  | Tensor.GPU g ->
      let indices = match l.gpu_indices with Some (Tensor.GPU i) -> i | _ -> failwith "Pooling: No GPU indices" in
      let n = g.Gpu.rows in
      let c = l.input_depth in
      let _, out_h, out_w = get_output_dims l in
      let d_in = Gpu.maxpool_bwd g indices n c l.input_height l.input_width out_h out_w in
      { Gradients.d_input = Tensor.GPU d_in; d_weights = Gradients.Empty; d_bias = d_b }
  | Tensor.CPU grad_out_cpu ->
      match l.max_indices with
      | None -> failwith "Pooling: Backward called before forward"
      | Some indices ->
          let batch_size = Array.length grad_out_cpu in
          let in_dim = l.input_depth * l.input_height * l.input_width in
          let grad_input = Array.make_matrix batch_size in_dim 0.0 in
          let out_dim = Array.length indices.(0) in
          
          for b = 0 to batch_size - 1 do
            for i = 0 to out_dim - 1 do
              let max_idx = indices.(b).(i) in
              let grad = grad_out_cpu.(b).(i) in
              grad_input.(b).(max_idx) <- grad_input.(b).(max_idx) +. grad
            done
          done;
          
          { Gradients.d_input = Tensor.CPU grad_input; d_weights = Gradients.Empty; d_bias = d_b }

