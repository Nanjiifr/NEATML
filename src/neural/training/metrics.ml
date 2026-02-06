type classification_metrics = {
  accuracy : float;
  precision : float;
  recall : float;
  f1_score : float;
  confusion_matrix : float array array;
}

let print_metrics metrics dataset_name =
  let dim = Array.length metrics.confusion_matrix in
  Printf.printf "\n\027[1;36m=== Performance Report [%s] ===\027[0m\n" dataset_name;
  Printf.printf "------------------------------------------------\n";
  Printf.printf " \027[1;32mAccuracy\027[0m  : \027[1;37m%.2f%%\027[0m\n" (metrics.accuracy *. 100.);
  Printf.printf " \027[1;32mPrecision\027[0m : \027[1;37m%.2f%%\027[0m (Macro Avg)\n" (metrics.precision *. 100.);
  Printf.printf " \027[1;32mRecall\027[0m    : \027[1;37m%.2f%%\027[0m (Macro Avg)\n" (metrics.recall *. 100.);
  Printf.printf " \027[1;32mF1-Score\027[0m  : \027[1;37m%.2f%%\027[0m\n" (metrics.f1_score *. 100.);
  Printf.printf "------------------------------------------------\n";
  
  if dim <= 20 then begin
    Printf.printf "\n\027[1;33mConfusion Matrix:\027[0m\n";
    Printf.printf "      ";
    for i = 0 to dim - 1 do Printf.printf "%4d " i done;
    Printf.printf "\n";
    Array.iteri (fun i row ->
      Printf.printf "%4d |" i;
      Array.iter (fun x -> 
        if x > 0. then Printf.printf "\027[1;37m%4.0f\027[0m " x 
        else Printf.printf "\027[1;30m%4.0f\027[0m " x
      ) row;
      Printf.printf "|\n"
    ) metrics.confusion_matrix;
    Printf.printf "\n"
  end

let compute_metrics cm_cpu =
  let total = Array.fold_left (fun acc row -> acc +. Array.fold_left ( +. ) 0. row) 0. cm_cpu in
  let dim = Array.length cm_cpu in
  
  (* Accuracy: sum of diagonal / total *)
  let diagonal_sum = ref 0. in
  for i = 0 to dim - 1 do
    diagonal_sum := !diagonal_sum +. cm_cpu.(i).(i)
  done;
  let accuracy = if total > 0. then !diagonal_sum /. total else 0. in

  (* Precision, Recall per class *)
  let precisions = ref 0. in
  let recalls = ref 0. in
  
  for i = 0 to dim - 1 do
    let tp = cm_cpu.(i).(i) in
    let col_sum = ref 0. in
    let row_sum = ref 0. in
    for j = 0 to dim - 1 do
      col_sum := !col_sum +. cm_cpu.(j).(i); (* Sum of prediction i *)
      row_sum := !row_sum +. cm_cpu.(i).(j); (* Sum of target i *)
    done;
    
    let p = if !col_sum > 0. then tp /. !col_sum else 0. in
    let r = if !row_sum > 0. then tp /. !row_sum else 0. in
    
    precisions := !precisions +. p;
    recalls := !recalls +. r;
  done;

  let precision = !precisions /. (float_of_int dim) in
  let recall = !recalls /. (float_of_int dim) in
  let f1_score = 
    if (precision +. recall) > 0. 
    then 2. *. (precision *. recall) /. (precision +. recall) 
    else 0. 
  in

  { accuracy; precision; recall; f1_score; confusion_matrix = cm_cpu }

let evaluate_dataset model x y batch_size name =
  if not !Utils.use_gpu then failwith "Metrics evaluation requires GPU enabled.";
  
  let x_cpu = Utils.to_cpu x in
  let y_cpu = Utils.to_cpu y in
  let n_samples = Array.length x_cpu in
  let n_classes = Sequential.get_out_dim model in
  let n_batches = (n_samples + batch_size - 1) / batch_size in
  
  (* Initialize CM on GPU *)
  let cm_gpu = 
    let t = Utils.zeros n_classes n_classes in
    Utils.to_gpu t 
  in
  
  let cm_tensor_gpu = match cm_gpu with Tensor.GPU g -> g | _ -> failwith "Expected GPU tensor" in

  Printf.printf "[Evaluation] Processing %s (%d samples)...\n%!" name n_samples;

  for b = 0 to n_batches - 1 do
    let idx = b * batch_size in
    let current_bs = min batch_size (n_samples - idx) in
    
    let b_x = Array.sub x_cpu idx current_bs in
    let b_y = Array.sub y_cpu idx current_bs in
    
    let t_x = Tensor.GPU (Gpu.of_cpu b_x) in
    let t_y = Tensor.GPU (Gpu.of_cpu b_y) in
    
    let preds = Sequential.forward_seq model t_x in
    
    (* Update CM on GPU *)
    (match preds, t_y with
    | Tensor.GPU p_g, Tensor.GPU y_g ->
        Gpu.update_confusion_matrix p_g y_g cm_tensor_gpu
    | _ -> failwith "Tensors must be on GPU");

    (* Cleanup batch tensors *)
    (match t_x with Tensor.GPU g -> Gpu.release g | _ -> ());
    (match t_y with Tensor.GPU g -> Gpu.release g | _ -> ());
    (match preds with Tensor.GPU g -> Gpu.release g | _ -> ());
    
    if b mod 10 = 0 then Gpu.commit_batch ();
  done;

  (* Convert atomic CM to float CM for CPU reading *)
  let cm_float_gpu = Gpu.cm_to_float_tensor cm_tensor_gpu in
  let cm_cpu = Gpu.to_cpu cm_float_gpu in
  
  Gpu.release cm_tensor_gpu;
  Gpu.release cm_float_gpu;

  let metrics = compute_metrics cm_cpu in
  print_metrics metrics name;
  metrics

let evaluate model x_train y_train x_test y_test batch_size =
  Utils.enable_gpu ();
  Sequential.set_training_mode model false;
  let _ = evaluate_dataset model x_train y_train batch_size "Train Set" in
  let _ = evaluate_dataset model x_test y_test batch_size "Test Set" in
  ()