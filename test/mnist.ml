open Neatml
open Neural.Core

let read_int32 ic =
  let b1 = input_byte ic in
  let b2 = input_byte ic in
  let b3 = input_byte ic in
  let b4 = input_byte ic in
  (b1 lsl 24) lor (b2 lsl 16) lor (b3 lsl 8) lor b4

let load_images filename =
  let ic = open_in_bin filename in
  try
    let magic = read_int32 ic in
    if magic <> 2051 then
      failwith (Printf.sprintf "Invalid magic number for images in %s" filename);
    let count = read_int32 ic in
    let rows = read_int32 ic in
    let cols = read_int32 ic in
    let size = rows * cols in
    let images = Array.make count [||] in
    for i = 0 to count - 1 do
      let image = Array.make size 0.0 in
      for j = 0 to size - 1 do
        image.(j) <- float_of_int (input_byte ic) /. 255.0
      done;
      images.(i) <- image
    done;
    close_in ic;
    images
  with e ->
    close_in_noerr ic;
    raise e

let load_labels filename =
  let ic = open_in_bin filename in
  try
    let magic = read_int32 ic in
    if magic <> 2049 then
      failwith (Printf.sprintf "Invalid magic number for labels in %s" filename);
    let count = read_int32 ic in
    let labels = Array.make count [||] in
    for i = 0 to count - 1 do
      let label = input_byte ic in
      let one_hot = Array.make 10 0.0 in
      one_hot.(label) <- 1.0;
      labels.(i) <- one_hot
    done;
    close_in ic;
    labels
  with e ->
    close_in_noerr ic;
    raise e

let load_mnist_data () =
  let train_images_path = "test/mnist_data/train-images.idx3-ubyte" in
  let train_labels_path = "test/mnist_data/train-labels.idx1-ubyte" in
  let test_images_path = "test/mnist_data/t10k-images.idx3-ubyte" in
  let test_labels_path = "test/mnist_data/t10k-labels.idx1-ubyte" in

  Printf.printf "[MNIST] Loading training data...\n";
  let x_train_cpu = load_images train_images_path in
  let y_train_cpu = load_labels train_labels_path in

  Printf.printf "[MNIST] Loading test data...\n";
  let x_test_cpu = load_images test_images_path in
  let y_test_cpu = load_labels test_labels_path in

  let to_tensor data =
    let t = Tensor.CPU data in
    if !Utils.use_gpu then Utils.to_gpu t else t
  in

  ( to_tensor x_train_cpu,
    to_tensor y_train_cpu,
    to_tensor x_test_cpu,
    to_tensor y_test_cpu )

let visualize_results train_losses val_losses x_test model =
  let open Matplotlib in
  let epochs = List.length train_losses in

  (* 1. Plot Losses *)
  let x_axis = Array.init epochs (fun i -> float_of_int (i + 1)) in
  let y_train_plot = Array.of_list train_losses in
  let y_val_plot = Array.of_list val_losses in

  Pyplot.plot ~xs:x_axis ~label:"Train Loss" y_train_plot;
  Pyplot.plot ~xs:x_axis ~label:"Validation Loss" y_val_plot;
  Pyplot.legend ();
  Pyplot.xlabel "Epochs";
  Pyplot.ylabel "Loss";
  Pyplot.title "Training vs Validation Loss";
  Mpl.show ();

  (* 2. Show 10 Random Predictions *)
  let test_cpu = Utils.to_cpu x_test in
  let n_test = Array.length test_cpu in

  let argmax arr =
    let max_val = ref neg_infinity in
    let max_idx = ref (-1) in
    Array.iteri
      (fun i v ->
        if v > !max_val then (
          max_val := v;
          max_idx := i))
      arr;
    !max_idx
  in

  let fig = Fig.create () in
  for i = 0 to 9 do
    let idx = Random.int n_test in
    let img_flat = test_cpu.(idx) in

    (* Reshape 784 -> 28x28 *)
    let img_2d = Array.make_matrix 28 28 0.0 in
    for r = 0 to 27 do
      for c = 0 to 27 do
        img_2d.(r).(c) <- img_flat.((r * 28) + c)
      done
    done;

    (* Predict *)
    let input_batch = [| img_flat |] in
    let t_in =
      if !Utils.use_gpu then Tensor.GPU (Gpu.of_cpu input_batch)
      else Tensor.CPU input_batch
    in
    let pred_t = Sequential.forward_seq model t_in in
    let pred_cpu = (Utils.to_cpu pred_t).(0) in

    (match t_in with Tensor.GPU g -> Gpu.release g | _ -> ());
    (match pred_t with Tensor.GPU g -> Gpu.release g | _ -> ());

    let pred_label = argmax pred_cpu in

    let ax = Fig.add_subplot fig ~nrows:2 ~ncols:5 ~index:(i + 1) in
    Ax.imshow ax ~cmap:"gray" (Imshow_data.scalar Imshow_data.float img_2d);
    Ax.set_title ax (Printf.sprintf "Pred: %d" pred_label);
    Ax.Expert.to_pyobject ax |> ignore
  done;
  Mpl.show ()

let _main_mlp () =
  Random.self_init ();

  Utils.enable_gpu ();
  Printf.printf
    "\027[1;32m[SYSTEM]\027[0m GPU Acceleration Enabled: \027[1;33m%b\027[0m\n"
    !Utils.use_gpu;
  let x_train, y_train, x_test, y_test = load_mnist_data () in
  let input_size = 28 * 28 in
  let output_size = 10 in
  let hidden_size = 128 in
  let batch_size = 256 in

  let hidden1 =
    Linear.create input_size hidden_size batch_size Activations.ReLU
  in
  let output_layer =
    Linear.create hidden_size output_size batch_size Activations.Sigmoid
  in

  let model =
    {
      Sequential.layers =
        [
          Layer.Linear hidden1;
          Layer.Dropout (Dropout.create 0.25);
          Layer.Linear output_layer;
        ];
    }
  in

  Sequential.summary model;

  Printf.printf "Evaluating initial state...\n%!";

  let lr = 0.001 in
  let optimizer = Optimizer.create lr Optimizer.Adam model in
  let epochs = 25 in

  (* Use Optimizer.fit and get history *)
  let train_losses, val_losses =
    Optimizer.fit model x_train y_train x_test y_test batch_size epochs
      optimizer Errors.CROSS_ENTROPY
  in

  Metrics.evaluate model x_train y_train x_test y_test batch_size;
  visualize_results train_losses val_losses x_test model

let _main_cnn () =
  Random.self_init ();

  Utils.enable_gpu ();
  Printf.printf
    "\027[1;32m[SYSTEM]\027[0m GPU Acceleration Enabled: \027[1;33m%b\027[0m\n"
    !Utils.use_gpu;

  let x_train, y_train, x_test, y_test = load_mnist_data () in
  let batch_size = 128 in

  (* Architecture: Conv(1, 28, 28, 5, 8) -> MaxPool(2, 2) -> Linear -> Linear *)
  let conv1 = Conv2d.create 1 28 28 5 8 Activations.ReLU in
  let out_h = 28 - 5 + 1 in
  let out_w = 28 - 5 + 1 in
  
  let pool = Pooling.create 2 2 8 out_h out_w in
  let p_d, p_h, p_w = Pooling.get_output_dims pool in
  let conv_out_dim = p_d * p_h * p_w in

  let hidden = Linear.create conv_out_dim 64 batch_size Activations.ReLU in
  let output = Linear.create 64 10 batch_size Activations.Sigmoid in

  let model =
    {
      Sequential.layers =
        [
          Layer.Conv2d conv1;
          Layer.MaxPool2d pool;
          Layer.Dropout (Dropout.create 0.2);
          Layer.Linear hidden;
          Layer.Linear output;
        ];
    }
  in

  Sequential.summary model;

  Printf.printf "\n\027[1;34m[INFO]\027[0m Training CNN on MNIST...\n%!";

  let lr = 0.001 in
  let optimizer = Optimizer.create lr Optimizer.Adam model in

  let train_losses, val_losses =
    Optimizer.fit model x_train y_train x_test y_test batch_size 10 optimizer
      Errors.CROSS_ENTROPY
  in

  Metrics.evaluate model x_train y_train x_test y_test batch_size;
  visualize_results train_losses val_losses x_test model

let () =
  let lib_path =
    "/opt/homebrew/opt/python@3.14/Frameworks/Python.framework/Versions/3.14/lib/libpython3.14.dylib"
  in
  if Sys.file_exists lib_path then
    Py.initialize ~library_name:lib_path ~version:3 ()
  else Py.initialize ~version:3 ();
  _main_cnn ()
