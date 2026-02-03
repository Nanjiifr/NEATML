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

let () =
  Random.self_init ();

  Utils.enable_gpu ();
  Printf.printf
    "\027[1;32m[SYSTEM]\027[0m GPU Acceleration Enabled: \027[1;33m%b\027[0m\n"
    !Utils.use_gpu
