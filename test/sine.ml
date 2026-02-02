open Neatml
open Neural.Layers
open Neural.Models
open Neural.Training
open Neural.Core

let create_dataset n =
  Array.init n (fun _ ->
      let seed = Random.float (2. *. Float.pi) in
      (seed, sin seed))

let () =
  Random.self_init ();

  (* 1. Enable GPU at the very beginning *)
  Utils.enable_gpu ();
  Printf.printf "GPU Acceleration Enabled: %b\n" !Utils.use_gpu;

  (* 2. Create a larger dataset to saturate the GPU *)
  let n_samples = 1000 in
  let dataset = create_dataset n_samples in

  let x_train = Tensor.CPU (Array.map (fun row -> [| fst row |]) dataset) in
  let y_train = Tensor.CPU (Array.map (fun row -> [| snd row |]) dataset) in

  let test_dataset = create_dataset 100 in
  let x_test = Tensor.CPU (Array.map (fun row -> [| fst row |]) test_dataset) in
  let y_test = Tensor.CPU (Array.map (fun row -> [| snd row |]) test_dataset) in

  (* 3. Use a larger model (Hidden layer: 512 neurons) 
     GPU overhead is only worth it for wider layers. *)
  let hidden_dim = 128 in
  let batch_size = 128 in

  let hidden_layer = Linear.create 1 hidden_dim batch_size Activations.ReLU in
  let output_layer = Linear.create hidden_dim 1 batch_size Activations.Tanh in

  let model =
    {
      Sequential.layers =
        [ Layer.Linear hidden_layer; Layer.Linear output_layer ];
    }
  in

  let lr = 0.001 in
  let optimizers = Optimizer.create lr Optimizer.Adam model in

  Printf.printf
    "Training SINE network on GPU (Samples: %d, Hidden: %d, Batch: %d)...\n"
    n_samples hidden_dim batch_size;
  flush stdout;

  (* Optimizer.fit now reports timing per epoch *)
  Optimizer.fit model x_train y_train x_test y_test batch_size 50 optimizers
    Errors.MSE;

  Printf.printf "Training complete. \n"

