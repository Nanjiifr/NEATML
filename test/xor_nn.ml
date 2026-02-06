open Neatml
open Neural.Layers
open Neural.Models
open Neural.Training
open Neural.Core

let () = 
  (* Enable GPU if desired *)
  Utils.enable_gpu ();

  (* XOR Dataset: x1, x2, y *)
  let dataset =
    [| [| 0.; 0.; 0. |]; [| 0.; 1.; 1. |]; [| 1.; 0.; 1. |]; [| 1.; 1.; 0. |] |]
  in

  (* Split into Inputs (X) and Targets (Y) *)
  let x_train_data = Array.map (fun row -> Array.sub row 0 2) dataset in 
  let y_train_data = Array.map (fun row -> Array.sub row 2 1) dataset in 
  
  let x_train = Tensor.CPU x_train_data in 
  let y_train = Tensor.CPU y_train_data in 

  (* Create a simple Feed-Forward Network
     Input: 2 -> Hidden: 4 (Tanh) -> Output: 1 (Sigmoid) *)
  let hidden_layer = Linear.create 2 4 4 Activations.Tanh in 
  let output_layer = Linear.create 4 1 4 Activations.Sigmoid in 
  
  let model = { Sequential.layers = [ Layer.Linear hidden_layer; Layer.Linear output_layer ] } in

  (* Initialize Optimizer (Adam) *)
  let learning_rate = 0.01 in 
  let optimizers = Optimizer.create learning_rate Optimizer.Adam model in 

  (* Train the model *)
  Printf.printf "Training XOR Neural Network...\n";
  let _ = Optimizer.fit 
    model 
    x_train 
    y_train 
    x_train (* Test on same data for XOR *) 
    y_train 
    4 (* Batch size *) 
    1000 (* Epochs *) 
    optimizers 
    Errors.MSE
  in

  Printf.printf "Training complete.\n"