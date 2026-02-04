open Neatml
open Neural.Core
open Neural.Layers
open Neural.Models
open Neural.Training

let () =
  Printf.printf "Testing Metrics Module...
";
  Utils.enable_gpu ();
  
  (* Create dummy data: 10 samples, 4 classes *)
  let n_samples = 20 in
  let n_classes = 4 in
  let input_dim = 5 in
  
  let x_data = Array.make_matrix n_samples input_dim 0.5 in
  let y_data = Array.make_matrix n_samples n_classes 0.0 in
  
  (* Fake one-hot targets *)
  for i = 0 to n_samples - 1 do
    y_data.(i).(i mod n_classes) <- 1.0
  done;
  
  let t_x = Tensor.CPU x_data |> Utils.to_gpu in
  let t_y = Tensor.CPU y_data |> Utils.to_gpu in
  
  (* Simple linear model *)
  let layer = Linear.create input_dim n_classes 1 Activations.Softmax in
  let model = { Sequential.layers = [ Layer.Linear layer ] } in
  
  (* Evaluate *)
  Metrics.evaluate model t_x t_y t_x t_y 5;
  
  Printf.printf "Test Complete.
"
