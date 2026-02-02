open Neatml
open Neural.Layers
open Neural.Models
open Neural.Training

let create_dataset n =
  Array.init n (fun _ ->
      let seed = Random.float (2. *. Float.pi) in
      (seed, sin seed))

let () =
  Random.self_init ();
  let dataset = create_dataset 10000 in

  let x_train = Array.map (fun row -> [| fst row |]) dataset in
  let y_train = Array.map (fun row -> [| snd row |]) dataset in

  let test_dataset = create_dataset 10 in
  let x_test = Array.map (fun row -> [| fst row |]) test_dataset in
  let y_test = Array.map (fun row -> [| snd row |]) test_dataset in

  let hidden_layer = Linear.create 1 128 32 Activations.ReLU in
  let output_layer = Linear.create 128 1 32 Activations.Tanh in

  let model =
    {
      Sequential.layers =
        [ Layer.Linear hidden_layer; Layer.Linear output_layer ];
    }
  in

  let lr = 0.01 in
  let optimizers = Optimizer.create lr Optimizer.Adam model in

  Printf.printf "Training SINE network... \n";
  Optimizer.fit model x_train y_train x_test y_test 32 100 optimizers Errors.MSE;

  Printf.printf "Training complete. \n"
