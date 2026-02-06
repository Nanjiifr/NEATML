open Neatml
open Types
open Neural.Layers
open Neural.Models
open Neural.Training
open Neural.Core

let create_dataset n =
  Array.init n (fun _ ->
      let seed = Random.float (2. *. Float.pi) in
      (seed, sin seed))

let _main_neat () =
  Random.self_init ();

  let train_dataset =
    let n = 100 in
    Array.init n (fun i ->
        let x = float i /. float (n - 1) *. (2. *. Float.pi) in
        (x, sin x))
  in

  let evaluator g =
    let nn = Phenotype.create_phenotype g in
    let error = ref 0. in
    Array.iter
      (fun (x, sx) ->
        let x_norm = x /. (2. *. Float.pi) in
        Phenotype.reset_network nn;
        let pred =
          match Phenotype.predict nn [ x_norm ] with h :: _ -> h | [] -> 0.
        in
        error := !error +. ((sx -. pred) *. (sx -. pred)))
      train_dataset;
    let mse = !error /. float (Array.length train_dataset) in
    4. -. mse
  in
  let input_size = 1 in
  let output_size = 1 in

  let pop_size = 300 in
  let epochs = 100 in

  let innov = Innovation.create (input_size + output_size + 5) in
  let pop = ref (Evolution.create_pop pop_size input_size output_size innov) in
  let species = ref [] in
  let threshold = ref 3. in

  let _start_time = Unix.gettimeofday () in
  let total_evals = ref 0 in

  let best_genome = ref (List.hd !pop.genomes) in

  for epoch = 0 to epochs - 1 do
    total_evals := !total_evals + pop_size;
    let new_pop, new_species, evaluated_genomes =
      Evolution.generation !pop !species evaluator innov threshold ()
    in
    pop := new_pop;
    species := new_species;

    (* Update best genome from the evaluated pool *)
    List.iter
      (fun g -> if g.fitness > !best_genome.fitness then best_genome := g)
      evaluated_genomes;

    Evolution.print_pop_summary !pop !species epoch epochs
  done;

  let nn = Phenotype.create_phenotype !best_genome in

  (* Plotting with Matplotlib *)
  let open Matplotlib in
  let n_plot = 200 in
  let xs =
    Array.init n_plot (fun i ->
        float i /. float (n_plot - 1) *. (2. *. Float.pi))
  in
  let ys_real = Array.map sin xs in
  let ys_pred =
    Array.map
      (fun x ->
        let x_norm = x /. (2. *. Float.pi) in
        Phenotype.reset_network nn;
        match Phenotype.predict nn [ x_norm ] with h :: _ -> h | [] -> 0.)
      xs
  in

  Pyplot.plot ~xs ~label:"Reality (sin x)" ys_real;
  Pyplot.plot ~xs ~label:"Prediction" ys_pred;
  Pyplot.legend ();
  Pyplot.xlabel "x";
  Pyplot.ylabel "sin(x)";
  Pyplot.title "SINE NEAT: Reality vs Prediction";
  Mpl.show ();

  Visualizer.draw_genome !best_genome

let _main () =
  Random.self_init ();

  (* 1. Enable GPU at the very beginning *)
  Utils.enable_gpu ();
  Printf.printf
    "\027[1;32m[SYSTEM]\027[0m GPU Acceleration Enabled: \027[1;33m%b\027[0m\n"
    !Utils.use_gpu;

  (* 2. Create a larger dataset to saturate the GPU *)
  let n_samples = 10000 in
  let dataset = create_dataset n_samples in

  let x_train =
    let t = Tensor.CPU (Array.map (fun row -> [| fst row |]) dataset) in
    if !Utils.use_gpu then Utils.to_gpu t else t
  in
  let y_train =
    let t = Tensor.CPU (Array.map (fun row -> [| snd row |]) dataset) in
    if !Utils.use_gpu then Utils.to_gpu t else t
  in

  let test_dataset = create_dataset 200 in
  let x_test =
    let t = Tensor.CPU (Array.map (fun row -> [| fst row |]) test_dataset) in
    if !Utils.use_gpu then Utils.to_gpu t else t
  in
  let y_test =
    let t = Tensor.CPU (Array.map (fun row -> [| snd row |]) test_dataset) in
    if !Utils.use_gpu then Utils.to_gpu t else t
  in

  (* 3. Use a larger model (Hidden layer: 128 neurons) *)
  let hidden_dim = 256 in
  let batch_size = 256 in

  let hidden_layer = Linear.create 1 hidden_dim batch_size Activations.ReLU in
  let output_layer = Linear.create hidden_dim 1 batch_size Activations.Tanh in

  let model =
    {
      Sequential.layers =
        [ Layer.Linear hidden_layer; Layer.Linear output_layer ];
    }
  in

  Sequential.summary model;

  let lr = 0.001 in
  let optimizers = Optimizer.create lr Optimizer.Adam model in

  Printf.printf
    "\027[1;34m[INFO]\027[0m Training SINE network (Samples: %d, Hidden: %d, \
     Batch: %d)...\n"
    n_samples hidden_dim batch_size;
  flush stdout;

  (* Optimizer.fit now reports timing per epoch *)
  let _ =
    Optimizer.fit model x_train y_train x_test y_test batch_size 500 optimizers
      Errors.MSE
  in

  Printf.printf "\027[1;32m[SUCCESS]\027[0m Training complete. \n";

  (* Plotting with Matplotlib *)
  let open Matplotlib in
  let n_plot = 200 in
  let xs =
    Array.init n_plot (fun i ->
        float i /. float (n_plot - 1) *. (2. *. Float.pi))
  in
  let ys_real = Array.map sin xs in
  let ys_pred =
    Array.map
      (fun x ->
        let x_tensor = Tensor.CPU [| [| x |] |] in
        let x_gpu =
          if !Utils.use_gpu then Utils.to_gpu x_tensor else x_tensor
        in
        let pred_tensor = Sequential.forward_seq model x_gpu in
        let pred = (Utils.to_cpu pred_tensor).(0).(0) in
        (match pred_tensor with Tensor.GPU g -> Gpu.release g | _ -> ());
        (match x_gpu with Tensor.GPU g -> Gpu.release g | _ -> ());
        pred)
      xs
  in

  Pyplot.plot ~xs ~label:"Reality (sin x)" ys_real;
  Pyplot.plot ~xs ~label:"Prediction" ys_pred;
  Pyplot.legend ();
  Pyplot.xlabel "x";
  Pyplot.ylabel "sin(x)";
  Pyplot.title "SINE Network: Reality vs Prediction";
  Mpl.show ()

let () =
  let lib_path =
    "/opt/homebrew/opt/python@3.14/Frameworks/Python.framework/Versions/3.14/lib/libpython3.14.dylib"
  in
  if Sys.file_exists lib_path then
    Py.initialize ~library_name:lib_path ~version:3 ()
  else Py.initialize ~version:3 ();
  _main ()
