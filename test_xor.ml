open Neat
open Evolution
open Visualizer
open Graphics

(* Dimensions de la fenêtre *)
let width = 800
let height = 600

let f predicted_list real_list =
  let predicted = List.hd predicted_list and real = List.hd real_list in
  1. -. abs_float (real -. predicted)

let main () =
  (* Initialisation graphiques *)
  open_graph (" " ^ string_of_int width ^ "x" ^ string_of_int height);
  set_window_title "NEAT Visualizer";
  set_color color_bg;
  fill_rect 0 0 width height;
  Random.self_init ();
  let pop_size = 100 in
  let number_inputs = 2 and number_outputs = 1 in
  let epochs = 100 in
  let innov_global =
    InnovationManager.create (number_outputs + number_inputs + 2)
  in
  let pop = ref (create_pop pop_size 2 1 innov_global) in
  let dataset =
    [|
      ([ 1.; 0.; 0. ], [ 0. ]);
      ([ 1.; 0.; 1. ], [ 1. ]);
      ([ 1.; 1.; 0. ], [ 1. ]);
      ([ 1.; 1.; 1. ], [ 0. ]);
    |]
  in
  for epoch = 0 to epochs - 1 do
    pop := generation !pop f dataset innov_global;
    let best_genome = List.hd !pop.genomes in
    draw_genome best_genome;
    Printf.printf "Epoch: %d | Best fitness : %f\n" epoch best_genome.fitness;
    flush stdout;

    Unix.sleepf 0.05
  done;
  let best_genome = List.hd !pop.genomes in
  let phenotype = create_phenotype best_genome in
  Array.iter
    (fun (inputs, _) ->
      let predicted = List.hd (predict phenotype inputs) in
      Printf.printf "Predicted : %f\n" predicted;
      flush stdout)
    dataset

let () = main ()
