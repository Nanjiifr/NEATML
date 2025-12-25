open Neat
open Evolution
open Visualizer
open Graphics

let f predicted_list real_list =
  let predicted = List.hd predicted_list and real = List.hd real_list in
  let error = abs_float (real -. predicted) in
  1.0 /. (1.0 +. (error *. error))

let main () =
  Random.self_init ();
  (* Fenêtre graphique *)
  open_graph (" " ^ string_of_int 800 ^ "x" ^ string_of_int 600);
  set_window_title "NEAT Affine Test";

  let pop_size = 250 in
  let number_inputs = 1 and number_outputs = 1 in
  let epochs = 1000 in

  let innov_global =
    InnovationManager.create (number_outputs + number_inputs + 2)
  in

  let pop = ref (create_pop pop_size 1 1 innov_global) in
  let l_species = ref [] in

  Printf.printf "Population créée avec 1 input + 1 biais\n";
  flush stdout;

  (* DATASET : f(x) = 0.5x + 0.2 
     Format d'entrée : [Biais; X] 
     Le Biais doit toujours être 1.0
  *)
  let dataset =
    [|
      ([ 1.; 0.0 ], [ 0.20 ]);
      ([ 1.; 0.2 ], [ 0.30 ]);
      ([ 1.; 0.4 ], [ 0.40 ]);
      ([ 1.; 0.6 ], [ 0.50 ]);
      ([ 1.; 0.8 ], [ 0.60 ]);
      ([ 1.; 1.0 ], [ 0.70 ]);
    |]
  in

  for epoch = 0 to epochs - 1 do
    Printf.printf "Epoch: %d\n" epoch;
    flush stdout;

    let new_pop, new_sp = generation !pop !l_species dataset f innov_global in
    pop := new_pop;
    l_species := new_sp;

    let best_genome =
      List.fold_left
        (fun max_g g -> if g.fitness > max_g.fitness then g else max_g)
        (List.hd !pop.genomes) !pop.genomes
    in

    (* Affichage *)
    draw_genome best_genome;

    (* On affiche aussi dans la console le résultat du meilleur pour x=0 et x=1 *)
    let phenotype = create_phenotype best_genome in
    let pred_0 = List.hd (predict phenotype [ 1.; 0.0 ]) in
    let pred_1 = List.hd (predict phenotype [ 1.; 1.0 ]) in
    Printf.printf
      "  Best Fitness: %.4f | Pred(0.0): %.2f (Target 0.2) | Pred(1.0): %.2f \
       (Target 0.7)\n"
      best_genome.fitness pred_0 pred_1;
    flush stdout;

    (* Petit délai pour l'animation *)
    Unix.sleepf 0.05
  done;

  Printf.printf "Fini. Appuyez sur une touche.\n";
  ignore (read_key ());
  close_graph ()

let () = main ()
