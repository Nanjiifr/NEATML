open Neat
open Evolution
open Visualizer
open Graphics

(* On veut maximiser le fitness. 
   L'erreur max possible est 1.0 (car sigmoid sort entre 0 et 1).
   Donc Fitness = 1.0 - erreur.
*)
let f predicted_list real_list =
  let predicted = List.hd predicted_list and real = List.hd real_list in
  let error = abs_float (real -. predicted) in
  (* On met au carré l'erreur pour pénaliser fortement les gros écarts *)
  1.0 -. (error *. error)

let main () =
  Random.self_init ();
  (* Fenêtre graphique *)
  open_graph (" " ^ string_of_int 800 ^ "x" ^ string_of_int 600);
  set_window_title "NEAT Affine Test";

  let pop_size = 50 in
  (* Une population un peu plus grande aide *)
  (* 1 seule entrée réelle + le biais *)
  let number_inputs = 1 and number_outputs = 1 in
  let epochs = 100 in
  (* Plus d'époques pour laisser le temps de converger *)

  let innov_global =
    InnovationManager.create (number_outputs + number_inputs + 2)
  in

  (* Attention : create_pop prend le nombre d'inputs HORS biais. 
     Donc ici 1 input (le biais est ajouté auto dans create_pop) *)
  let pop = ref (create_pop pop_size 1 1 innov_global) in

  Printf.printf "Population créée avec 1 input + 1 biais\n";
  flush stdout;

  (* DATASET : f(x) = 0.5x + 0.2 
     Format d'entrée : [Biais; X] 
     Le Biais doit toujours être 1.0
  *)
  let dataset =
    [|
      ([ 1.; 0.0 ], [ 0.20 ]);
      (* 0.5*0 + 0.2 = 0.2 *)
      ([ 1.; 0.2 ], [ 0.30 ]);
      (* 0.5*0.2 + 0.2 = 0.3 *)
      ([ 1.; 0.4 ], [ 0.40 ]);
      ([ 1.; 0.6 ], [ 0.50 ]);
      ([ 1.; 0.8 ], [ 0.60 ]);
      ([ 1.; 1.0 ], [ 0.70 ]);
    |]
  in

  for epoch = 0 to epochs - 1 do
    Printf.printf "Epoch: %d\n" epoch;
    flush stdout;

    pop := generation !pop f dataset innov_global;

    let best_genome = List.hd !pop.genomes in

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
