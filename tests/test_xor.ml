open Neat
open Evolution
open Visualizer
open Graphics

let f_xor predicted_list real_list =
  let predicted = List.hd predicted_list in
  let target = List.hd real_list in
  let error = abs_float (target -. predicted) in

  (* Si l'erreur est petite (< 0.5), le score est haut. 
     Si l'erreur est grande (ex: 0.98 pour target 0), le score s'effondre vers 0. *)
  let score = 1.0 -. (error *. error) in
  max 0. score

(* Vérification plus robuste *)
let check_success phenotype =
  let cases =
    [
      ([ 1.; 0.; 0. ], 0.);
      ([ 1.; 0.; 1. ], 1.);
      ([ 1.; 1.; 0. ], 1.);
      ([ 1.; 1.; 1. ], 0.);
    ]
  in
  List.for_all
    (fun (input, target) ->
      let pred = List.hd (predict phenotype input) in
      let output_binary = if pred >= 0.5 then 1. else 0. in
      output_binary = target)
    cases

let main () =
  Random.self_init ();
  (try open_graph (" " ^ string_of_int 800 ^ "x" ^ string_of_int 600)
   with _ -> ());
  (* Ignore si déjà ouvert *)
  set_window_title "NEAT XOR Test";

  let pop_size = 150 in
  (* XOR : 2 entrées réelles + Biais (géré par le code inputs=2) *)
  let number_inputs = 2 in
  let number_outputs = 1 in
  let max_epochs = 300 in
  (* XOR peut prendre un peu de temps *)

  (* Initialisation de l'innovation globale *)
  (* IDs réservés : 0 (Biais), 1-2 (Inputs), 3 (Output) -> Prochaine innov = 4 *)
  let innov_global =
    InnovationManager.create (number_outputs + number_inputs + 10)
  in

  (* Création de la population avec 2 inputs *)
  let pop =
    ref (create_pop pop_size number_inputs number_outputs innov_global)
  in
  let l_species = ref [] in

  Printf.printf "Population XOR créée (2 inputs + 1 biais, 1 output)\n";
  flush stdout;

  (* DATASET XOR 
     Format : [Biais; In1; In2], [Target]
  *)
  let dataset =
    [|
      ([ 1.; 0.; 0. ], [ 0. ]);
      (* 0 XOR 0 -> 0 *)
      ([ 1.; 0.; 1. ], [ 1. ]);
      (* 0 XOR 1 -> 1 *)
      ([ 1.; 1.; 0. ], [ 1. ]);
      (* 1 XOR 0 -> 1 *)
      ([ 1.; 1.; 1. ], [ 0. ]);
      (* 1 XOR 1 -> 0 *)
    |]
  in

  let solved = ref false in
  let epoch = ref 0 in

  while !epoch < max_epochs && not !solved do
    let new_pop, new_sp =
      generation !pop !l_species dataset f_xor innov_global
    in
    pop := new_pop;
    l_species := new_sp;

    (* Récupérer le meilleur global *)
    let best_genome =
      List.fold_left
        (fun acc g -> if g.fitness > acc.fitness then g else acc)
        (List.hd !pop.genomes) !pop.genomes
    in

    (* Visualisation *)
    draw_genome best_genome;

    (* Test de la solution *)
    let phenotype = create_phenotype best_genome in

    (* Affichage console tous les 10 epochs ou si résolu *)
    if !epoch mod 10 = 0 || best_genome.fitness > 3.8 then begin
      let p00 = List.hd (predict phenotype [ 1.; 0.; 0. ]) in
      let p01 = List.hd (predict phenotype [ 1.; 0.; 1. ]) in
      let p10 = List.hd (predict phenotype [ 1.; 1.; 0. ]) in
      let p11 = List.hd (predict phenotype [ 1.; 1.; 1. ]) in

      Printf.printf "Epoch %d | Best Fit: %.4f (Max 4.0)\n" !epoch
        best_genome.fitness;
      Printf.printf "  [0,0]->%.2f | [0,1]->%.2f | [1,0]->%.2f | [1,1]->%.2f\n"
        p00 p01 p10 p11;
      Printf.printf "  Hidden nodes: %d | Connections: %d\n"
        (List.length best_genome.nodes - (number_inputs + 1 + number_outputs))
        (List.length best_genome.connections);
      flush stdout
    end;

    if check_success phenotype then begin
      solved := true;
      set_window_title ("NEAT XOR - RESOLU (Epoch " ^ string_of_int !epoch ^ ")");

      (* 1. VISUEL : On dessine le champion *)
      draw_genome best_genome;

      (* 2. CONSOLE : On affiche les prédictions précises *)
      Printf.printf "\n\n===========================================\n";
      Printf.printf "       SOLUTION VALIDÉE (Epoch %d)         \n" !epoch;
      Printf.printf "===========================================\n";
      Printf.printf "Fitness Finale : %.4f\n" best_genome.fitness;
      Printf.printf "  Hidden nodes: %d | Connections: %d\n"
        (List.length best_genome.nodes - (number_inputs + 1 + number_outputs))
        (List.length best_genome.connections);
      Printf.printf "-------------------------------------------\n";

      let test_cases =
        [
          ([ 1.; 0.; 0. ], 0., "0 XOR 0");
          ([ 1.; 0.; 1. ], 1., "0 XOR 1");
          ([ 1.; 1.; 0. ], 1., "1 XOR 0");
          ([ 1.; 1.; 1. ], 0., "1 XOR 1");
        ]
      in

      List.iter
        (fun (inputs, target, label) ->
          let raw_output = List.hd (predict phenotype inputs) in
          (* Classification binaire pour l'affichage *)
          let interpretation = if raw_output >= 0.5 then 1 else 0 in

          Printf.printf " %s (Cible %.0f) -> Output: %.5f  [Interpreté: %d]\n"
            label target raw_output interpretation)
        test_cases;

      Printf.printf "===========================================\n";
      flush stdout
    end
    else begin
      (* Si pas encore résolu, on met à jour le dessin pour voir l'évolution *)
      draw_genome best_genome
    end;
    incr epoch;
    Unix.sleepf 0.01
  done;

  if not !solved then
    Printf.printf "\nEchec de convergence après %d epochs.\n" max_epochs;

  Printf.printf "Appuyez sur une touche pour quitter.\n";
  ignore (read_key ());
  close_graph ()

let () = main ()
