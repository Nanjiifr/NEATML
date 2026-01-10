open Neatml
open Neatml.Types

exception Break

let predict_sequence (nn : network) (word : string) =
  Phenotype.reset_network nn;

  (* Reset au début du mot *)
  let last_output = ref 0.0 in
  let len = String.length word in

  String.iteri
    (fun i c ->
      let inputs =
        match c with
        | 'a' -> [ 1.0; 0.0 ]
        | 'b' -> [ 0.0; 1.0 ]
        | _ -> [ 0.0; 0.0 ]
      in

      let res = Phenotype.predict nn inputs in

      if i = len - 1 then
        match res with
        | [ r ] -> last_output := r
        | _ -> failwith "Erreur output size")
    word;

  !last_output

let generate_dataset n_max =
  let dataset = ref [] in
  for n = 1 to n_max do
    (* Cas positifs : a^n b^n *)
    let pos = String.make n 'a' ^ String.make n 'b' in
    dataset := (pos, 1.0) :: (pos, 1.0) :: (pos, 1.0) :: !dataset;

    (* Cas négatifs : a^n b^(n+1) ou a^(n+1) b^n *)
    let neg1 = String.make n 'a' ^ String.make (n + 1) 'b' in
    let neg2 = String.make (n + 1) 'a' ^ String.make n 'b' in
    dataset := (neg1, 0.0) :: (neg2, 0.0) :: !dataset;

    (* Cas négatifs : mauvais ordre (b^n a^n) *)
    let wrong_order = String.make n 'b' ^ String.make n 'a' in
    dataset := (wrong_order, 0.0) :: !dataset
  done;
  !dataset

let evaluate_genome n_max genome =
  let nn = Phenotype.create_phenotype genome in
  let dataset = generate_dataset n_max in

  let error =
    List.fold_left
      (fun acc (word, expected) ->
        let result = predict_sequence nn word in
        acc +. abs_float (expected -. result))
      0.0 dataset
  in

  1. /. (1. +. error)

let evaluate_genome_test n_max genome =
  let nn = Phenotype.create_phenotype genome in
  let dataset = generate_dataset n_max in

  let true_positives = ref 0
  and true_negatives = ref 0
  and false_positives = ref 0
  and false_negatives = ref 0 in

  let error =
    List.fold_left
      (fun acc (word, expected) ->
        let result = predict_sequence nn word in
        let expected_bool = if expected > 0.5 then true else false in
        let is_in = if result > 0.5 then true else false in
        if is_in && expected_bool then incr true_positives
        else if is_in && not expected_bool then incr false_positives
        else if (not is_in) && expected_bool then incr false_negatives
        else incr true_negatives;
        acc +. abs_float (expected -. result))
      0.0 dataset
  in

  Printf.printf "  %d  |  %d  \n" !true_positives !false_negatives;
  Printf.printf "  %d  |  %d  \n" !false_positives !true_negatives;
  Printf.printf
    "Total dataset size = %d | True answers over the whole dataset = %d | \
     False answers over the whole dataset = %d\n\n"
    (List.length dataset)
    (!true_negatives + !true_positives)
    (!false_negatives + !false_positives);
  1. /. (1. +. error)

let main () =
  Random.self_init ();
  Visualizer.init_window ();

  let number_inputs = 2 in
  let number_outputs = 1 in
  let pop_size = 300 in
  let epochs = 1000 in
  let n_max = ref 2 in
  let n_test = 100 in

  let dynamic_threshold = ref 3. in
  let innov = Innovation.create (number_inputs + number_outputs + 5) in
  let pop =
    ref (Evolution.create_pop pop_size number_inputs number_outputs innov)
  in
  let l_species = ref [] in
  (try
     for epoch = 0 to epochs - 1 do
       let new_pop, new_sp, genomes_evaluated =
         Evolution.generation !pop !l_species (evaluate_genome !n_max) innov
           dynamic_threshold
       in
       pop := new_pop;
       l_species := new_sp;

       let best_genome =
         List.fold_left
           (fun acc g -> if g.fitness > acc.fitness then g else acc)
           (List.hd genomes_evaluated)
           genomes_evaluated
       in

       Visualizer.draw_genome best_genome;
       Graphics.synchronize ();

       Evolution.print_pop_summary !pop !l_species epoch;
       Printf.printf "Curr n_max : %d\n" !n_max;
       flush stdout;

       if best_genome.fitness >= 0.95 && !n_max <= 10 then incr n_max;
       if !n_max > 10 && best_genome.fitness > 0.99 then raise Break
     done
   with
  | Break -> ()
  | exn -> raise exn);

  let best_genome =
    List.fold_left
      (fun acc g -> if g.fitness > acc.fitness then g else acc)
      (List.hd !pop.genomes) !pop.genomes
  in
  ignore (evaluate_genome_test n_test best_genome);
  Printf.printf
    "Evaluated with n_max = %d, the network achieves the following fitness : %f\n"
    n_test
    (evaluate_genome n_test best_genome);

  Printf.printf "Entraînement terminé. Appuyez sur une touche pour quitter.\n";
  flush stdout;

  ignore (Graphics.read_key ())

let () = main ()
