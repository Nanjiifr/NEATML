open Neatml
open Types

type automaton = bool array

type rule =
  bool list ->
  bool ->
  bool list ->
  bool (* left neighbors -> center -> right neighbors -> new state*)

exception Break

let create_automaton (n : int) : automaton =
  let density = Random.float 1. in
  let whites = int_of_float (float n *. density) in
  let temp = Array.init n (fun i -> if i < whites then true else false) in
  Array.shuffle ~rand:Random.int temp;
  temp

let create_dataset (n : int) (k : int) : automaton array =
  Array.init k (fun _ -> create_automaton n)

let _apply_rule (a : automaton) (f : rule) (r : int) =
  (* r : rayon de visuel des automates cellulaires *)
  let n = Array.length a in
  Array.mapi
    (fun i s ->
      let left_neighbors =
        List.init r (fun j ->
            let new_index = (((i - j - 1) mod n) + n) mod n in
            a.(new_index))
      in
      let right_neighbors =
        List.init r (fun j ->
            let new_index = (((i + j + 1) mod n) + n) mod n in
            a.(new_index))
      in
      let new_state = f left_neighbors s right_neighbors in
      new_state)
    a

let _get_inputs (index : int) (a : automaton) (r : int) =
  let n = Array.length a in
  let left_neighbors =
    List.init r (fun j ->
        let new_index = (((index - j - 1) mod n) + n) mod n in
        if a.(new_index) then 1. else 0.)
  in
  let right_neighbors =
    List.init r (fun j ->
        let new_index = (((index + j + 1) mod n) + n) mod n in
        if a.(new_index) then 1. else 0.)
  in
  left_neighbors @ [ (if a.(index) then 1. else 0.) ] @ right_neighbors

let int_to_input r val_int =
  let size = (2 * r) + 1 in
  List.init size (fun i ->
      (* On récupère le bit à la position (size - 1 - i) *)
      let shift = size - 1 - i in
      if val_int land (1 lsl shift) <> 0 then 1.0 else 0.0)

let build_lookup_table (nn : network) (r : int) : bool array =
  let size = (2 * r) + 1 in
  let num_configs = 1 lsl size in
  Array.init num_configs (fun i ->
      let inputs = int_to_input r i in
      Phenotype.reset_network nn;
      match Phenotype.predict nn inputs with
      | [ res ] -> res > 0.5
      | _ -> failwith "Error: exepcted output_size to be 1\n")

let get_input_int (index : int) (a : automaton) (n : int) (r : int) =
  let res = ref 0 in
  for delta = -r to r do
    let neighbor_idx = (((index + delta) mod n) + n) mod n in
    res := !res lsl 1;
    if a.(neighbor_idx) then res := !res lor 1
  done;
  !res

let get_density (a : automaton) =
  let n = Array.length a in
  let whites = ref 0 in
  Array.iter (fun s -> if s then incr whites) a;
  float !whites /. float n

let evaluator (n : int) (r : int) (dataset : automaton array) (g : genome) =
  let nn = Phenotype.create_phenotype g in

  let rule_cache = build_lookup_table nn r in

  let total_reward = ref 0. in

  Array.iter
    (fun l ->
      let current_line = ref (Array.copy l) in
      let density = get_density !current_line in
      let target = if density > 0.5 then 1 else 0 in

      (* Simulation temporelle *)
      (try
         for _ = 0 to n - 1 do
           let prev_line = !current_line in
           let next_line =
             Array.init n (fun i ->
                 let config_int = get_input_int i prev_line n r in
                 rule_cache.(config_int))
           in
           if next_line = prev_line then raise Break;
           current_line := next_line
         done
       with
      | Break -> ()
      | exn -> raise exn);

      let errors = ref 0 in
      Array.iter
        (fun s ->
          let val_s = if s then 1 else 0 in
          if val_s <> target then incr errors)
        !current_line;

      let accuracy = float (n - !errors) /. float n in
      total_reward := !total_reward +. accuracy;

      if !errors = 0 then total_reward := !total_reward +. 1.0)
    dataset;

  !total_reward

let visulize_automaton (a : automaton) : unit =
  Printf.printf "|";
  Array.iter
    (fun s -> if s then Printf.printf "\u{25A0}" else Printf.printf " ")
    a;
  Printf.printf "|\n"

let visualize_best_genome (n : int) (r : int) (g : genome) : unit =
  let nn = Phenotype.create_phenotype g in

  let rule_cache = build_lookup_table nn r in

  let current_line = ref (create_automaton n) in
  let density = get_density !current_line in
  let target = if density > 0.5 then 1 else 0 in
  Printf.printf "The expected color is %s (%d / %d)\n"
    (if target = 1 then "white" else "black")
    (if target = 1 then int_of_float (density *. float n)
     else int_of_float ((1. -. density) *. float n))
    n;

  for _ = 0 to n - 1 do
    visulize_automaton !current_line;
    let prev_line = !current_line in
    let next_line =
      Array.init n (fun i ->
          let config_int = get_input_int i prev_line n r in
          rule_cache.(config_int))
    in
    current_line := next_line;
    flush stdout;
    Unix.sleepf 0.02
  done

let _main () =
  Random.self_init ();

  let n = 149 in
  let k = 250 in
  let r = 3 in

  let input_size = (2 * r) + 1 in
  let output_size = 1 in
  let pop_size = 300 in
  let epochs = 1000 in
  let target_fitness = float k *. 1.9 in
  Printf.printf "Target fitness : %f\n" target_fitness;

  let innov = Innovation.create (input_size + output_size + 5) in
  let pop = ref (Evolution.create_pop pop_size input_size output_size innov) in
  let l_species = ref [] in
  let dynamic_threshold = ref 3. in

  let start_time = Unix.gettimeofday () in
  let total_evals = ref 0 in

  (try
     for epoch = 0 to epochs - 1 do
       total_evals := !total_evals + pop_size;
       let dataset = create_dataset n k in
       let new_pop, new_species, genomes_evaluated =
         Evolution.generation !pop !l_species (evaluator n r dataset) innov
           dynamic_threshold ()
       in

       pop := new_pop;
       l_species := new_species;

       Evolution.print_pop_summary !pop !l_species epoch epochs;

       let best_genome =
         List.fold_left
           (fun acc g -> if g.fitness > acc.fitness then g else acc)
           (List.hd genomes_evaluated)
           genomes_evaluated
       in

       if best_genome.fitness >= target_fitness then raise Break
     done
   with
  | Break -> ()
  | exn -> raise exn);

  let duration = Unix.gettimeofday () -. start_time in
  let best_genome =
    List.fold_left
      (fun acc g -> if g.fitness > acc.fitness then g else acc)
      (List.hd !pop.genomes) !pop.genomes
  in
  let avg_fitness =
    let sum = List.fold_left (fun acc g -> acc +. g.fitness) 0. !pop.genomes in
    sum /. float pop_size
  in
  Evolution.print_training_stats !total_evals duration avg_fitness
    best_genome.fitness;

  visualize_best_genome n r best_genome;

  Visualizer.init_window ();
  Visualizer.draw_genome best_genome;
  ignore (Graphics.read_key ());
  Graphics.close_graph ();

  let filename = "test/density_class" in

  Parser.save_model best_genome filename

let _test_best () =
  Random.self_init ();
  let n = 149 in
  let r = 3 in
  let k = 10 in

  let filename = "test/density_class" in
  let genome = Parser.load_model filename in

  let nn = Phenotype.create_phenotype genome in
  let rule_cache = build_lookup_table nn r in

  let dataset = create_dataset n k in

  Array.iter
    (fun l ->
      let current_line = ref l in
      let density = get_density !current_line in
      let target = if density > 0.5 then 1 else 0 in
      Printf.printf "The expected color is %s (%d / %d)\n"
        (if target = 1 then "white" else "black")
        (if target = 1 then int_of_float (density *. float n)
         else int_of_float ((1. -. density) *. float n))
        n;
      (try
         (* Simulation temporelle *)
         for _ = 0 to 2 * n do
           visulize_automaton !current_line;
           let prev_line = !current_line in
           let next_line =
             Array.init n (fun i ->
                 let config_int = get_input_int i prev_line n r in
                 rule_cache.(config_int))
           in
           if next_line = prev_line then raise Break;
           current_line := next_line;
           flush stdout;
           Unix.sleepf 0.02
         done
       with
      | Break -> ()
      | exn -> raise exn);
      visulize_automaton !current_line;
      Printf.printf "\n\n")
    dataset

let () = _test_best ()
