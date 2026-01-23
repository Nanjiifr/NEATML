open Neatml
open Types

type automaton = bool array
type signals = float array array

exception Break

let create_automaton (n : int) : automaton =
  let density = Random.float 1. in
  let whites = int_of_float (float n *. density) in
  let temp = Array.init n (fun i -> i < whites) in
  Array.shuffle ~rand:Random.int temp;
  temp

let create_dataset (n : int) (k : int) : automaton array =
  Array.init k (fun _ -> create_automaton n)

let get_inputs (index : int) (states : automaton) (sigs : signals) (n : int)
    (r : int) =
  let acc = ref [] in
  for delta = r downto -r do
    let neighbor_idx = (((index + delta) mod n) + n) mod n in
    let signal_vals = sigs.(neighbor_idx) in
    let len = Array.length signal_vals in
    for i = len - 1 downto 0 do
      acc := signal_vals.(i) :: !acc
    done;
    let state_val = if states.(neighbor_idx) then 1.0 else 0.0 in
    acc := state_val :: !acc
  done;
  !acc

let get_density (a : automaton) =
  let n = Array.length a in
  let whites = ref 0 in
  Array.iter (fun s -> if s then incr whites) a;
  float !whites /. float n

let evaluator (n : int) (r : int) (n_channels : int) (dataset : automaton array)
    (g : genome) =
  let nets = Array.init n (fun _ -> Phenotype.create_phenotype g) in

  let total_reward = ref 0. in

  (* Pre-allocate buffers for double buffering *)
  let buf_states_1 = Array.make n false in
  let buf_states_2 = Array.make n false in
  let buf_signals_1 = Array.make_matrix n n_channels 0. in
  let buf_signals_2 = Array.make_matrix n n_channels 0. in

  Array.iter
    (fun initial_state ->
      Array.iter Phenotype.reset_network nets;

      (* Initialize first buffer *)
      Array.blit initial_state 0 buf_states_1 0 n;
      (* Reset signal buffer 1 *)
      for i = 0 to n - 1 do
        let row = buf_signals_1.(i) in
        for j = 0 to n_channels - 1 do
          row.(j) <- 0.
        done
      done;

      let current_states = ref buf_states_1 in
      let next_states = ref buf_states_2 in
      let current_signals = ref buf_signals_1 in
      let next_signals = ref buf_signals_2 in

      let density = get_density !current_states in
      let target = if density > 0.5 then 1 else 0 in

      (try
         for _ = 0 to n - 1 do
           let c_states = !current_states in
           let c_sigs = !current_signals in
           let n_states = !next_states in
           let n_sigs = !next_signals in

           for i = 0 to n - 1 do
             let inputs = get_inputs i c_states c_sigs n r in

             match Phenotype.predict nets.(i) inputs with
             | new_state_raw :: new_signal_vals ->
                 n_states.(i) <- new_state_raw > 0.5;

                 if List.length new_signal_vals <> n_channels then
                   failwith
                     (Printf.sprintf
                        "Error: expected output size %d (1 State + %d Signals)"
                        (1 + n_channels) n_channels);

                 let row = n_sigs.(i) in
                 List.iteri (fun idx v -> row.(idx) <- v) new_signal_vals
             | [] -> failwith "Error: Empty output from phenotype"
           done;
           if n_states = c_states then raise Break;

           (* Swap buffers *)
           let tmp_s = !current_states in
           current_states := !next_states;
           next_states := tmp_s;

           let tmp_sig = !current_signals in
           current_signals := !next_signals;
           next_signals := tmp_sig
         done
       with
      | Break -> ()
      | exn -> raise exn);
      let errors = ref 0 in
      Array.iter
        (fun s ->
          let val_s = if s then 1 else 0 in
          if val_s <> target then incr errors)
        !current_states;
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

let visualize_best_genome (n : int) (r : int) (n_channels : int) (g : genome) :
    unit =
  let nets = Array.init n (fun _ -> Phenotype.create_phenotype g) in

  let current_states = ref (create_automaton n) in
  let current_signals = ref (Array.make_matrix n n_channels 0.) in

  let density = get_density !current_states in
  let target = if density > 0.5 then 1 else 0 in
  Printf.printf "The expected color is %s (%d / %d)\n"
    (if target = 1 then "white" else "black")
    (if target = 1 then int_of_float (density *. float n)
     else int_of_float ((1. -. density) *. float n))
    n;
  for _ = 0 to n - 1 do
    visulize_automaton !current_states;
    let prev_states = !current_states in
    let prev_signals = !current_signals in

    let next_states = Array.make n false in
    let next_signals = Array.make_matrix n n_channels 0. in

    for i = 0 to n - 1 do
      let inputs = get_inputs i prev_states prev_signals n r in
      match Phenotype.predict nets.(i) inputs with
      | new_state_raw :: new_signal_vals ->
          next_states.(i) <- new_state_raw > 0.5;
          next_signals.(i) <- Array.of_list new_signal_vals
      | [] -> ()
    done;

    current_states := next_states;
    current_signals := next_signals;

    flush stdout;
    Unix.sleepf 0.05
  done

let _main () =
  Random.self_init ();
  let n = 99 in
  let k = 250 in
  let r = 3 in

  let n_channels = 1 in

  let window_size = (2 * r) + 1 in

  let input_size = window_size * (1 + n_channels) in

  let output_size = 1 + n_channels in

  let pop_size = 150 in
  let epochs = 500 in
  let target_fitness = float k *. 1.95 in
  Printf.printf "Target fitness : %f\n" target_fitness;
  Printf.printf "Channels : %d | Input Size: %d | Output Size: %d\n" n_channels
    input_size output_size;

  let innov = Innovation.create (input_size + output_size + 5) in
  let pop = ref (Evolution.create_pop pop_size input_size output_size innov) in
  let l_species = ref [] in
  let dynamic_threshold = ref 3. in

  (try
     for epoch = 0 to epochs - 1 do
       let dataset = create_dataset n k in
       let new_pop, new_species, genomes_evaluated =
         Evolution.generation !pop !l_species
           (evaluator n r n_channels dataset)
           innov dynamic_threshold
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
  let best_genome =
    List.fold_left
      (fun acc g -> if g.fitness > acc.fitness then g else acc)
      (List.hd !pop.genomes) !pop.genomes
  in

  visualize_best_genome n r n_channels best_genome;
  Visualizer.init_window ();
  Visualizer.draw_genome best_genome;
  ignore (Graphics.read_key ());
  Graphics.close_graph ();

  let filename = "test/density_class_continuous" in
  Parser.save_model best_genome filename

let _test_best () =
  Random.self_init ();
  let n = 149 in
  let r = 3 in
  let k = 10 in

  let n_channels = 1 in

  let filename = "test/density_class_continuous" in
  let genome = Parser.load_model filename in

  let nets = Array.init n (fun _ -> Phenotype.create_phenotype genome) in
  let dataset = create_dataset n k in

  Array.iter
    (fun l ->
      Array.iter Phenotype.reset_network nets;

      let current_states = ref l in
      let current_signals = ref (Array.make_matrix n n_channels 0.) in

      let density = get_density !current_states in
      let target = if density > 0.5 then 1 else 0 in
      Printf.printf "Expected: %s\n" (if target = 1 then "white" else "black");

      (try
         for _ = 0 to 2 * n do
           visulize_automaton !current_states;
           let prev_states = !current_states in
           let prev_signals = !current_signals in

           let next_states = Array.make n false in
           let next_signals = Array.make_matrix n n_channels 0. in

           for i = 0 to n - 1 do
             let inputs = get_inputs i prev_states prev_signals n r in
             match Phenotype.predict nets.(i) inputs with
             | res :: sig_vals ->
                 next_states.(i) <- res > 0.5;
                 next_signals.(i) <- Array.of_list sig_vals
             | [] -> ()
           done;
           if next_states = prev_states then raise Break;

           current_states := next_states;
           current_signals := next_signals;

           flush stdout;
           Unix.sleepf 0.02
         done
       with
      | Break -> ()
      | exn -> raise exn);
      visulize_automaton !current_states;
      Printf.printf "\n\n")
    dataset

let () = _main ()
