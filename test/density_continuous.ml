open Neatml
open Types

type cellular = { mutable color : float; channels : float array }
type automaton = cellular array

exception Break

let nbChannels = 1
let automaton_size = 99
let window_width = 1200
let window_height = 1000

let get_heatmap_color value =
  (* 1. On s'assure que la valeur reste entre 0.0 et 1.0 *)
  let v = max 0.0 (min 1.0 value) in

  (* 2. Calcul des composantes R, G, B *)
  let r, g, b =
    if v < 0.5 then
      (* Segment 1 : Bleu (0.0) -> Rouge (0.5) *)
      let ratio = v /. 0.5 in
      (* Normalise de 0..0.5 vers 0..1 *)
      ( int_of_float (255. *. ratio),
        (* R augmente *)
        0,
        (* G reste nul *)
        int_of_float (255. *. (1. -. ratio))
        (* B diminue *) )
    else
      (* Segment 2 : Rouge (0.5) -> Jaune (1.0) *)
      let ratio = (v -. 0.5) /. 0.5 in
      (* Normalise de 0.5..1 vers 0..1 *)
      ( 255,
        (* R reste au max *)
        int_of_float (255. *. ratio),
        (* G augmente *)
        0
        (* B reste nul *) )
  in

  (* 3. Retourne la couleur au format Graphics *)
  Graphics.rgb r g b

(* --- Automaton & Data Helpers --- *)
let create_random () : automaton =
  Array.init automaton_size (fun _ ->
      { color = Random.float 1.; channels = Array.make nbChannels 0. })

let create_dataset (n : int) : automaton array =
  Array.init n (fun _ -> create_random ())

let copy_automaton (src : automaton) : automaton =
  Array.map (fun c -> { color = c.color; channels = Array.copy c.channels }) src

let get_density (a : automaton) =
  let sum = Array.fold_left (fun acc c -> acc +. c.color) 0. a in
  sum /. float automaton_size

(* --- Network Input Logic --- *)
let get_inputs (index : int) (states : automaton) (n : int) (r : int) =
  let acc = ref [] in
  (* Gather inputs from neighbors within radius r *)
  for delta = r downto -r do
    let neighbor_idx = (((index + delta) mod n) + n) mod n in
    let cell = states.(neighbor_idx) in
    (* Inputs: Channels then Color *)
    for i = nbChannels - 1 downto 0 do
      acc := cell.channels.(i) :: !acc
    done;
    acc := cell.color :: !acc
  done;
  !acc

(* --- Evaluation Logic --- *)
let evaluator (r : int) (dataset : automaton array) (g : genome) =
  let nets =
    Array.init automaton_size (fun _ -> Phenotype.create_phenotype g)
  in
  let total_reward = ref 0. in
  let n = automaton_size in

  Array.iter
    (fun initial_state ->
      Array.iter Phenotype.reset_network nets;

      let current_states = ref (copy_automaton initial_state) in
      let next_states = ref (create_random ()) in
      (* buffer *)

      let density = get_density !current_states in
      let target = if density > 0.5 then 1.0 else 0.0 in
      let stable = ref false in

      (try
         for _ = 0 to n (* Run for n steps *) do
           let c_st = !current_states in
           let n_st = !next_states in

           let changed = ref false in

           for i = 0 to n - 1 do
             let inputs = get_inputs i c_st n r in
             match Phenotype.predict nets.(i) inputs with
             | raw_output :: new_signals ->
                 (* ResNet-like update: output is a delta *)
                 (* Map sigmoid [0,1] to [-1,1] then scale by 0.1 *)
                 let delta = (raw_output -. 0.5) *. 2.0 in
                 let new_val = c_st.(i).color +. (delta *. 0.1) in
                 let safe_color = max 0.0 (min 1.0 new_val) in

                 n_st.(i).color <- safe_color;

                 (* Optimization: Check stability *)
                 if abs_float (c_st.(i).color -. safe_color) > 0.001 then
                   changed := true;

                 if List.length new_signals <> nbChannels then
                   failwith "Error: Output size mismatch";

                 List.iteri
                   (fun idx v -> n_st.(i).channels.(idx) <- v)
                   new_signals
             | [] -> failwith "Empty output"
           done;

           if not !changed then (
             stable := true;
             raise Break);

           (* Swap *)
           let tmp = !current_states in
           current_states := !next_states;
           next_states := tmp
         done
       with Break -> ());

      if not !stable then
        (* Harsh penalty for instability/oscillation *)
        total_reward := !total_reward +. 0.0
      else
        (* Calculate Error (MSE) and Smoothness Penalty *)
        let error_sum = ref 0. in
        let smoothness_sum = ref 0. in

        for i = 0 to n - 1 do
          let c = !current_states.(i) in
          let diff = target -. c.color in
          error_sum := !error_sum +. (diff *. diff);

          let next_i = (i + 1) mod n in
          let neighbor = !current_states.(next_i) in
          smoothness_sum :=
            !smoothness_sum +. abs_float (c.color -. neighbor.color)
        done;

        let mse = !error_sum /. float n in
        let avg_smoothness = !smoothness_sum /. float n in

        (* Fitness is 1.0 - MSE - Smoothness, clamped at 0 *)
        (* We penalize high frequency changes (0,1,0,1 patterns) *)
        let score = max 0.0 (1.0 -. mse -. (2.5 *. avg_smoothness)) in
        total_reward := !total_reward +. score)
    dataset;

  !total_reward

(* --- Visualization --- *)
let draw_line (y : int) (a : automaton) =
  let cell_width = window_width / automaton_size in
  Array.iteri
    (fun i c ->
      Graphics.set_color (get_heatmap_color c.color);
      Graphics.fill_rect (i * cell_width) y cell_width 10)
    a

let visualize_best (r : int) (g : genome) (wait : bool) =
  let n = automaton_size in
  let nets = Array.init n (fun _ -> Phenotype.create_phenotype g) in
  let initial = create_random () in
  let current_states = ref initial in
  let next_states = ref (create_random ()) in

  let density = get_density !current_states in
  if wait then
    Printf.printf "Initial Density: %.2f -> Target: %.1f\n" density
      (if density > 0.5 then 1.0 else 0.0);
  flush stdout;

  (* Setup Graphics *)
  (try Graphics.open_graph (Printf.sprintf " %dx%d" window_width window_height)
   with _ -> ());
  Graphics.clear_graph ();
  Graphics.set_color Graphics.black;
  Graphics.fill_rect 0 0 window_width window_height;

  let y = ref (window_height - 10) in

  try
    while !y >= 0 do
      draw_line !y !current_states;

      let c_st = !current_states in
      let n_st = !next_states in

      for i = 0 to n - 1 do
        let inputs = get_inputs i c_st n r in
        match Phenotype.predict nets.(i) inputs with
        | raw_output :: new_signals ->
            let delta = (raw_output -. 0.5) *. 2.0 in
            let new_val = c_st.(i).color +. (delta *. 0.1) in
            n_st.(i).color <- max 0.0 (min 1.0 new_val);
            List.iteri (fun idx v -> n_st.(i).channels.(idx) <- v) new_signals
        | [] -> ()
      done;

      (* Swap *)
      let tmp = !current_states in
      current_states := !next_states;
      next_states := tmp;

      y := !y - 10
      (* Unix.sleepf 0.005 *)
      (* Remove sleep for fast rendering of the map *)
    done;
    if wait then (
      Printf.printf "Visualization complete. Press key to exit.\n";
      flush stdout;
      ignore (Graphics.read_key ());
      Graphics.close_graph ())
  with _ -> Graphics.close_graph ()

let test_network filename r =
  let g = Parser.load_model filename in
  let n = automaton_size in
  let nets = Array.init n (fun _ -> Phenotype.create_phenotype g) in

  (try Graphics.open_graph (Printf.sprintf " %dx%d" window_width window_height)
   with _ -> ());

  let rec loop () =
    let target_density = Random.float 1.0 in
    let initial =
      Array.init n (fun _ ->
          let color = if Random.float 1.0 < target_density then 1.0 else 0.0 in
          { color; channels = Array.make nbChannels 0. })
    in

    let current_states = ref initial in
    let next_states = ref (create_random ()) in

    Printf.printf "Testing Density: %.2f\n" target_density;
    flush stdout;

    Graphics.clear_graph ();
    Graphics.set_color Graphics.black;
    Graphics.fill_rect 0 0 window_width window_height;

    let y = ref (window_height - 10) in

    try
      while !y >= 0 do
        draw_line !y !current_states;

        let c_st = !current_states in
        let n_st = !next_states in

        for i = 0 to n - 1 do
          let inputs = get_inputs i c_st n r in
          match Phenotype.predict nets.(i) inputs with
          | raw_output :: new_signals ->
              let delta = (raw_output -. 0.5) *. 2.0 in
              let new_val = c_st.(i).color +. (delta *. 0.1) in
              n_st.(i).color <- max 0.0 (min 1.0 new_val);
              List.iteri (fun idx v -> n_st.(i).channels.(idx) <- v) new_signals
          | [] -> ()
        done;

        let tmp = !current_states in
        current_states := !next_states;
        next_states := tmp;

        y := !y - 10
      done;

      Printf.printf "Press Space for new, q to quit.\n";
      flush stdout;
      let status = Graphics.read_key () in
      if status = ' ' then loop () else if status = 'q' then () else loop ()
    with _ -> ()
  in
  loop ();
  Graphics.close_graph ()

let _main () =
  Random.self_init ();
  let r = 3 in

  if Array.length Sys.argv > 1 then test_network Sys.argv.(1) r
  else begin
    let k = 50 in
    (* Dataset size *)

    let window_size = (2 * r) + 1 in
    let input_size = window_size * (1 + nbChannels) in
    let output_size = 1 + nbChannels in

    let pop_size = 150 in
    let epochs = 1000 in
    let target_fitness = float k *. 0.98 in
    (* 98% accuracy roughly *)

    Printf.printf "Starting Evolution...\n";
    Printf.printf "Inputs: %d | Outputs: %d\n" input_size output_size;
    Printf.printf "Target fitness: %f\n" target_fitness;

    let innov = Innovation.create (input_size + output_size + 5) in
    let pop =
      ref (Evolution.create_pop pop_size input_size output_size innov)
    in
    let l_species = ref [] in
    let dynamic_threshold = ref 3. in
    let dataset = create_dataset k in

    try
      for epoch = 0 to epochs - 1 do
        let new_pop, new_species, evaluated =
          Evolution.generation !pop !l_species (evaluator r dataset) innov
            dynamic_threshold ()
        in
        pop := new_pop;
        l_species := new_species;

        Evolution.print_pop_summary !pop !l_species epoch epochs;
        let best =
          List.fold_left
            (fun acc g -> if g.fitness > acc.fitness then g else acc)
            (List.hd evaluated) evaluated
        in

        if epoch mod 5 = 0 then visualize_best r best false;

        (* sorted in generation *)
        if best.fitness >= target_fitness then raise Break
      done
    with Break ->
      Printf.printf "Target fitness reached!\n";

      let best_genome =
        List.fold_left
          (fun acc g -> if g.fitness > acc.fitness then g else acc)
          (List.hd !pop.genomes) !pop.genomes
      in

      Printf.printf "Best Fitness: %f\n" best_genome.fitness;
      visualize_best r best_genome true;

      let filename = "test/density_continuous.neat" in
      Parser.save_model best_genome filename;
      (* Uncomment if save needed *)
      Printf.printf "Done.\n"
  end

let () = _main ()
