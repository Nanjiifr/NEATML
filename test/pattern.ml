module P = Neatml.Phenotype
module T = Neatml.Types
module E = Neatml.Evolution
module G = Neatml.Genome

(* --- Configuration --- *)
let grid_width = 16
let grid_height = 16
let radius = 5.0
let max_steps = 25
let pop_size = 300
let inputs_per_channel = 4 (* Self, GradientX, GradientY, AvgNeighbors *)
let num_channels = 3
let input_size = num_channels * inputs_per_channel
let output_size = num_channels (* Next state for each channel *)

(* --- Simulation --- *)

(* Target Circle Mask *)
let target_grid =
  Array.init grid_width (fun x ->
      Array.init grid_height (fun y ->
          let cx, cy = (7.5, 7.5) in
          let dx = float x -. cx in
          let dy = float y -. cy in
          let dist = sqrt ((dx *. dx) +. (dy *. dy)) in
          if dist <= radius then 1.0 else 0.0))

(* Reusable buffers for the evaluator (Thread-local if parallel, but Parmap uses processes usually or we allocate inside) *)
(* Since Parmap might use fork, we can allocate inside the function to be safe. 
   Allocating 2 tiny 16x16 arrays is cheap. *)

let get_val grid x y ch =
  if x < 0 || x >= grid_width || y < 0 || y >= grid_height then 0.0
  else grid.(x).(y).(ch)

let calculate_iou grid target =
  let intersection = ref 0. in
  let union = ref 0. in
  for x = 0 to grid_width - 1 do
    for y = 0 to grid_height - 1 do
      let val_out = grid.(x).(y).(0) in
      let val_target = target.(x).(y) in
      let is_on_out = val_out > 0.5 in
      let is_on_target = val_target > 0.5 in
      if is_on_out && is_on_target then intersection := !intersection +. 1.;
      if is_on_out || is_on_target then union := !union +. 1.
    done
  done;
  if !union = 0. then 0.0 else (!intersection /. !union) *. 100.0

let evaluator genome =
  let phenotype = P.create_phenotype genome in

  (* Grid state: [x][y][channel] *)
  let grid_a =
    Array.init grid_width (fun _ ->
        Array.init grid_height (fun _ -> Array.make num_channels 0.0))
  in
  let grid_b =
    Array.init grid_width (fun _ ->
        Array.init grid_height (fun _ -> Array.make num_channels 0.0))
  in

  (* Initialize Center *)
  grid_a.(8).(8).(0) <- 1.0;

  let current = ref grid_a in
  let next = ref grid_b in

  let inputs = Array.make input_size 0.0 in
  let total_fitness = ref 0. in
  let steps_counted = ref 0 in

  for step = 1 to max_steps do
    let g = !current in
    let n = !next in

    for x = 0 to grid_width - 1 do
      for y = 0 to grid_height - 1 do
        (* Prepare Inputs *)
        for ch = 0 to num_channels - 1 do
          let base = ch * inputs_per_channel in
          let self_val = g.(x).(y).(ch) in
          let v_right = get_val g (x + 1) y ch in
          let v_left = get_val g (x - 1) y ch in
          let v_down = get_val g x (y + 1) ch in
          let v_up = get_val g x (y - 1) ch in

          (* Gradient X: Right - Left *)
          let gx = v_right -. v_left in
          (* Gradient Y: Down - Up *)
          let gy = v_down -. v_up in
          (* Average Neighbors (Von Neumann) *)
          let avg = (v_right +. v_left +. v_down +. v_up) /. 4.0 in

          inputs.(base) <- self_val;
          inputs.(base + 1) <- gx;
          inputs.(base + 2) <- gy;
          inputs.(base + 3) <- avg
        done;

        (* Run Network *)
        let outputs = P.predict_array phenotype inputs in

        (* Apply Outputs *)
        n.(x).(y).(0) <- (if outputs.(0) > 0.5 then 1.0 else 0.0);
        n.(x).(y).(1) <- outputs.(1);
        n.(x).(y).(2) <- outputs.(2)
      done
    done;

    (* Stability Check: Accumulate fitness over the last few steps *)
    if step >= max_steps - 5 then (
      total_fitness := !total_fitness +. calculate_iou !next target_grid;
      incr steps_counted
    );

    (* Swap buffers *)
    let temp = !current in
    current := !next;
    next := temp
  done;

  if !steps_counted > 0 then !total_fitness /. float !steps_counted
  else 0.0

(* --- Visualization (Optional helper) --- *)
let print_grid genome =
  let phenotype = P.create_phenotype genome in
  let grid =
    Array.init grid_width (fun _ ->
        Array.init grid_height (fun _ -> Array.make num_channels 0.0))
  in
  let next_grid =
    Array.init grid_width (fun _ ->
        Array.init grid_height (fun _ -> Array.make num_channels 0.0))
  in
  grid.(8).(8).(0) <- 1.0;

  let current = ref grid in
  let next = ref next_grid in
  let inputs = Array.make input_size 0.0 in

  Printf.printf "\n--- Simulation Replay ---\n";
  for step = 1 to max_steps do
    let g = !current in
    let n = !next in
    Printf.printf "Step %d:\n" step;
    for y = 0 to grid_height - 1 do
      for x = 0 to grid_width - 1 do
        if g.(x).(y).(0) > 0.5 then print_string "O " else print_string ". "
      done;
      print_newline ()
    done;

    for x = 0 to grid_width - 1 do
      for y = 0 to grid_height - 1 do
        for ch = 0 to num_channels - 1 do
          let base = ch * inputs_per_channel in
          let self_val = g.(x).(y).(ch) in
          let v_right = get_val g (x + 1) y ch in
          let v_left = get_val g (x - 1) y ch in
          let v_down = get_val g x (y + 1) ch in
          let v_up = get_val g x (y - 1) ch in

          let gx = v_right -. v_left in
          let gy = v_down -. v_up in
          let avg = (v_right +. v_left +. v_down +. v_up) /. 4.0 in

          inputs.(base) <- self_val;
          inputs.(base + 1) <- gx;
          inputs.(base + 2) <- gy;
          inputs.(base + 3) <- avg
        done;
        let outputs = P.predict_array phenotype inputs in
        n.(x).(y).(0) <- (if outputs.(0) > 0.5 then 1.0 else 0.0);
        n.(x).(y).(1) <- outputs.(1);
        n.(x).(y).(2) <- outputs.(2)
      done
    done;
    let temp = !current in
    current := !next;
    next := temp;
    Unix.sleepf 0.1
  done

(* --- Main --- *)
let () =
  Random.self_init ();
  let epochs = 5000 in
  let innov_global = Neatml.Innovation.create (input_size + output_size + 5) in
  (* inputs: 9, outputs: 3 *)
  let pop = E.create_pop pop_size input_size output_size innov_global in
  let dynamic_threshold = ref 3.0 in
  let species = ref (E.speciate_population pop.genomes [] !dynamic_threshold) in
  let start_time = Unix.gettimeofday () in

  let rec run_epochs current_pop current_species epoch =
    let next_pop, next_species, evaluated =
      E.generation current_pop current_species evaluator innov_global
        dynamic_threshold
    in
    let best =
      List.fold_left
        (fun acc g -> if g.T.fitness > acc.T.fitness then g else acc)
        (List.hd evaluated) evaluated
    in

    E.print_pop_summary next_pop next_species epoch epochs;

    if best.T.fitness > 95.0 || epoch >= epochs then (
      (* IoU > 95% is excellent *)
      print_newline ();
      print_grid best;
      let avg_fit =
        List.fold_left (fun acc g -> acc +. g.T.fitness) 0. evaluated
        /. float (List.length evaluated)
      in
      E.print_training_stats (epoch * pop_size)
        (Unix.gettimeofday () -. start_time)
        avg_fit best.T.fitness;
      Printf.printf "Done.\n")
    else run_epochs next_pop next_species (epoch + 1)
  in

  run_epochs pop !species 1
