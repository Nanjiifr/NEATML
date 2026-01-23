open Neatml
open Types

(* --- Configurations --- *)
let grid_width = 20
let grid_height = 20
let pixel_size = 20
let max_moves_without_food = 100 (* More leeway for long snakes *)

(* --- Types --- *)
type direction = North | East | South | West
type point = { x : int; y : int }

type gamestate = {
  snake : point list;
  food : point;
  dir : direction;
  score : int;
  moves_left : int;
  alive : bool;
}

(* --- Helpers --- *)
let get_random_point () =
  { x = Random.int grid_width; y = Random.int grid_height }

let is_opposite d1 d2 =
  match (d1, d2) with
  | North, South | South, North | East, West | West, East -> true
  | _ -> false

(* Initialize game *)
let init_game () =
  let center_x = grid_width / 2 in
  let center_y = grid_height / 2 in
  let start_snake =
    [
      { x = center_x; y = center_y };
      { x = center_x; y = center_y + 1 };
      { x = center_x; y = center_y + 2 };
    ]
  in
  let rec safe_food () =
    let f = get_random_point () in
    if List.exists (fun b -> b.x = f.x && b.y = f.y) start_snake then
      safe_food ()
    else f
  in
  {
    snake = start_snake;
    food = safe_food ();
    dir = North;
    score = 0;
    moves_left = max_moves_without_food;
    alive = true;
  }

(* --- Physics --- *)
let move_point p d =
  match d with
  | North -> { x = p.x; y = p.y - 1 }
  | South -> { x = p.x; y = p.y + 1 }
  | East -> { x = p.x + 1; y = p.y }
  | West -> { x = p.x - 1; y = p.y }

let check_collision p body =
  if p.x < 0 || p.x >= grid_width || p.y < 0 || p.y >= grid_height then true
  else List.exists (fun b -> b.x = p.x && b.y = p.y) body

let rec spawn_food snake =
  let f = get_random_point () in
  if List.exists (fun b -> b.x = f.x && b.y = f.y) snake then spawn_food snake
  else f

let update_state state action_idx =
  if not state.alive then state
  else
    (* Outputs: 0:North, 1:East, 2:South, 3:West *)
    let requested_dir =
      match action_idx with
      | 0 -> North
      | 1 -> East
      | 2 -> South
      | 3 -> West
      | _ -> state.dir
    in

    let new_dir =
      if is_opposite requested_dir state.dir then state.dir else requested_dir
    in

    let head = List.hd state.snake in
    let new_head = move_point head new_dir in

    if check_collision new_head state.snake then { state with alive = false }
    else if new_head = state.food then
      {
        state with
        snake = new_head :: state.snake;
        food = spawn_food (new_head :: state.snake);
        score = state.score + 1;
        moves_left = max_moves_without_food + (state.score * 2);
        (* Bonus time for long snakes *)
        dir = new_dir;
      }
    else
      (* Move tail *)
      let rec remove_last = function
        | [] -> []
        | [ _ ] -> []
        | x :: xs -> x :: remove_last xs
      in
      let new_body = new_head :: remove_last state.snake in
      let new_moves = state.moves_left - 1 in
      if new_moves <= 0 then { state with alive = false }
      else
        { state with snake = new_body; dir = new_dir; moves_left = new_moves }

(* --- Sensors --- *)
(* 8 Directions *)
let sensor_directions =
  [
    (0, -1);
    (* N *)
    (1, -1);
    (* NE *)
    (1, 0);
    (* E *)
    (1, 1);
    (* SE *)
    (0, 1);
    (* S *)
    (-1, 1);
    (* SW *)
    (-1, 0);
    (* W *)
    (-1, -1) (* NW *);
  ]

(* Returns [inv_dist_wall; inv_dist_body] *)
let look state (dx, dy) =
  let head = List.hd state.snake in

  let cx = ref head.x in
  let cy = ref head.y in
  let distance = ref 0. in

  let found_wall = ref false in
  let found_body = ref false in
  let dist_body = ref 0. in

  let break = ref false in

  while not !break do
    cx := !cx + dx;
    cy := !cy + dy;
    distance := !distance +. 1.;

    (* Check Wall *)
    if !cx < 0 || !cx >= grid_width || !cy < 0 || !cy >= grid_height then (
      found_wall := true;
      break := true (* Check Body *))
    else if
      (not !found_body)
      && List.exists (fun b -> b.x = !cx && b.y = !cy) state.snake
    then (
      found_body := true;
      dist_body := !distance
      (* Don't break here, we want to find the wall too if it's behind body? 
         Actually standard look stops at first obstacle. 
         But here we want to separate inputs. 
         Let's keep going until wall to get both info if possible?
         No, simplistic view: First thing hit. *)

      (* Wait, to give good info, better to separate completely. 
          Let's assume Body is transparent for the Wall sensor? 
          Yes, that gives more info. *))
  done;

  let val_wall = 1.0 /. !distance in
  let val_body = if !found_body then 1.0 /. !dist_body else 0.0 in

  [ val_wall; val_body ]

let get_food_inputs state =
  let head = List.hd state.snake in
  let fx, fy = (state.food.x, state.food.y) in
  let hx, hy = (head.x, head.y) in
  [
    (if fy < hy then 1.0 else 0.0);
    (* Food North *)
    (if fx > hx then 1.0 else 0.0);
    (* Food East *)
    (if fy > hy then 1.0 else 0.0);
    (* Food South *)
    (if fx < hx then 1.0 else 0.0);
    (* Food West *)
  ]

let get_tail_inputs state =
  let head = List.hd state.snake in
  let tail = List.hd (List.rev state.snake) in
  let tx, ty = (tail.x, tail.y) in
  let hx, hy = (head.x, head.y) in
  [
    (if ty < hy then 1.0 else 0.0);
    (* Tail North *)
    (if tx > hx then 1.0 else 0.0);
    (* Tail East *)
    (if ty > hy then 1.0 else 0.0);
    (* Tail South *)
    (if tx < hx then 1.0 else 0.0);
    (* Tail West *)
  ]

let get_dir_inputs state =
  match state.dir with
  | North -> [ 1.; 0.; 0.; 0. ]
  | East -> [ 0.; 1.; 0.; 0. ]
  | South -> [ 0.; 0.; 1.; 0. ]
  | West -> [ 0.; 0.; 0.; 1. ]

(* --- Flood Fill --- *)
let get_accessibility_inputs state =
  let head = List.hd state.snake in
  let total_cells = grid_width * grid_height in
  let occupied = Array.make_matrix grid_width grid_height false in
  List.iter (fun p -> occupied.(p.x).(p.y) <- true) state.snake;

  (* Helper for BFS *)
  let count_accessible start_p =
    if
      start_p.x < 0 || start_p.x >= grid_width || start_p.y < 0
      || start_p.y >= grid_height
    then 0.
    else if occupied.(start_p.x).(start_p.y) then 0.
    else
      let q = Queue.create () in
      Queue.add start_p q;
      let visited = Array.make_matrix grid_width grid_height false in
      visited.(start_p.x).(start_p.y) <- true;

      (* Mark initial blocked for this search so we don't count snake body *)
      (* Actually we use a local visited combined with global occupied *)
      let count = ref 0 in
      while not (Queue.is_empty q) do
        let curr = Queue.take q in
        incr count;

        let neighbors =
          [
            { x = curr.x; y = curr.y - 1 };
            { x = curr.x; y = curr.y + 1 };
            { x = curr.x - 1; y = curr.y };
            { x = curr.x + 1; y = curr.y };
          ]
        in

        List.iter
          (fun n ->
            if n.x >= 0 && n.x < grid_width && n.y >= 0 && n.y < grid_height
            then
              if (not occupied.(n.x).(n.y)) && not visited.(n.x).(n.y) then (
                visited.(n.x).(n.y) <- true;
                Queue.add n q))
          neighbors
      done;
      float_of_int !count /. float_of_int (total_cells - List.length state.snake)
  in

  let dirs =
    [
      { x = head.x; y = head.y - 1 };
      (* North *)
      { x = head.x + 1; y = head.y };
      (* East *)
      { x = head.x; y = head.y + 1 };
      (* South *)
      { x = head.x - 1; y = head.y } (* West *);
    ]
  in

  List.map count_accessible dirs

let get_inputs state =
  let obstacles = List.flatten (List.map (look state) sensor_directions) in
  let food = get_food_inputs state in
  let tail = get_tail_inputs state in
  let dir = get_dir_inputs state in
  let access = get_accessibility_inputs state in
  obstacles @ food @ tail @ dir @ access

(* --- View --- *)
module View = struct
  open Graphics

  let open_window () =
    open_graph
      (Printf.sprintf " %dx%d" (grid_width * pixel_size)
         (grid_height * pixel_size));
    set_window_title "Snake NEAT";
    auto_synchronize false

  let draw state gen =
    clear_graph ();
    set_color (rgb 30 30 30);
    fill_rect 0 0 (size_x ()) (size_y ());
    set_color red;
    fill_circle
      ((state.food.x * pixel_size) + 10)
      (size_y () - ((state.food.y * pixel_size) + 10))
      8;
    List.iteri
      (fun i p ->
        if i = 0 then set_color (rgb 0 255 0) else set_color (rgb 0 180 0);
        fill_rect (p.x * pixel_size)
          (size_y () - ((p.y + 1) * pixel_size))
          pixel_size pixel_size;
        set_color black;
        draw_rect (p.x * pixel_size)
          (size_y () - ((p.y + 1) * pixel_size))
          pixel_size pixel_size)
      state.snake;
    set_color white;
    moveto 10 10;
    draw_string
      (Printf.sprintf "Gen: %d | Score: %d | Moves: %d" gen state.score
         state.moves_left);
    synchronize ()
end

(* --- Evaluator --- *)
let manhattan_dist p1 p2 = abs (p1.x - p2.x) + abs (p1.y - p2.y)

let run_simulation seed g =
  Random.init seed;
  let phenotype = Phenotype.create_phenotype g in
  let state = ref (init_game ()) in
  let fitness = ref 0. in
  let duration = ref 0 in

  while !state.alive do
    let inputs = get_inputs !state in
    let outputs = Phenotype.predict phenotype inputs in

    let action =
      let max_val = ref neg_infinity in
      let max_idx = ref 0 in
      List.iteri
        (fun i v ->
          if v > !max_val then (
            max_val := v;
            max_idx := i))
        outputs;
      !max_idx
    in

    let head_before = List.hd !state.snake in
    let dist_before = manhattan_dist head_before !state.food in
    let score_before = !state.score in

    state := update_state !state action;
    incr duration;

    if !state.alive then begin
      (* Standard survival reward *)
      fitness := !fitness +. 0.01;

      let head_after = List.hd !state.snake in
      let dist_after = manhattan_dist head_after !state.food in

      (* Reward Structure *)
      if !state.score > score_before then fitness := !fitness +. 50.0
      else if
        (* Reduced greedy incentive *)
        dist_after < dist_before
      then fitness := !fitness +. 0.1
      else fitness := !fitness -. 0.15
    end
  done;
  !fitness

exception Break

let main () =
  Random.self_init ();
  (* Inputs: 8 Dirs * 2 + 4 Food + 4 Tail + 4 Dir + 4 Access = 32 Inputs *)
  let input_size = 32 in
  let output_size = 4 in

  let pop_size = 300 in
  let epochs = 1000 in
  let target_fitness = 10000. in
  let maps_per_genome = 5 in

  let innov = Innovation.create (input_size + output_size + 1) in
  let pop = ref (Evolution.create_pop pop_size input_size output_size innov) in
  let l_species = ref [] in
  let dynamic_threshold = ref 3. in

  let start_time = Unix.gettimeofday () in
  let total_evals = ref 0 in
  let window_open = ref false in

  (try
     for epoch = 0 to epochs - 1 do
       total_evals := !total_evals + pop_size;
       let map_seeds = List.init maps_per_genome (fun _ -> Random.bits ()) in

       let multi_map_evaluator g =
         let total =
           List.fold_left
             (fun acc seed -> acc +. run_simulation seed g)
             0. map_seeds
         in
         total /. float maps_per_genome
       in

       let new_pop, new_sp, genomes_evaluated =
         Evolution.generation !pop !l_species multi_map_evaluator innov
           dynamic_threshold
       in
       pop := new_pop;
       l_species := new_sp;

       Evolution.print_pop_summary !pop !l_species epoch epochs;

       let best_genome =
         List.fold_left
           (fun acc g -> if g.fitness > acc.fitness then g else acc)
           (List.hd genomes_evaluated)
           genomes_evaluated
       in

       if best_genome.fitness > 2500. && epoch mod 100 = 0 then (
         if not !window_open then (
           View.open_window ();
           window_open := true);
         Random.self_init ();
         let phenotype = Phenotype.create_phenotype best_genome in
         let state = ref (init_game ()) in
         while !state.alive do
           View.draw !state epoch;
           let inputs = get_inputs !state in
           let outputs = Phenotype.predict phenotype inputs in
           let action =
             let max_val = ref neg_infinity in
             let max_idx = ref 0 in
             List.iteri
               (fun i v ->
                 if v > !max_val then (
                   max_val := v;
                   max_idx := i))
               outputs;
             !max_idx
           in
           state := update_state !state action;
           Unix.sleepf 0.01
         done);

       if best_genome.fitness > target_fitness then raise Break
     done
   with
  | Break -> ()
  | exn -> raise exn);

  if !window_open then Graphics.close_graph ();
  let duration = Unix.gettimeofday () -. start_time in
  let best_genome =
    List.fold_left
      (fun acc g -> if g.fitness > acc.fitness then g else acc)
      (List.hd !pop.genomes) !pop.genomes
  in
  let avg_fitness =
    List.fold_left (fun acc g -> acc +. g.fitness) 0. !pop.genomes
    /. float pop_size
  in
  Evolution.print_training_stats !total_evals duration avg_fitness
    best_genome.fitness;
  Parser.save_model best_genome "snake_net";
  Printf.printf "Model saved.\n"

let () = main ()
