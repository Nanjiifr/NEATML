open Neatml
open Types

type config = {
  gravity : float;
  jump_strength : float;
  speed_x : float;
  pipe_gap : float;
  pipe_interval : float;
  bird_radius : float;
  pipe_width : float;
  window_width : float;
  window_height : float;
  tau : float;
}

type bird = { y : float; velocity : float; alive : bool }
type pipe = { x : float; gap_y : float; passed : bool }

type game_state = {
  player : bird;
  pipes : pipe list;
  score : int;
  time : int;
  config : config;
}

module View = struct
  open Graphics

  let open_window width height =
    open_graph
      (Printf.sprintf " %dx%d" (int_of_float width) (int_of_float height));
    set_window_title "Flappy Bird NEAT";
    auto_synchronize false

  let clear () =
    set_color (rgb 135 206 235);
    fill_rect 0 0 (size_x ()) (size_y ())

  let draw_bird bird radius =
    set_color (rgb 255 215 0);
    (* L'oiseau est centré en X (width / 2) *)
    let center_x = size_x () / 2 in
    fill_circle center_x (int_of_float bird.y) (int_of_float radius);
    set_color black;
    draw_circle center_x (int_of_float bird.y) (int_of_float radius)

  let draw_pipes pipes width window_height pipe_gap =
    set_color (rgb 34 139 34);

    let half_width = width *. 0.5 in
    let half_gap = pipe_gap *. 0.5 in

    List.iter
      (fun p ->
        let left_x = int_of_float (p.x -. half_width) in
        let w = int_of_float width in

        (* Limites du trou *)
        let top_edge = p.gap_y +. half_gap in
        let bot_edge = p.gap_y -. half_gap in

        (* TUYAU DU BAS *)
        let bot_h = int_of_float bot_edge in
        if bot_h > 0 then (
          fill_rect left_x 0 w bot_h;
          draw_rect left_x 0 w bot_h);

        (* TUYAU DU HAUT *)
        let top_y_int = int_of_float top_edge in
        let top_h = int_of_float (window_height -. top_edge) in
        if top_h > 0 then (
          fill_rect left_x top_y_int w top_h;
          draw_rect left_x top_y_int w top_h))
      pipes

  let draw_score score generation =
    set_color black;
    moveto 10 10;
    draw_string (Printf.sprintf "Gen: %d | Score: %d" generation score)

  let _render state generation =
    clear ();
    draw_pipes state.pipes state.config.pipe_width state.config.window_height
      state.config.pipe_gap;
    draw_bird state.player state.config.bird_radius;
    draw_score state.score generation;
    synchronize ()
end

exception Break

let init_game conf =
  let player = { y = conf.window_height /. 2.; velocity = 0.; alive = true } in
  let pipes = [] in
  let score = 0 in
  let time = 0 in
  let config = conf in
  { player; pipes; score; time; config }

let rec update_bird conf player is_jump =
  match is_jump with
  | true ->
      let velocity = conf.jump_strength in
      update_bird conf { player with velocity } false
  | false ->
      let velocity = player.velocity -. (conf.tau *. conf.gravity) in
      let y = player.y +. (conf.tau *. player.velocity) in
      { player with velocity; y }

let rec add_pipes conf pipes =
  let last_x = match pipes with [] -> conf.pipe_gap | h :: _ -> h.x in
  if last_x < 2. *. conf.window_width then
    let margin = conf.pipe_gap +. (conf.window_height *. 0.05) in
    let new_pipe =
      {
        x = last_x +. conf.pipe_interval;
        gap_y = Random.float (conf.window_height -. (2. *. margin)) +. margin;
        passed = false;
      }
    in
    add_pipes conf (new_pipe :: pipes)
  else pipes

let remove_pipes conf pipes =
  List.rev
    (List.fold_left
       (fun acc p -> if p.x < -.conf.pipe_width then acc else p :: acc)
       [] pipes)

let update_pipes conf pipes =
  let moved_pipes =
    List.map (fun p -> { p with x = p.x -. (conf.tau *. conf.speed_x) }) pipes
  in
  let to_score = ref false in
  let passed_pipes =
    List.map
      (fun p ->
        if
          (not p.passed)
          && p.x +. (conf.pipe_width *. 0.5) < conf.window_width *. 0.5
        then to_score := true;
        {
          p with
          passed = p.x +. (conf.pipe_width *. 0.5) < conf.window_width *. 0.5;
        })
      moved_pipes
  in
  (remove_pipes conf (add_pipes conf passed_pipes), !to_score)

let check_collision conf player pipes =
  let wall_collision = player.y < 0. || player.y > conf.window_height in
  let pipe_collision =
    List.fold_left
      (fun acc p ->
        acc
        || abs_float (p.x -. (conf.window_width *. 0.5))
           <= (conf.pipe_width *. 0.5) +. conf.bird_radius
           && abs_float (player.y -. p.gap_y)
              > (conf.pipe_gap *. 0.5) -. conf.bird_radius)
      false pipes
  in
  wall_collision || pipe_collision

let next_frame state is_jump =
  let player = update_bird state.config state.player is_jump in
  let pipes, to_score = update_pipes state.config state.pipes in
  let alive =
    state.player.alive && not (check_collision state.config player pipes)
  in
  let score = if to_score && alive then state.score + 1 else state.score in
  {
    player = { player with alive };
    pipes;
    score;
    time = state.time + 1;
    config = state.config;
  }

let get_inputs state =
  let bird_x = state.config.window_width *. 0.5 in

  let target_pipe =
    List.fold_left
      (fun closest p ->
        let p_right_side = p.x +. (state.config.pipe_width *. 0.5) in
        let bird_left_side = bird_x -. state.config.bird_radius in

        if p_right_side > bird_left_side then
          match closest with
          | None -> Some p
          | Some c ->
              if p.x < c.x then Some p
              else Some c (* On prend le plus petit X (le plus à gauche) *)
        else closest)
      None state.pipes
  in

  match target_pipe with
  | None -> [ 0.0; 1.0; 0.0 ]
  | Some p ->
      [
        (p.x -. bird_x) /. state.config.window_width;
        (p.gap_y -. state.player.y) /. state.config.window_height;
        state.player.velocity
        /. 10.0 (* Normalisation approximative de la vitesse *);
      ]

let evaluator g =
  let max_ticks = 3600 in
  (*Correspond à une minute "réelle" i.e. lors des simulations visuelles*)

  let phenotype = Phenotype.create_phenotype g in
  let conf =
    {
      window_width = 400.;
      window_height = 600.;
      bird_radius = 15.;
      gravity = 1800.;
      jump_strength = 500.;
      pipe_width = 60.;
      pipe_gap = 140.;
      pipe_interval = 250.;
      speed_x = 200.;
      tau = 0.016;
    }
  in

  let state = ref (init_game conf) in
  while !state.time < max_ticks && !state.player.alive do
    let inputs = get_inputs !state in
    let pred = List.hd (Phenotype.predict phenotype inputs) in
    let action = if pred > 0.5 then true else false in
    state := next_frame !state action
  done;
  let fitness = !state.time + (1000 * !state.score) in
  float fitness

let _play_visual_game genome config generation =
  let phenotype = Phenotype.create_phenotype genome in
  let state = ref (init_game config) in
  let continue = ref true in

  while !continue do
    View._render !state generation;

    let inputs = get_inputs !state in
    let pred = List.hd (Phenotype.predict phenotype inputs) in
    let action = pred > 0.5 in

    state := next_frame !state action;

    if (not !state.player.alive) || !state.time > 3600 then continue := false;

    Unix.sleepf 0.016
  done

let play_visual_population genomes config generation =
  let population =
    List.map
      (fun g ->
        ( Phenotype.create_phenotype g,
          { y = config.window_height /. 2.; velocity = 0.; alive = true } ))
      genomes
  in

  let rec loop birds pipes score time =
    View.clear ();
    View.draw_pipes pipes config.pipe_width config.window_height config.pipe_gap;

    List.iter
      (fun (_, bird) ->
        if bird.alive then View.draw_bird bird config.bird_radius)
      birds;

    View.draw_score score generation;
    Graphics.synchronize ();

    let moved_pipes, to_score = update_pipes config pipes in

    let next_birds =
      List.map
        (fun (phenotype, bird) ->
          if not bird.alive then (phenotype, bird)
          else
            let temp_state = { player = bird; pipes; score; time; config } in

            let inputs = get_inputs temp_state in
            let pred = List.hd (Phenotype.predict phenotype inputs) in
            let jump = pred > 0.5 in

            let new_bird_phys = update_bird config bird jump in

            let is_alive =
              new_bird_phys.alive
              && not (check_collision config new_bird_phys moved_pipes)
            in

            (phenotype, { new_bird_phys with alive = is_alive }))
        birds
    in

    let any_alive = List.exists (fun (_, b) -> b.alive) next_birds in

    if any_alive && time < 1800 then (
      let new_score = if to_score then score + 1 else score in
      Unix.sleepf 0.01;
      loop next_birds moved_pipes new_score (time + 1))
    else ()
  in

  loop population [] 0 0

let main () =
  Random.self_init ();
  let number_inputs = 3 in
  let number_outputs = 1 in
  let pop_size = 300 in
  let epochs = 100 in
  let target_fitness = 30000. in

  View.open_window 400. 600.;

  let dynamic_threshold = ref 3. in
  let innov = Innovation.create (number_inputs + number_outputs + 5) in
  let pop =
    ref (Evolution.create_pop pop_size number_inputs number_outputs innov)
  in
  let l_species = ref [] in

  let start_time = Unix.gettimeofday () in
  let total_evals = ref 0 in

  (try
     for epoch = 0 to epochs - 1 do
       total_evals := !total_evals + pop_size;
       let new_pop, new_sp, genomes_evaluated =
         Evolution.generation !pop !l_species evaluator innov dynamic_threshold
       in
       pop := new_pop;
       l_species := new_sp;

       let best_genome =
         List.fold_left
           (fun (acc : Types.genome) (g : Types.genome) ->
             if g.fitness > acc.fitness then g else acc)
           (List.hd genomes_evaluated)
           genomes_evaluated
       in

       let sorted_genomes =
         List.sort
           (fun g1 g2 -> compare g2.fitness g1.fitness)
           genomes_evaluated
       in
       let to_simulate = List.take 30 sorted_genomes in

       Evolution.print_pop_summary !pop !l_species epoch epochs;

       play_visual_population to_simulate
         {
           window_width = 400.;
           window_height = 600.;
           bird_radius = 15.;
           gravity = 1800.;
           jump_strength = 500.;
           pipe_width = 60.;
           pipe_gap = 140.;
           pipe_interval = 250.;
           speed_x = 200.;
           tau = 0.016;
         }
         epoch;

       if best_genome.fitness > target_fitness then raise Break
     done
   with
  | Break -> ()
  | exn -> raise exn);

  let duration = Unix.gettimeofday () -. start_time in
  let best_genome =
    List.fold_left
      (fun (acc : genome) (g : genome) ->
        if g.fitness > acc.fitness then g else acc)
      (List.hd !pop.genomes) !pop.genomes
  in

  let avg_fitness =
    let sum = List.fold_left (fun acc g -> acc +. g.fitness) 0. !pop.genomes in
    sum /. float pop_size
  in
  Evolution.print_training_stats !total_evals duration avg_fitness
    best_genome.fitness;

  let filename = "flappy_net" in
  Parser.save_model best_genome filename;

  Graphics.close_graph ();

  Visualizer.init_window ();
  Visualizer.draw_genome best_genome;
  Graphics.synchronize ();

  ignore (Graphics.read_key ());
  Graphics.close_graph ()

let () = main ()
