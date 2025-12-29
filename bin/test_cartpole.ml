open Neat
open Evolution

type active_agent = {
  id : int;
  phenotype : Neat.network;
  mutable state : Cartpole.state;
  genome : Neat.genome; (* Gardé pour l'affichage du réseau *)
}

let pop_size = 150
let number_inputs = 4
let number_outputs = 1
let epochs = 100
let max_ticks = 400
let nb_trys = 25

let evaluator genome =
  let total_fitness = ref 0 in
  for _ = 0 to nb_trys - 1 do
    let score = ref 0 in
    let state = ref (Cartpole.initial_state ()) in
    let phenotype = create_phenotype genome in
    while (not (Cartpole.is_failed !state)) && !score < max_ticks do
      let inputs = 1. :: Cartpole.get_inputs !state in
      let action = List.hd (predict phenotype inputs) in
      state := Cartpole.step !state action;
      incr score
    done;
    total_fitness := !score + !total_fitness
  done;
  float !total_fitness /. float nb_trys

(*let watch_genome genome =
  let state = ref (Cartpole.initial_state ()) in
  let phenotype = create_phenotype genome in
  let score = ref 0 in

  try
    while (not (Cartpole.is_failed !state)) && !score < max_ticks do
      let inputs = 1. :: Cartpole.get_inputs !state in
      let action = List.hd (predict phenotype inputs) in
      state := Cartpole.step !state action;
      incr score;

      Visualizer.draw_all !state !score genome;

      Unix.sleepf 0.02;

      if Graphics.key_pressed () then raise Exit
    done
  with Exit -> ()
*)

let watch_population population =
  Printf.printf "Starting population replay...\n";
  flush stdout;

  let sorted_genomes =
    List.sort (fun a b -> compare b.fitness a.fitness) population.genomes
  in

  let agents =
    List.mapi
      (fun i g ->
        {
          id = i;
          phenotype = create_phenotype g;
          state = Cartpole.initial_state ();
          (* Chaque agent a son propre état physique *)
          genome = g;
        })
      sorted_genomes
  in

  let current_agents = ref agents in
  let score = ref 0 in

  try
    while !current_agents <> [] && !score < max_ticks do
      let survivors =
        List.filter_map
          (fun agent ->
            let inputs = 1. :: Cartpole.get_inputs agent.state in
            let action = List.hd (predict agent.phenotype inputs) in

            let new_state = Cartpole.step agent.state action in
            agent.state <- new_state;

            if Cartpole.is_failed agent.state then None
              (* Il meurt, on le retire de la liste *)
            else Some agent (* Il survit *))
          !current_agents
      in

      current_agents := survivors;
      incr score;

      if !current_agents <> [] then begin
        let states = List.map (fun a -> a.state) !current_agents in
        let leader_genome = (List.hd !current_agents).genome in

        Visualizer.draw_population states !score leader_genome;

        Unix.sleepf 0.02
        (* 50 FPS *)
      end;

      if Graphics.key_pressed () then raise Exit
    done
  with Exit -> ()

let main () =
  Random.self_init ();
  Cartpole_vis.init_window ();

  let innov = InnovationManager.create (number_inputs + number_outputs + 5) in
  let pop = ref (create_pop pop_size number_inputs number_outputs innov) in
  let sp = ref [] in
  for epoch = 0 to epochs - 1 do
    let new_pop, new_sp = generation !pop !sp evaluator innov in
    pop := new_pop;
    sp := new_sp;

    print_pop_summary !pop !sp epoch;
    flush stdout;
    let best_genome =
      List.fold_left
        (fun acc g -> if g.fitness > acc.fitness then g else acc)
        (List.hd !pop.genomes) !pop.genomes
    in

    if best_genome.fitness > 50. && epoch mod 10 = 0 then begin
      Printf.printf "Visualizing full population...\n";
      flush stdout;
      watch_population !pop
    end
  done

let () = main ()
