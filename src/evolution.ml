open Genome
open Mutation
open Speciation
open Types

let create_pop pop_size number_inputs number_outputs innov_global =
  let genomes = ref [] in
  for _ = 0 to pop_size - 1 do
    let bias_node = { id = 0; kind = Sensor } in
    let nodes = ref [] and connections = ref [] in
    let output_ids =
      List.init number_outputs (fun i -> i + number_inputs + 1)
    in
    for id_in = 1 to number_inputs do
      let new_node = { id = id_in; kind = Sensor } in
      List.iter
        (fun id_out ->
          connections :=
            {
              in_node = id_in;
              out_node = id_out;
              weight = Random.float 4. -. 2.;
              enabled = true;
              innov =
                Innovation.get_innov_id innov_global id_in id_out Connexion;
            }
            :: !connections)
        output_ids;
      nodes := new_node :: !nodes
    done;
    for id_out = number_inputs + 1 to number_inputs + number_outputs do
      let new_node = { id = id_out; kind = Output } in
      nodes := new_node :: !nodes
    done;
    List.iter
      (fun id_out ->
        connections :=
          {
            in_node = 0;
            out_node = id_out;
            weight = 1.;
            enabled = true;
            innov = Innovation.get_innov_id innov_global 0 id_out Connexion;
          }
          :: !connections)
      output_ids;
    nodes := List.rev !nodes;
    nodes := bias_node :: !nodes;
    connections := List.rev !connections;
    genomes :=
      {
        nodes = !nodes;
        connections = !connections;
        fitness = 0.;
        adj_fitness = 0.;
      }
      :: !genomes
  done;
  { pop_size; genomes = !genomes }

let speciate_population genomes l_species threshold =
  let new_species = ref l_species in
  List.iter
    (fun g -> new_species := speciate_gemome !new_species g threshold)
    genomes;
  !new_species

let update_repr l_species =
  List.map
    (fun s ->
      let new_repr = List.nth s.members (Random.int (List.length s.members)) in
      {
        sp_id = s.sp_id;
        repr = new_repr;
        members = s.members;
        best_fitness = s.best_fitness;
        stagn_count = s.stagn_count;
        spawn_amount = s.spawn_amount;
      })
    l_species

let get_best_fitness s =
  List.fold_left max 0. (List.map (fun g -> g.fitness) s.members)

let manage_satgn s =
  let best_fitness = get_best_fitness s in
  if best_fitness > s.best_fitness then
    {
      sp_id = s.sp_id;
      repr = s.repr;
      members = s.members;
      best_fitness;
      stagn_count = 0;
      spawn_amount = s.spawn_amount;
    }
  else
    {
      sp_id = s.sp_id;
      repr = s.repr;
      members = s.members;
      best_fitness = s.best_fitness;
      stagn_count = s.stagn_count + 1;
      spawn_amount = s.spawn_amount;
    }

let calculate_adj_fitness l_species =
  List.map
    (fun s ->
      let n_members = List.length s.members in
      let new_members =
        List.map
          (fun g -> { g with adj_fitness = g.fitness /. float n_members })
          s.members
      in
      { s with members = new_members })
    l_species

let calculate_spawn_amounts l_species pop_size =
  let total_adj_fitness =
    List.fold_left
      (fun acc s ->
        List.fold_left (fun acc_g g -> acc_g +. g.adj_fitness) acc s.members)
      0. l_species
  in

  let temp_species =
    List.map
      (fun s ->
        let species_adj_fitness =
          List.fold_left (fun acc g -> acc +. g.adj_fitness) 0. s.members
        in
        let share =
          if total_adj_fitness > 0. then
            species_adj_fitness /. total_adj_fitness
          else 0.
        in
        let raw_spawn = int_of_float (share *. float pop_size) in
        { s with spawn_amount = raw_spawn })
      l_species
  in

  let total_allocated =
    List.fold_left (fun acc s -> acc + s.spawn_amount) 0 temp_species
  in
  let remainder = pop_size - total_allocated in

  if remainder > 0 then
    let sorted_sp =
      List.sort
        (fun s1 s2 -> compare s2.best_fitness s1.best_fitness)
        temp_species
    in
    match sorted_sp with
    | [] -> [] (* Ne devrait pas arriver si pop_size > 0 *)
    | best :: rest ->
        { best with spawn_amount = best.spawn_amount + remainder } :: rest
  else temp_species

let reproduce_species s best_genome innov =
  let n_members = List.length s.members in
  let to_spawn = ref s.spawn_amount in

  let sorted_members =
    List.sort (fun g1 g2 -> compare g2.fitness g1.fitness) s.members
  in

  let best_sp_genome = List.hd sorted_members in

  let childs = ref [] in
  if best_sp_genome = best_genome then begin
    childs := [ best_genome ];
    decr to_spawn
  end
  else if n_members >= 5 && !to_spawn > 0 then begin
    childs := List.hd sorted_members :: !childs;
    decr to_spawn
  end;

  let to_take = max 1 (n_members / 2) in

  let parents_list = List.take to_take sorted_members in
  let parents = Array.of_list parents_list in

  let n_parents = Array.length parents in
  if n_parents = 0 then failwith "No parents available in species!";

  while !to_spawn > 0 do
    let seed = Random.int 100 in
    let p1 = parents.(Random.int n_parents) in

    let child =
      if seed < 75 && n_parents > 1 then
        let p2 = parents.(Random.int n_parents) in
        crossover p1 p2
      else p1
    in

    let mutated = mutate child innov in
    childs := mutated :: !childs;
    decr to_spawn
  done;
  !childs

let delete_empty_sp l_species =
  List.rev
    (List.fold_left
       (fun acc s -> if s.members <> [] then s :: acc else acc)
       [] l_species)

let delete_stagn l_species =
  List.rev
    (List.fold_left
       (fun acc s -> if s.stagn_count <= 15 then s :: acc else acc)
       [] l_species)

let generation pop l_species evaluator innov_global dynamic_treshold =
  let new_fitness_genomes =
    Parmap.parmap ~ncores:8
      (fun g ->
        let new_fitness = evaluator g in
        { g with fitness = new_fitness })
      (Parmap.L pop.genomes)
  in

  let target_species = 20 in
  let nb_species = List.length l_species in

  let delta_species = float (target_species - nb_species) in
  dynamic_treshold := max 0.5 (!dynamic_treshold -. (delta_species *. 0.05));

  let empty_species = clear_species_members l_species in
  let species_filled =
    delete_empty_sp
      (speciate_population new_fitness_genomes empty_species !dynamic_treshold)
  in

  let species_updated_stats = List.map manage_satgn species_filled in

  let species_active = delete_stagn species_updated_stats in
  let species_adjusted = calculate_adj_fitness species_active in

  let species_calculated =
    calculate_spawn_amounts species_adjusted pop.pop_size
  in

  let new_genomes = ref [] in
  let best_genome =
    List.fold_left
      (fun acc g -> if g.fitness > acc.fitness then g else acc)
      (List.hd new_fitness_genomes)
      new_fitness_genomes
  in
  List.iter
    (fun s ->
      let babies = reproduce_species s best_genome innov_global in
      new_genomes := babies @ !new_genomes)
    species_calculated;

  let species_updated = update_repr species_active in

  let new_pop = { pop_size = pop.pop_size; genomes = !new_genomes } in
  (new_pop, species_updated, new_fitness_genomes)

let rec get_digits n = match n / 10 with 0 -> 1 | n' -> 1 + get_digits n'

let print_pop_summary pop l_species epoch max_epochs =
  (* --- Constants for layout --- *)
  let lines_to_clear = 17 in
  (* Must match the total lines printed below *)

  (* --- Move cursor up and clear lines if not first epoch --- *)
  if epoch > 0 then Printf.printf "\027[%dA" lines_to_clear;

  (* --- Progress Bar --- *)
  let width = 50 in
  let progress =
    if max_epochs > 0 then float epoch /. float max_epochs else 0.
  in
  let filled = int_of_float (float width *. progress) in
  let bar = String.make filled '#' ^ String.make (max 0 (width - filled)) '.' in
  Printf.printf "\rProgress: [%s] %3d%% (%d/%d)\027[K\n" bar
    (int_of_float (progress *. 100.))
    epoch max_epochs;

  (* --- Header --- *)
  Printf.printf "===================== Epoch %d =====================\027[K\n"
    epoch;
  Printf.printf "=============== POPULATION SUMMARY ===============\027[K\n";
  Printf.printf "--------------------------------------------------\027[K\n";

  (* --- Species Table (Fixed Height: Header + 5 rows) --- *)
  let sorted_species =
    List.take 5
      (List.sort
         (fun s1 s2 -> compare s2.best_fitness s1.best_fitness)
         l_species)
  in

  Printf.printf "Species [%d active(s)]:\027[K\n" (List.length l_species);
  Printf.printf "  %-4s | %-8s | %-12s | %-8s\027[K\n" "ID" "Members" "Best Fit"
    "Stagn";
  Printf.printf "  -----+----------+--------------+--------\027[K\n";

  (* Print top 5 species, pad with empty lines if fewer than 5 *)
  for i = 0 to 4 do
    if i < List.length sorted_species then
      let s = List.nth sorted_species i in
      Printf.printf "  %-4d | %-8d | %-12.4f | %-8d\027[K\n" s.sp_id
        (List.length s.members) s.best_fitness s.stagn_count
    else Printf.printf "                                          \027[K\n"
  done;

  (* --- Best Genome Stats (Fixed Height: ~4 lines) --- *)
  let best_genome_opt =
    match pop.genomes with
    | [] -> None
    | h :: t ->
        Some
          (List.fold_left
             (fun acc g -> if g.fitness > acc.fitness then g else acc)
             h t)
  in

  (match best_genome_opt with
  | Some g ->
      let enabled_conns = List.filter (fun c -> c.enabled) g.connections in
      Printf.printf "\nGlobal Best Genome Stats:\027[K\n";
      Printf.printf "  Fitness     : %.4f\027[K\n" g.fitness;
      Printf.printf "  Nodes       : %d\027[K\n" (List.length g.nodes);
      Printf.printf "  Connections : %d (Enabled: %d)\027[K\n"
        (List.length g.connections)
        (List.length enabled_conns)
  | None ->
      Printf.printf "\nGlobal Best Genome: N/A (Population empty)\027[K\n";
      Printf.printf " \027[K\n";
      Printf.printf " \027[K\n";
      Printf.printf " \027[K\n";

      Printf.printf "==================================================\027[K\n");
  flush stdout
