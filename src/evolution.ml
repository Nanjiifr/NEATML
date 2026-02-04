open Genome
open Mutation
open Speciation
open Types

(* Helper function for List.take which doesn't exist in older OCaml *)
let list_take n lst =
  let rec aux n acc = function
    | [] -> List.rev acc
    | h :: t -> if n <= 0 then List.rev acc else aux (n - 1) (h :: acc) t
  in
  aux n [] lst

let create_pop pop_size number_inputs number_outputs innov_global =
  let genomes = ref [] in
  for _ = 0 to pop_size - 1 do
    let bias_node = { id = 0; kind = Sensor; activation = Identity } in
    let nodes = ref [] and connections = ref [] in
    let output_ids =
      List.init number_outputs (fun i -> i + number_inputs + 1)
    in
    for id_in = 1 to number_inputs do
      let new_node = { id = id_in; kind = Sensor; activation = Identity } in
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
      let new_node =
        {
          id = id_out;
          kind = Output;
          activation = Mutation.random_activation ();
        }
      in
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

  let parents_list = list_take to_take sorted_members in
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
  (* Find global max fitness across all species to identify the best species *)
  let global_best_fitness =
    List.fold_left (fun acc s -> max acc s.best_fitness) 0. l_species
  in
  List.rev
    (List.fold_left
       (fun acc s ->
         (* Keep if stagnation is low OR it contains the global best *)
         if s.stagn_count <= 15 || s.best_fitness >= global_best_fitness then
           s :: acc
         else acc)
       [] l_species)

let generation pop l_species evaluator innov_global dynamic_treshold
    ?hyperneat_config () =
  (* Find best from PREVIOUS generation (to ensure monotonicity) *)
  let best_from_prev =
    match pop.genomes with
    | [] -> None
    | h :: t ->
        Some
          (List.fold_left
             (fun acc g -> if g.fitness > acc.fitness then g else acc)
             h t)
  in

  let new_fitness_genomes =
    Parmap.parmap ~ncores:8
      (fun g ->
        let eval_genome =
          match hyperneat_config with
          | Some hn_config ->
              (* For HyperNEAT: g is a CPPN, convert to substrate network *)
              Hyperneat.create_substrate_network g hn_config
          | None ->
              (* For regular NEAT: use genome directly *)
              g
        in
        let new_fitness = evaluator eval_genome in
        { g with fitness = new_fitness })
      (Parmap.L pop.genomes)
  in

  (* Explicitly add the previous best to the pool to prevent fitness drop due to noise *)
  let new_fitness_genomes =
    match best_from_prev with
    | Some best -> best :: new_fitness_genomes
    | None -> new_fitness_genomes
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

(* ANSI Color codes *)
let color_bold = "\027[1m"
let color_red = "\027[31m"
let color_green = "\027[32m"
let color_yellow = "\027[33m"
let color_blue = "\027[34m"
let color_magenta = "\027[35m"
let color_cyan = "\027[36m"
let color_reset = "\027[0m"

let print_pop_summary pop l_species epoch max_epochs =
  (* --- Constants for layout --- *)
  let lines_to_clear = 18 in
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

  let bar_color = if progress >= 1.0 then color_green else color_cyan in

  Printf.printf "\rProgress: [%s%s%s%s] %3d%% (%d/%d)\027[K\n" color_bold
    bar_color bar color_reset
    (int_of_float (progress *. 100.))
    epoch max_epochs;

  (* --- Header --- *)
  let epoch_text = Printf.sprintf " Epoch %d " epoch in
  let total_width = 50 in
  let padding = total_width - String.length epoch_text in
  let left_pad = padding / 2 in
  let right_pad = padding - left_pad in
  let epoch_line =
    String.make left_pad '=' ^ epoch_text ^ String.make right_pad '='
  in

  Printf.printf "%s%s%s\027[K\n" (color_bold ^ color_blue) epoch_line
    color_reset;
  Printf.printf "%s=============== POPULATION SUMMARY ===============%s\027[K\n"
    (color_bold ^ color_blue) color_reset;
  Printf.printf "%s--------------------------------------------------%s\027[K\n"
    color_blue color_reset;

  (* --- Species Table (Fixed Height: Header + 5 rows) --- *)
  let sorted_species =
    list_take 5
      (List.sort
         (fun s1 s2 -> compare s2.best_fitness s1.best_fitness)
         l_species)
  in

  Printf.printf "Species [%s%d active(s)%s]:\027[K\n" (color_bold ^ color_green)
    (List.length l_species) color_reset;
  Printf.printf "  %s%-4s | %-8s | %-12s | %-8s%s\027[K\n" color_bold "ID"
    "Members" "Best Fit" "Stagn" color_reset;
  Printf.printf "  %s-----+----------+--------------+--------%s\027[K\n"
    color_blue color_reset;

  (* Print top 5 species, pad with empty lines if fewer than 5 *)
  for i = 0 to 4 do
    if i < List.length sorted_species then
      let s = List.nth sorted_species i in
      let stagn_color =
        if s.stagn_count > 10 then color_red
        else if s.stagn_count > 5 then color_yellow
        else ""
      in
      Printf.printf "  %-4d | %-8d | %-12.4f | %s%-8d%s\027[K\n" s.sp_id
        (List.length s.members) s.best_fitness stagn_color s.stagn_count
        color_reset
    else Printf.printf "                                          \027[K\n"
  done;

  (* --- Best Genome Stats (Fixed Height: ~6 lines) --- *)
  (* 1 blank line + 1 header + 3 stats + 1 separator = 6 lines *)
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
      Printf.printf "\n%sGlobal Best Genome Stats:%s\027[K\n"
        (color_bold ^ color_magenta)
        color_reset;
      Printf.printf "  Fitness     : %s%.4f%s\027[K\n"
        (color_bold ^ color_green) g.fitness color_reset;
      Printf.printf "  Nodes       : %d\027[K\n" (List.length g.nodes);
      Printf.printf "  Connections : %d (Enabled: %d)\027[K\n"
        (List.length g.connections)
        (List.length enabled_conns);
      Printf.printf
        "%s==================================================%s\027[K\n"
        color_blue color_reset
  | None ->
      Printf.printf "\n%sGlobal Best Genome: N/A (Population empty)%s\027[K\n"
        color_red color_reset;
      Printf.printf " \027[K\n";
      Printf.printf " \027[K\n";
      Printf.printf " \027[K\n";

      Printf.printf
        "%s==================================================%s\027[K\n"
        color_blue color_reset);
  flush stdout

let print_training_stats total_evals duration avg_fitness best_fitness =
  let hours = int_of_float duration / 3600 in
  let minutes = int_of_float duration mod 3600 / 60 in
  let seconds = mod_float duration 60. in

  Printf.printf "\n%s================= TRAINING STATS ==================%s\n"
    (color_bold ^ color_blue) color_reset;
  Printf.printf "  %sTotal Evaluations :%s %d\n" color_bold color_reset
    total_evals;
  Printf.printf "  %sTotal Time        :%s %02dh %02dm %05.2fs\n" color_bold
    color_reset hours minutes seconds;
  Printf.printf "  %sAverage Fitness   :%s %.4f\n" color_bold color_reset
    avg_fitness;
  Printf.printf "  %sBest Fitness      :%s %s%.4f%s\n" color_bold color_reset
    (color_bold ^ color_green) best_fitness color_reset;
  Printf.printf "%s==================================================%s\n"
    (color_bold ^ color_blue) color_reset;
  flush stdout
