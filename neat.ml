module IntSet = Set.Make (Int)

type innovation_id = int
type node_id = int
type node_type = Sensor | Hidden | Output
type mutation_type = Connexion | Node
type mutation = { innov : innovation_id; kind : mutation_type }
type node_gene = { id : node_id; kind : node_type }

type connection_gene = {
  in_node : node_id;
  out_node : node_id;
  weight : float;
  enabled : bool;
  innov : innovation_id;
}

type genome = {
  (* This is the genotype representation of a neural network *)
  nodes : node_gene list;
  connections : connection_gene list;
  fitness : float;
  adj_fitness : float;
}

type incoming_link = { source_id : int; weight : float }

type neuron = {
  kind : node_type;
  incoming : incoming_link list;
  mutable value : float;
}

type network = {
  neuron_map : (int, neuron) Hashtbl.t;
  inputs : int list;
  outputs : int list;
}

type species = {
  sp_id : int;
  repr : genome;
  members : genome list;
  best_fitness : float;
  stagn_count : int;
  spawn_amount : int;
}

module InnovationManager = struct
  type t = {
    curr : innovation_id ref;
    mutations : (node_id * node_id * mutation_type, innovation_id) Hashtbl.t;
  }

  let create init_id = { curr = ref init_id; mutations = Hashtbl.create 16 }
  let reset_innovation innov = Hashtbl.reset innov.mutations

  let get_innov_id innov source_id target_id mut =
    match Hashtbl.find_opt innov.mutations (source_id, target_id, mut) with
    | None ->
        incr innov.curr;
        Hashtbl.add innov.mutations (source_id, target_id, mut) !(innov.curr);
        !(innov.curr)
    | Some c -> c
end

let reconstruct_nodes (child_conns : connection_gene list) (p1 : genome)
    (p2 : genome) : node_gene list =
  let expected_size = List.length p1.nodes + List.length p2.nodes in
  let node_lookup = Hashtbl.create expected_size in

  let add_to_table genome =
    List.iter
      (fun node -> Hashtbl.replace node_lookup node.id node)
      genome.nodes
  in

  add_to_table p1;
  add_to_table p2;

  let initial_set =
    Hashtbl.fold
      (fun id (node : node_gene) acc_set ->
        match node.kind with
        | Sensor | Output -> IntSet.add id acc_set
        | Hidden -> acc_set)
      node_lookup IntSet.empty
  in

  let required_ids =
    List.fold_left
      (fun acc_set conn ->
        acc_set |> IntSet.add conn.in_node |> IntSet.add conn.out_node)
      initial_set child_conns
  in

  let child_nodes =
    IntSet.fold
      (fun id acc_list ->
        try
          let node = Hashtbl.find node_lookup id in
          node :: acc_list
        with Not_found ->
          Printf.printf "Warning: Node %d referenced but not found in parents\n"
            id;
          acc_list)
      required_ids []
  in

  List.sort (fun n1 n2 -> compare n1.id n2.id) child_nodes

let crossover g1 g2 =
  (* Exepected to neural network genomes, connection genes are exepected to always be sorted by innov growing *)
  (* f is the fitness function that evaluates the nework *)
  let is_best_1 = g1.fitness > g2.fitness in
  let rec aux gene1 gene2 =
    match (gene1, gene2) with
    | [], [] -> []
    | h :: t, [] when is_best_1 -> h :: aux t []
    | [], h :: t when not is_best_1 -> h :: aux [] t
    | h1 :: t1, h2 :: t2 ->
        if h1.innov = h2.innov then
          let child_gene = if Random.bool () then h1 else h2 in
          child_gene :: aux t1 t2
        else if h1.innov < h2.innov && is_best_1 then h1 :: aux t1 gene2
        else if h1.innov > h2.innov && not is_best_1 then h2 :: aux gene1 t2
        else aux t1 t2
    | _ -> []
  in
  let new_connexion_gene = aux g1.connections g2.connections in
  let new_nodes = reconstruct_nodes new_connexion_gene g1 g2 in
  {
    connections = new_connexion_gene;
    nodes = new_nodes;
    fitness = 0.;
    adj_fitness = 0.;
  }

let mutate_weights g =
  let conn =
    List.map
      (fun (c : connection_gene) ->
        let seed = Random.int 100 in
        if seed < 80 then
          let curr_weight = c.weight in
          let modify = Random.int 100 < 95 in
          if modify then
            let power = 0.5 in
            let new_weight =
              curr_weight +. (Random.float (2. *. power) -. power)
            in
            let new_weight = max (-20.) (min 20. new_weight) in
            {
              in_node = c.in_node;
              out_node = c.out_node;
              weight = new_weight;
              enabled = c.enabled;
              innov = c.innov;
            }
          else
            let new_weight = Random.float 20. -. 10. in
            {
              in_node = c.in_node;
              out_node = c.out_node;
              weight = new_weight;
              enabled = c.enabled;
              innov = c.innov;
            }
        else c)
      g.connections
  in
  {
    connections = conn;
    nodes = g.nodes;
    fitness = g.fitness;
    adj_fitness = g.adj_fitness;
  }

let mutate_topology g mod_type innov_global =
  let nodes_array = Array.of_list g.nodes in
  match mod_type with
  | Connexion -> begin
      let max_attempts = 20 in
      let attempts = ref 0 in
      let found = ref false in

      let best_start = ref 0 in
      let best_end = ref 0 in

      while (not !found) && !attempts < max_attempts do
        incr attempts;

        let s = ref (Random.int (Array.length nodes_array)) in
        while nodes_array.(!s).kind = Output do
          s := Random.int (Array.length nodes_array)
        done;

        let e = ref (Random.int (Array.length nodes_array)) in
        while nodes_array.(!e).kind = Sensor do
          e := Random.int (Array.length nodes_array)
        done;

        let exists =
          List.exists
            (fun c ->
              c.in_node = nodes_array.(!s).id
              && c.out_node = nodes_array.(!e).id)
            g.connections
        in

        let self_loop = nodes_array.(!s).id = nodes_array.(!e).id in

        if (not exists) && not self_loop then begin
          found := true;
          best_start := !s;
          best_end := !e
        end
      done;

      if !found then begin
        let weight = Random.float 20. -. 10. in
        let innov_id =
          InnovationManager.get_innov_id innov_global
            nodes_array.(!best_start).id nodes_array.(!best_end).id Connexion
        in
        let new_connection =
          {
            in_node = nodes_array.(!best_start).id;
            out_node = nodes_array.(!best_end).id;
            weight;
            enabled = true;
            innov = innov_id;
          }
        in
        {
          connections = g.connections @ [ new_connection ];
          nodes = g.nodes;
          fitness = g.fitness;
          adj_fitness = g.adj_fitness;
        }
      end
      else g
    end
  | Node ->
      let enabled_conns = List.filter (fun c -> c.enabled) g.connections in
      if enabled_conns = [] then g
      else
        let target_conn =
          List.nth enabled_conns (Random.int (List.length enabled_conns))
        in

        let new_connections_list =
          List.map
            (fun c -> if c = target_conn then { c with enabled = false } else c)
            g.connections
        in

        let new_id =
          InnovationManager.get_innov_id innov_global target_conn.in_node
            target_conn.out_node Node
        in
        let innov_id_in =
          InnovationManager.get_innov_id innov_global target_conn.in_node new_id
            Connexion
        in
        let innov_id_out =
          InnovationManager.get_innov_id innov_global new_id
            target_conn.out_node Connexion
        in

        let new_conn_in =
          {
            in_node = target_conn.in_node;
            out_node = new_id;
            weight = 1.;
            enabled = true;
            innov = innov_id_in;
          }
        in
        let new_conn_out =
          {
            in_node = new_id;
            out_node = target_conn.out_node;
            weight = target_conn.weight;
            enabled = true;
            innov = innov_id_out;
          }
        in
        let new_node = { id = new_id; kind = Hidden } in

        {
          connections = new_connections_list @ [ new_conn_in; new_conn_out ];
          nodes = g.nodes @ [ new_node ];
          fitness = g.fitness;
          adj_fitness = g.adj_fitness;
        }

let mutate g innov_global =
  let new_genome = mutate_weights g in
  let modified_gemome =
    if Random.int 100 < 5 then mutate_topology new_genome Connexion innov_global
    else if Random.int 100 < 3 then mutate_topology new_genome Node innov_global
    else new_genome
  in
  modified_gemome

let compatibility_distance g1 g2 =
  let c1 = 1. and c2 = 1. and c3 = 0.4 in
  let n =
    float (max (List.length g1.connections) (List.length g2.connections))
  in
  let rec aux gene1 gene2 =
    match (gene1, gene2) with
    | [], [] -> 0.
    | _ :: t, [] -> (c1 /. n) +. aux t []
    | [], _ :: t -> (c1 /. n) +. aux [] t
    | h1 :: t1, h2 :: t2 ->
        if h1.innov = h2.innov then
          (c3 *. abs_float (h1.weight -. h2.weight)) +. aux t1 t2
        else if h1.innov < h2.innov then (c2 /. n) +. aux t1 gene2
        else (c2 /. n) +. aux gene1 t2
  in
  aux g1.connections g2.connections

let sigmoid x = 1. /. (1. +. exp (-1. *. x))

let create_phenotype g =
  let neuron_map = Hashtbl.create 16 in
  let inputs = ref [] and outputs = ref [] in
  List.iter
    (fun (n : node_gene) ->
      let new_neuron = { kind = n.kind; incoming = []; value = 0. } in
      Hashtbl.add neuron_map n.id new_neuron;
      if n.kind = Output then outputs := n.id :: !outputs;
      if n.kind = Sensor then inputs := n.id :: !inputs)
    g.nodes;
  inputs := List.sort compare !inputs;
  outputs := List.sort compare !outputs;
  List.iter
    (fun (c : connection_gene) ->
      if c.enabled then begin
        let existing = Hashtbl.find neuron_map c.out_node in
        let new_link = { source_id = c.in_node; weight = c.weight } in
        let new_neuron =
          {
            kind = existing.kind;
            incoming = new_link :: existing.incoming;
            value = existing.value;
          }
        in
        Hashtbl.replace neuron_map c.out_node new_neuron
      end)
    g.connections;
  { neuron_map; inputs = !inputs; outputs = !outputs }

let predict nn ninputs =
  let epochs = 30 in
  if List.length ninputs <> List.length nn.inputs then
    failwith "Predict : wrong input size\n";
  List.iter2
    (fun i v -> (Hashtbl.find nn.neuron_map i).value <- v)
    nn.inputs ninputs;
  for _ = 0 to epochs - 1 do
    let temp_values = ref [] in
    Hashtbl.iter
      (fun i (n : neuron) ->
        if n.kind <> Sensor then begin
          let sum =
            List.fold_left
              (fun acc v ->
                let source = v.source_id and weight = v.weight in
                let curr_value = (Hashtbl.find nn.neuron_map source).value in
                acc +. (curr_value *. weight))
              0. n.incoming
          in
          (*let activated = if n.kind = Output then sum else sigmoid sum in*)
          let activated = sigmoid sum in
          temp_values := (i, activated) :: !temp_values
        end)
      nn.neuron_map;
    List.iter
      (fun (i, v) -> (Hashtbl.find nn.neuron_map i).value <- v)
      !temp_values
  done;
  List.rev
    (List.fold_left
       (fun acc i -> (Hashtbl.find nn.neuron_map i).value :: acc)
       [] nn.outputs)

let clear_species_members l_species =
  List.map
    (fun s ->
      {
        sp_id = s.sp_id;
        members = [];
        spawn_amount = s.spawn_amount;
        repr = s.repr;
        best_fitness = s.best_fitness;
        stagn_count = s.stagn_count;
      })
    l_species

let speciate_gemome l_species g =
  let treshold = 3.0 in
  let has_fit = ref false in
  let new_species =
    List.map
      (fun s ->
        let delta = compatibility_distance s.repr g in
        if delta < treshold && not !has_fit then begin
          has_fit := true;
          {
            sp_id = s.sp_id;
            members = g :: s.members;
            spawn_amount = s.spawn_amount;
            repr = s.repr;
            best_fitness = s.best_fitness;
            stagn_count = s.stagn_count;
          }
        end
        else s)
      l_species
  in
  if not !has_fit then
    let max_id =
      List.fold_left (fun max_id s -> max s.sp_id max_id) 0 l_species
    in
    {
      sp_id = max_id + 1;
      repr = g;
      members = [ g ];
      best_fitness = g.fitness;
      stagn_count = 0;
      spawn_amount = 0;
    }
    :: l_species
  else new_species

let get_fitness g dataset f =
  let n = float (Array.length dataset) in
  let phenotype = create_phenotype g in
  1. /. n
  *. Array.fold_left
       (fun acc (i, o) ->
         let fitness = f (predict phenotype i) o in
         fitness +. acc)
       0. dataset
