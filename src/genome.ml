open Types

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
