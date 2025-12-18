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
}

module InnovationManager = struct
  type t = {
    curr : innovation_id ref;
    mutations : (node_id * node_id * mutation_type, innovation_id) Hashtbl.t;
  }

  let create () = { curr = ref 100; mutations = Hashtbl.create 16 }
  let reset_innovation innov = Hashtbl.reset innov.mutations

  let get_innov_id innov source_id target_id mut =
    match Hashtbl.find_opt innov.mutations (source_id, target_id, mut) with
    | None ->
        incr innov.curr;
        Hashtbl.add innov.mutations (source_id, target_id, mut) !(innov.curr);
        !(innov.curr)
    | Some c -> c
end

(* TODO : Implement the reconstruciton of a genome from a genotype, at least partially for the crossover function *)

let reconstruct_nodes (child_conns : connection_gene list) (p1 : genome)
    (p2 : genome) : node_gene list =
  (* 1. CRÉATION DE LA TABLE DE RECHERCHE (Lookup Table) *)
  (* Taille estimée : somme des nœuds des deux parents *)
  let expected_size = List.length p1.nodes + List.length p2.nodes in
  let node_lookup = Hashtbl.create expected_size in

  (* Fonction pour remplir la table. 
     Hashtbl.replace permet de gérer les doublons sans erreur :
     si l'ID existe déjà (car présent chez les 2 parents), on le met juste à jour. *)
  let add_to_table genome =
    List.iter
      (fun node -> Hashtbl.replace node_lookup node.id node)
      genome.nodes
  in

  add_to_table p1;
  add_to_table p2;

  (* 2. IDENTIFICATION DES NOEUDS REQUIS *)
  (* On commence par récupérer les Inputs/Outputs qui doivent toujours être présents *)
  (* On peut parcourir la Hashtbl directement pour ça *)
  let initial_set =
    Hashtbl.fold
      (fun id node acc_set ->
        match node.kind with
        | Sensor | Output -> IntSet.add id acc_set
        | Hidden -> acc_set)
      node_lookup IntSet.empty
  in

  (* On ajoute les nœuds mentionnés dans les nouvelles connexions *)
  let required_ids =
    List.fold_left
      (fun acc_set conn ->
        acc_set |> IntSet.add conn.in_node |> IntSet.add conn.out_node)
      initial_set child_conns
  in

  (* 3. CONSTRUCTION DE LA LISTE FINALE *)
  (* On transforme le Set d'IDs en liste de NodeGene grâce à la Hashtbl *)
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
  { connections = new_connexion_gene; nodes = new_nodes; fitness = 0. }

let mutate_weights g =
  let conn =
    List.map
      (fun c ->
        let seed = Random.int 100 in
        if seed < 80 then
          let curr_weight = c.weight in
          let modify = Random.int 100 < 95 in
          if modify then
            let new_weight =
              curr_weight +. ((Random.float 0.2 -. 0.1) *. curr_weight)
            in
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
  { connections = conn; nodes = g.nodes; fitness = g.fitness }

let mutate_topology g mod_type innov_global =
  let nodes_array = Array.of_list g.nodes in
  match mod_type with
  | Connexion ->
      let ind_start = ref (Random.int (Array.length nodes_array)) in
      while nodes_array.(!ind_start).kind = Output do
        ind_start := Random.int (Array.length nodes_array)
      done;
      let ind_end = ref (Random.int (Array.length nodes_array)) in
      while nodes_array.(!ind_end).kind = Sensor do
        ind_end := Random.int (Array.length nodes_array)
      done;
      while
        List.fold_left
          (fun acc c ->
            c.in_node = nodes_array.(!ind_start).id
            && c.out_node = nodes_array.(!ind_end).id
            || acc)
          false g.connections
      do
        ind_start := Random.int (Array.length nodes_array);
        while nodes_array.(!ind_start).kind <> Output do
          ind_start := Random.int (Array.length nodes_array)
        done;
        ind_end := Random.int (Array.length nodes_array);
        while nodes_array.(!ind_end).kind <> Sensor do
          ind_end := Random.int (Array.length nodes_array)
        done
      done;
      let weight = Random.float 20. -. 10. in
      let innov_id =
        InnovationManager.get_innov_id innov_global nodes_array.(!ind_start).id
          nodes_array.(!ind_end).id Connexion
      in
      let new_connection =
        {
          in_node = !ind_start;
          out_node = !ind_end;
          weight;
          enabled = true;
          innov = innov_id;
        }
      in
      {
        connections = g.connections @ [ new_connection ];
        nodes = g.nodes;
        fitness = g.fitness;
      }
  | Node ->
      let enabled_conns = List.filter (fun c -> c.enabled) g.connections in
      if enabled_conns = [] then g (* Pas de mutation possible *)
      else
        let target_conn =
          List.nth enabled_conns (Random.int (List.length enabled_conns))
        in

        (* On crée la nouvelle liste où target_conn est désactivée *)
        let new_connections_list =
          List.map
            (fun c -> if c = target_conn then { c with enabled = false } else c)
            g.connections
        in

        (* On récupère les IDs via InnovationManager (Ton code était bon ici) *)
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

        (* Création des nouveaux gènes (Ton code était bon ici) *)
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
          (* On utilise la liste modifiée ! *)
          nodes = g.nodes @ [ new_node ];
          fitness = g.fitness;
        }

let mutate g innov_global =
  let new_genome = mutate_weights g in
  let modified_gemome =
    if Random.int 100 < 5 then mutate_topology new_genome Connexion innov_global
    else if Random.int 100 < 3 then mutate_topology new_genome Node innov_global
    else new_genome
  in
  modified_gemome
