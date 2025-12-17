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
