type innovation_id = int
type node_id = int
type node_type = Sensor | Hidden | Ouput
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

(* TODO : Implement the reconstruciton of a genome from a genotype, at least partially for the crossover function*)

let crossover g1 g2 f =
  (* Exepected to neural network genomes, connection genes are exepected to always be sorted by innov growing *)
  (* f is the fitness function that evaluates the nework *)
  let is_best_1 = f g1 > f g2 in
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
  aux g1.connections g2.connections
