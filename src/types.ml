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

type population = { pop_size : int; genomes : genome list }

(* --- FastNet Types --- *)
type connection = {
  src_idx : int;
  weight : float;
}

type fast_neuron = {
  mutable value : float;
  kind : node_type;
  incoming : connection array;
}

type network = {
  neurons : fast_neuron array;
  input_indices : int array;
  output_indices : int array;
  topo_order : int array;
  neuron_map : (int, int) Hashtbl.t; (* Map ID -> Array Index for lookup *)
  original_map : (int, node_gene) Hashtbl.t option; (* Optional: Keep track of original nodes for viz if needed *)
}

type species = {
  sp_id : int;
  repr : genome;
  members : genome list;
  best_fitness : float;
  stagn_count : int;
  spawn_amount : int;
}
