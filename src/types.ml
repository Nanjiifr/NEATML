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
