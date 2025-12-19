open Neat

type population = {
  pop_size: int;
  genomes: genome list;
}

let create_pop pop_size number_inputs number_outputs innov_global =
  let genomes = ref [] in
  for _ = 0 to pop_size - 1 do
    let bias_node = {id=0; kind=Sensor} in
    let nodes = ref []
    and connections = ref [] in
    let output_ids = List.init number_outputs (fun i -> i + number_inputs + 1) in
    for id_in = 1 to number_inputs do 
      let new_node = {id=id_in; kind=Sensor} in
      List.iter (fun id_out -> connections := {in_node = id_in; out_node=id_out; weight=Random.float 20. -. 10.; enabled=true; innov=InnovationManager.get_innov_id innov_global id_in id_out Connexion}::!connections) output_ids;
      nodes := new_node::!nodes ;
    done ;
    List.iter (fun id_out -> connections := {in_node = 0; out_node=id_out; weight=1.; enabled=true; innov=InnovationManager.get_innov_id innov_global 0 id_out Connexion}::!connections) output_ids ;
    nodes := List.rev !nodes ;
    nodes := bias_node::!nodes ; 
    connections := List.rev !connections ;
    genomes := {nodes=(!nodes); connections=(!connections); fitness=0.}::!genomes ;
  done ;

let generation pop f dataset =
  (*f is the fitness function*)
  let n = float (Array.length dataset) in
  let to_cross = pop.pop_size / 3 in
  let mutated_genomes = List.map (fun g -> mutate g innov_global) pop.genomes in
  
  let crossed_gemomes = ref [] in
  List.iteri (fun i g1 ->
    List.iteri (fun j g2 ->
      if j > i then
        crossed_gemomes := (crossover g1 g2)::!crossed_gemomes ;) (List.take to_cross pop.genomes)) (List.take to_cross pop.genomes) ;
  let new_genomes_fitness = List.map (fun g -> 
    let phenotype = create_phenotype g in
    let new_fitness = Array.fold_left (fun acc (inputs, outputs) -> f (predict phenotype inputs) outputs +. acc) 0. dataset in
    {nodes=g.nodes; connections=g.connections; fitness=new_fitness}) pop.gen in
  
  let new_genomes = List.sort (fun g1 g2 -> compare g1.fitness g2.fitness) new_genomes_fitness in
  