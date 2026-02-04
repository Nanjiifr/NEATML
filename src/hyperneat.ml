open Types
open Phenotype

(* Distance function between two coordinates *)
let distance coord1 coord2 =
  let sum_squares = List.fold_left2 
    (fun acc c1 c2 -> acc +. ((c1 -. c2) ** 2.0))
    0.0 coord1 coord2
  in
  sqrt sum_squares

(* CPPN activation function - can use different functions *)
let cppn_activation x = 
  (* Using sigmoid for now *)
  1. /. (1. +. exp (-1. *. x))

(* Query the CPPN network to get weight between two substrate nodes *)
let query_cppn cppn_network source_coord target_coord =
  (* CPPN input: [x1, y1, x2, y2, distance] *)
  let dist = distance source_coord target_coord in
  let inputs = source_coord @ target_coord @ [dist] in
  let outputs = predict cppn_network inputs in
  List.hd outputs (* Weight value *)

(* Create substrate network from CPPN genome *)
let create_substrate_network cppn_genome config =
  let cppn = create_phenotype cppn_genome in
  
  (* Create bias node at id 0 *)
  let bias_node = { position = [0.; 0.]; layer = 0; node_id = 0 } in
  
  (* Build list of all substrate nodes (starting from id 1 for inputs) *)
  let input_nodes = List.mapi (fun i coord -> 
    { position = coord; layer = 0; node_id = i + 1 }
  ) config.substrate.input_coords in
  
  let hidden_offset = 1 + List.length input_nodes in
  let hidden_nodes = List.mapi (fun i coord -> 
    { position = coord; layer = 1; node_id = hidden_offset + i }
  ) config.substrate.hidden_coords in
  
  let output_offset = hidden_offset + List.length hidden_nodes in
  let output_nodes = List.mapi (fun i coord -> 
    { position = coord; layer = 2; node_id = output_offset + i }
  ) config.substrate.output_coords in
  
  let all_nodes = bias_node :: input_nodes @ hidden_nodes @ output_nodes in
  
  (* Query CPPN for all potential connections (feedforward only) *)
  let connections = ref [] in
  List.iter (fun source ->
    List.iter (fun target ->
      (* Only create feedforward connections *)
      if source.layer < target.layer then begin
        let weight = query_cppn cppn source.position target.position in
        (* Apply weight threshold if LEO is enabled *)
        if not config.use_leo || abs_float weight > config.weight_threshold then
          connections := {
            in_node = source.node_id;
            out_node = target.node_id;
            weight;
            enabled = true;
            innov = 0; (* Not used in substrate *)
          } :: !connections
      end
    ) all_nodes
  ) all_nodes;
  
  (* Create genome representation of substrate *)
  let substrate_nodes = List.map (fun sn ->
    let kind = 
      if sn.layer = 0 then Sensor
      else if sn.layer = 2 then Output
      else Hidden
    in
    let activation = if kind = Sensor then Identity else Tanh in
    { id = sn.node_id; kind; activation }
  ) all_nodes in
  
  {
    nodes = substrate_nodes;
    connections = !connections;
    fitness = 0.;
    adj_fitness = 0.;
  }
