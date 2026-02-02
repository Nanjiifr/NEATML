open Types

(* let sigmoid x = 1. /. (1. +. exp (-1. *. x)) *)
let sigmoid x = x

let create_phenotype g =
  (* Map ID -> Array Index *)
  let id_to_idx = Hashtbl.create 16 in
  let idx_counter = ref 0 in

  (* Identify Inputs and Outputs from Genome *)
  let inputs_list =
    List.filter (fun (n : node_gene) -> n.kind = Sensor) g.nodes
    |> List.map (fun (n : node_gene) -> n.id)
    |> List.sort compare
  in
  let outputs_list =
    List.filter (fun (n : node_gene) -> n.kind = Output) g.nodes
    |> List.map (fun (n : node_gene) -> n.id)
    |> List.sort compare
  in

  (* Assign indices: Inputs first *)
  List.iter
    (fun id ->
      if not (Hashtbl.mem id_to_idx id) then begin
        Hashtbl.add id_to_idx id !idx_counter;
        incr idx_counter
      end)
    inputs_list;

  (* Assign indices: Others *)
  List.iter
    (fun (n : node_gene) ->
      if not (Hashtbl.mem id_to_idx n.id) then begin
        Hashtbl.add id_to_idx n.id !idx_counter;
        incr idx_counter
      end)
    g.nodes;

  let num_neurons = !idx_counter in
  let neurons =
    Array.make num_neurons { value = 0.; kind = Hidden; incoming = [||] }
  in

  (* Build fast neurons *)
  (* First, build a map of incoming connections per node ID *)
  let incoming_map = Hashtbl.create num_neurons in
  List.iter
    (fun (c : connection_gene) ->
      if c.enabled then
        let current =
          try Hashtbl.find incoming_map c.out_node with Not_found -> []
        in
        Hashtbl.replace incoming_map c.out_node (c :: current))
    g.connections;

  List.iter
    (fun (n : node_gene) ->
      let idx = Hashtbl.find id_to_idx n.id in
      let incoming_genes =
        try Hashtbl.find incoming_map n.id with Not_found -> []
      in
      let incoming_arr =
        incoming_genes
        |> List.map (fun (c : connection_gene) ->
            try
              { src_idx = Hashtbl.find id_to_idx c.in_node; weight = c.weight }
            with Not_found ->
              (* Should not happen if genome is valid, but safe fallback or ignore *)
              { src_idx = 0; weight = 0.0 }
            (* Dummy, potentially dangerous if 0 is used *))
        |> List.filter (fun _ -> true) (* could filter dummy if needed *)
        |> Array.of_list
      in
      neurons.(idx) <- { value = 0.; kind = n.kind; incoming = incoming_arr })
    g.nodes;

  let input_indices =
    List.map (Hashtbl.find id_to_idx) inputs_list |> Array.of_list
  in
  let output_indices =
    List.map (Hashtbl.find id_to_idx) outputs_list |> Array.of_list
  in

  (* Topo order for update: All non-sensors *)
  let update_indices_list = ref [] in
  for i = 0 to num_neurons - 1 do
    if neurons.(i).kind <> Sensor then
      update_indices_list := i :: !update_indices_list
  done;

  (* Optional: Map original genes for visualizer if needed *)
  let original_map = Hashtbl.create (List.length g.nodes) in
  List.iter (fun (n : node_gene) -> Hashtbl.add original_map n.id n) g.nodes;

  {
    neurons;
    input_indices;
    output_indices;
    topo_order = Array.of_list (List.rev !update_indices_list);
    neuron_map = id_to_idx;
    original_map = Some original_map;
  }

let reset_network nn =
  for i = 0 to Array.length nn.neurons - 1 do
    nn.neurons.(i).value <- 0.0
  done

let predict nn inputs_list =
  let inputs = Array.of_list inputs_list in
  (* Note: Standard Phenotype.predict expected 1. :: ninputs. 
     Here we assume inputs_list contains exactly what the sensors need. 
     If bias is handled as a sensor in the genome (id 0 usually), it should be in the input list provided by the user OR handled here.
     In `evolution.ml`, bias is added as a Sensor node.
     In old `phenotype.ml`, `let ninputs = 1. :: ninputs` was done.
     So we must prepend 1.0 to the inputs array if the network expects it.
     The convention in this project seems to be: Caller provides N inputs, Network has N+1 inputs (Bias + N).
     Bias is usually the first sensor.
  *)

  (* Check size match *)
  (* Network inputs size: Array.length nn.input_indices *)
  (* If network has Bias + N inputs, and user provides N inputs, we prepend 1.0 *)

  let full_inputs =
    if Array.length nn.input_indices = Array.length inputs + 1 then
      Array.append [| 1.0 |] inputs
    else inputs
  in

  let n_inputs_provided = Array.length full_inputs in

  (* Set inputs *)
  for i = 0 to n_inputs_provided - 1 do
    if i < Array.length nn.input_indices then
      let idx = nn.input_indices.(i) in
      nn.neurons.(idx).value <- full_inputs.(i)
  done;

  (* Propagate (Epochs = 1) *)
  (* Using a temp array for synchronous update logic if needed, or simple forward pass if sorted. 
     The old implementation used a temp list and then updated. 
     Let's stick to that "snapshot" behavior for consistency with recurrent connections. *)
  let updates = Array.make (Array.length nn.neurons) 0.0 in
  let active_indices = nn.topo_order in

  (* Calculate activations *)
  for k = 0 to Array.length active_indices - 1 do
    let i = active_indices.(k) in
    let n = nn.neurons.(i) in
    let sum = ref 0. in
    let incoming = n.incoming in
    for j = 0 to Array.length incoming - 1 do
      let conn = incoming.(j) in
      sum := !sum +. (nn.neurons.(conn.src_idx).value *. conn.weight)
    done;
    updates.(i) <- sigmoid !sum
  done;

  (* Apply updates *)
  for k = 0 to Array.length active_indices - 1 do
    let i = active_indices.(k) in
    nn.neurons.(i).value <- updates.(i)
  done;

  (* Gather outputs *)
  Array.map (fun idx -> nn.neurons.(idx).value) nn.output_indices
  |> Array.to_list

let predict_array nn inputs =
  (* Same logic as predict but assumes array input and returns array output *)
  (* Bias handling: If network expects Bias+N and inputs has N, we assume Bias is 1.0 and prepended virtually or we copy. 
     To be fast, we might expect the caller to handle bias or we handle it efficiently. *)
  let full_inputs =
    if Array.length nn.input_indices = Array.length inputs + 1 then (
      (* Prepend 1.0 - Copying is necessary unless we have a different API *)
      let arr = Array.make (Array.length inputs + 1) 1.0 in
      Array.blit inputs 0 arr 1 (Array.length inputs);
      arr)
    else inputs
  in

  let n_inputs_provided = Array.length full_inputs in

  for i = 0 to n_inputs_provided - 1 do
    if i < Array.length nn.input_indices then
      let idx = nn.input_indices.(i) in
      nn.neurons.(idx).value <- full_inputs.(i)
  done;

  let updates = Array.make (Array.length nn.neurons) 0.0 in
  let active_indices = nn.topo_order in

  for k = 0 to Array.length active_indices - 1 do
    let i = active_indices.(k) in
    let n = nn.neurons.(i) in
    let sum = ref 0. in
    let incoming = n.incoming in
    for j = 0 to Array.length incoming - 1 do
      let conn = incoming.(j) in
      sum := !sum +. (nn.neurons.(conn.src_idx).value *. conn.weight)
    done;
    updates.(i) <- sigmoid !sum
  done;

  for k = 0 to Array.length active_indices - 1 do
    let i = active_indices.(k) in
    nn.neurons.(i).value <- updates.(i)
  done;

  Array.map (fun idx -> nn.neurons.(idx).value) nn.output_indices
