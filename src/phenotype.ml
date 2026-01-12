open Types

let sigmoid x = 1. /. (1. +. exp (-1. *. x))

let reset_network nn =
  Hashtbl.iter
    (fun key n -> Hashtbl.replace nn.neuron_map key { n with value = 0. })
    nn.neuron_map

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
  let ninputs = 1. :: ninputs in
  let epochs = 50 in
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
