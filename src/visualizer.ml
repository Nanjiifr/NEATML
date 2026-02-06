open Matplotlib
open Types

let string_of_activation = function
  | Sigmoid -> "Sigmoid"
  | Tanh -> "Tanh"
  | Relu -> "ReLU"
  | Gaussian -> "Gauss"
  | Sin -> "Sin"
  | Cos -> "Cos"
  | Abs -> "Abs"
  | Square -> "Sqr"
  | Identity -> "Id"

let calculate_positions (nodes : node_gene list) (connections : connection_gene list) =
  let positions = Hashtbl.create (List.length nodes) in
  let node_map = Hashtbl.create (List.length nodes) in
  List.iter (fun n -> Hashtbl.add node_map n.id n) nodes;

  (* 1. BFS to determine layers *)
  let layers = Hashtbl.create (List.length nodes) in
  let sensors = List.filter (fun (n : node_gene) -> n.kind = Sensor) nodes in
  let queue = Queue.create () in

  (* Initialize sensors at layer 0 *)
  List.iter (fun n -> 
    Hashtbl.add layers n.id 0;
    Queue.push n.id queue
  ) sensors;

  (* Propagate layers *)
  while not (Queue.is_empty queue) do
    let current_id = Queue.pop queue in
    let current_layer = Hashtbl.find layers current_id in
    
    List.iter (fun conn ->
      if conn.enabled && conn.in_node = current_id then
        let target_id = conn.out_node in
        let target_layer = 
          try Hashtbl.find layers target_id 
          with Not_found -> -1 
        in
        if target_layer < current_layer + 1 then (
          Hashtbl.replace layers target_id (current_layer + 1);
          Queue.push target_id queue
        )
    ) connections
  done;

  (* Ensure all nodes have a layer (orphans) *)
  List.iter (fun (n : node_gene) ->
    if not (Hashtbl.mem layers n.id) then
      match n.kind with
      | Sensor -> Hashtbl.add layers n.id 0
      | Output -> Hashtbl.add layers n.id 100 (* Will be adjusted later *)
      | Hidden -> Hashtbl.add layers n.id 1
  ) nodes;

  (* Final adjustment: outputs should be in the last layer *)
  let max_layer = Hashtbl.fold (fun _ l acc -> if l < 100 then max acc l else acc) layers 0 in
  List.iter (fun (n : node_gene) ->
    if n.kind = Output then Hashtbl.replace layers n.id (max_layer + 1)
  ) nodes;

  let final_max_layer = Hashtbl.fold (fun _ l acc -> max acc l) layers 0 in

  (* Group nodes by layer *)
  let nodes_by_layer = Hashtbl.create 10 in
  List.iter (fun n ->
    let l = Hashtbl.find layers n.id in
    let list = try Hashtbl.find nodes_by_layer l with Not_found -> [] in
    Hashtbl.replace nodes_by_layer l (n :: list)
  ) nodes;

  (* Calculate X and Y coordinates *)
  for l = 0 to final_max_layer do
    let layer_nodes = try Hashtbl.find nodes_by_layer l with Not_found -> [] in
    let n_in_layer = List.length layer_nodes in
    let x = if final_max_layer = 0 then 0.5 else 0.1 +. (float l *. (0.8 /. float final_max_layer)) in
    
    List.iteri (fun i node ->
      let y = 
        if n_in_layer = 1 then 0.5 
        else 0.1 +. (float i *. (0.8 /. float (n_in_layer - 1)))
      in
      Hashtbl.add positions node.id (x, y)
    ) (List.sort (fun a b -> compare a.id b.id) layer_nodes)
  done;
  positions

let draw_genome genome =
  let positions = calculate_positions genome.nodes genome.connections in
  let fig = Fig.create ~figsize:(12., 8.) () in
  let ax = Fig.add_subplot fig ~nrows:1 ~ncols:1 ~index:1 in
  
  (* Hide axes *)
  let ax_py = Ax.Expert.to_pyobject ax in
  ignore (Py.Module.get_function_with_keywords ax_py "axis" [| Py.String.of_string "off" |] []);

  (* 1. Draw connections *)
  List.iter
    (fun conn ->
      try
        let x1, y1 = Hashtbl.find positions conn.in_node in
        let x2, y2 = Hashtbl.find positions conn.out_node in
        if conn.enabled then (
          let color = if conn.weight >= 0. then "#00AA00" else "#AA0000" in
          let linewidth = min 5.0 (max 1.0 (abs_float conn.weight)) in
          let alpha = min 1.0 (max 0.2 (abs_float conn.weight /. 3.0)) in
          
          (* Use pyplot directly for more control over alpha if needed, 
             but here we use the Ax wrapper and if alpha isn't supported we skip it or use Py calls. 
             The standard wrapper usually doesn't expose alpha in plot. 
             Let's use a direct Python call on the axis object for 'plot' to support alpha.
          *)
          ignore (Py.Module.get_function_with_keywords ax_py "plot" 
            [| Py.List.of_array (Array.map Py.Float.of_float [|x1; x2|]); 
               Py.List.of_array (Array.map Py.Float.of_float [|y1; y2|]) |]
            [
              "color", Py.String.of_string color;
              "linewidth", Py.Float.of_float linewidth;
              "alpha", Py.Float.of_float alpha;
              "zorder", Py.Int.of_int 1
            ]
          )
        )
      with Not_found -> ())
    genome.connections;

  (* 2. Draw nodes *)
  List.iter
    (fun (node : node_gene) ->
      try
        let x, y = Hashtbl.find positions node.id in
        let color, size = 
          match node.kind with
          | Sensor -> "#AAAAFF", 600.
          | Output -> "#FFAAAA", 600.
          | Hidden -> "#DDDDDD", 400.
        in
        
        (* Draw Node Circle *)
        ignore (Py.Module.get_function_with_keywords ax_py "scatter"
          [| Py.List.of_array (Array.map Py.Float.of_float [|x|]); 
             Py.List.of_array (Array.map Py.Float.of_float [|y|]) |]
          [
            "c", Py.String.of_string color;
            "s", Py.Float.of_float size;
            "edgecolors", Py.String.of_string "black";
            "zorder", Py.Int.of_int 2
          ]
        );

        (* Node Label (ID + Activation) *)
        let label = Printf.sprintf "%d\n%s" node.id (string_of_activation node.activation) in
        ignore (Py.Module.get_function_with_keywords ax_py "text"
          [| Py.Float.of_float x; Py.Float.of_float y; Py.String.of_string label |]
          [
            "ha", Py.String.of_string "center";
            "va", Py.String.of_string "center";
            "fontsize", Py.Int.of_int 8;
            "color", Py.String.of_string "black";
            "fontweight", Py.String.of_string "bold";
            "zorder", Py.Int.of_int 3
          ]
        )
      with Not_found -> ())
    genome.nodes;

  Fig.suptitle fig (Printf.sprintf "NEAT Genome (Fitness: %.4f)" genome.fitness);
  Mpl.show ()

