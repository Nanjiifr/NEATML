open Types

let mutate_weights g =
  let conn =
    List.map
      (fun (c : connection_gene) ->
        let seed = Random.int 100 in
        if seed < 80 then
          let curr_weight = c.weight in
          let modify = Random.int 100 < 95 in
          if modify then
            let power = 0.5 in
            let new_weight =
              curr_weight +. (Random.float (2. *. power) -. power)
            in
            let new_weight = max (-5.) (min 5. new_weight) in
            {
              in_node = c.in_node;
              out_node = c.out_node;
              weight = new_weight;
              enabled = c.enabled;
              innov = c.innov;
            }
          else
            let new_weight = Random.float 4. -. 2. in
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
  {
    connections = conn;
    nodes = g.nodes;
    fitness = g.fitness;
    adj_fitness = g.adj_fitness;
  }

let mutate_topology g mod_type innov_global =
  let nodes_array = Array.of_list g.nodes in
  match mod_type with
  | Connexion -> begin
      let max_attempts = 20 in
      let attempts = ref 0 in
      let found = ref false in

      let best_start = ref 0 in
      let best_end = ref 0 in

      while (not !found) && !attempts < max_attempts do
        incr attempts;

        let s = ref (Random.int (Array.length nodes_array)) in
        while nodes_array.(!s).kind = Output do
          s := Random.int (Array.length nodes_array)
        done;

        let e = ref (Random.int (Array.length nodes_array)) in
        while nodes_array.(!e).kind = Sensor do
          e := Random.int (Array.length nodes_array)
        done;

        let exists =
          List.exists
            (fun c ->
              c.in_node = nodes_array.(!s).id
              && c.out_node = nodes_array.(!e).id)
            g.connections
        in

        if not exists then begin
          found := true;
          best_start := !s;
          best_end := !e
        end
      done;

      if !found then begin
        let weight = Random.float 4. -. 2. in
        let innov_id =
          Innovation.get_innov_id innov_global nodes_array.(!best_start).id
            nodes_array.(!best_end).id Connexion
        in
        let new_connection =
          {
            in_node = nodes_array.(!best_start).id;
            out_node = nodes_array.(!best_end).id;
            weight;
            enabled = true;
            innov = innov_id;
          }
        in
        {
          connections = g.connections @ [ new_connection ];
          nodes = g.nodes;
          fitness = g.fitness;
          adj_fitness = g.adj_fitness;
        }
      end
      else g
    end
  | Node ->
      let enabled_conns = List.filter (fun c -> c.enabled) g.connections in
      if enabled_conns = [] then g
      else
        let target_conn =
          List.nth enabled_conns (Random.int (List.length enabled_conns))
        in

        let new_connections_list =
          List.map
            (fun c -> if c = target_conn then { c with enabled = false } else c)
            g.connections
        in

        let new_id =
          Innovation.get_innov_id innov_global target_conn.in_node
            target_conn.out_node Node
        in
        let innov_id_in =
          Innovation.get_innov_id innov_global target_conn.in_node new_id
            Connexion
        in
        let innov_id_out =
          Innovation.get_innov_id innov_global new_id target_conn.out_node
            Connexion
        in

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
          nodes = g.nodes @ [ new_node ];
          fitness = g.fitness;
          adj_fitness = g.adj_fitness;
        }

let mutate g innov_global =
  let new_genome = mutate_weights g in
  let modified_gemome =
    if Random.int 100 < 3 then mutate_topology new_genome Connexion innov_global
    else if Random.int 100 < 1 then mutate_topology new_genome Node innov_global
    else new_genome
  in
  modified_gemome
