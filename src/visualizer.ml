open Graphics
open Types

let win_width = 1000 (* Largeur de la fenêtre *)
let net_height = 500 (* Hauteur réservée au réseau (en bas) *)
let game_height = 400 (* Hauteur réservée au jeu (en haut) *)
let win_height = net_height + game_height
let color_sensor = rgb 100 200 100
let color_output = rgb 200 100 100
let color_hidden = rgb 150 150 255
let node_radius = 12

let init_window () =
  open_graph (Printf.sprintf " %dx%d" win_width win_height);
  set_window_title "NEAT Genome Visualization";
  auto_synchronize false

let draw_arrow x1 y1 x2 y2 radius =
  let dx = float (x2 - x1) in
  let dy = float (y2 - y1) in
  let dist = sqrt ((dx *. dx) +. (dy *. dy)) in
  if dist > float radius then begin
    (* Point d'arrêt de la ligne (bord du cercle) *)
    let stop_x = float x2 -. (dx /. dist *. float radius) in
    let stop_y = float y2 -. (dy /. dist *. float radius) in

    (* Dessin de la ligne principale *)
    moveto x1 y1;
    lineto (int_of_float stop_x) (int_of_float stop_y);

    (* Calcul des pointes de la flèche *)
    let arrow_len = 15. in
    let angle = atan2 dy dx in
    let a1 = angle +. 3.14159 -. 0.5 in
    let a2 = angle +. 3.14159 +. 0.5 in

    moveto (int_of_float stop_x) (int_of_float stop_y);
    lineto
      (int_of_float (stop_x +. (cos a1 *. arrow_len)))
      (int_of_float (stop_y +. (sin a1 *. arrow_len)));
    moveto (int_of_float stop_x) (int_of_float stop_y);
    lineto
      (int_of_float (stop_x +. (cos a2 *. arrow_len)))
      (int_of_float (stop_y +. (sin a2 *. arrow_len)))
  end

let calculate_positions (nodes : node_gene list) =
  let positions = Hashtbl.create (List.length nodes) in
  let sensors = List.filter (fun (n : node_gene) -> n.kind = Sensor) nodes in
  let outputs = List.filter (fun (n : node_gene) -> n.kind = Output) nodes in
  let hidden = List.filter (fun (n : node_gene) -> n.kind = Hidden) nodes in

  let distribute_vertical x_pos node_list =
    let n = List.length node_list in
    let margin = 50 in
    let available_h = net_height - (2 * margin) in
    (* Si un seul noeud, on le met au milieu, sinon on espace *)
    let step = if n > 1 then available_h / (n - 1) else 0 in
    let start_y = margin in

    List.iteri
      (fun i node ->
        let y = if n = 1 then net_height / 2 else start_y + (i * step) in
        Hashtbl.add positions node.id (x_pos, y))
      node_list
  in

  distribute_vertical 100 sensors;
  (* Capteurs à gauche *)
  distribute_vertical (win_width - 100) outputs;

  (* Sorties à droite *)

  (* Les nœuds cachés sont placés aléatoirement au centre *)
  let prng = Random.State.make [| 42 |] in
  List.iter
    (fun node ->
      let min_x = 200 in
      let max_x = win_width - 200 in
      let min_y = 50 in
      let max_y = net_height - 50 in
      let x = min_x + Random.State.int prng (max_x - min_x) in
      let y = min_y + Random.State.int prng (max_y - min_y) in
      Hashtbl.add positions node.id (x, y))
    hidden;
  positions

let draw_genome genome =
  clear_graph ();

  let positions = calculate_positions genome.nodes in

  (* 1. Dessiner les connexions *)
  List.iter
    (fun conn ->
      try
        let x1, y1 = Hashtbl.find positions conn.in_node in
        let x2, y2 = Hashtbl.find positions conn.out_node in
        if conn.enabled then (
          (* Vert pour positif, Rouge pour négatif *)
          if conn.weight >= 0. then set_color (rgb 0 200 0)
          else set_color (rgb 200 0 0);

          (* Epaisseur selon la force du poids *)
          let thickness =
            int_of_float (min 5.0 (max 1.0 (abs_float conn.weight)))
          in
          set_line_width thickness;

          draw_arrow x1 y1 x2 y2 node_radius;

          let mid_x = (x1 + x2) / 2 in
          let mid_y = (y1 + y2) / 2 in
          let weight_str = Printf.sprintf "%.2f" conn.weight in
          let tw, th = text_size weight_str in

          set_color white;
          fill_rect (mid_x - (tw / 2)) (mid_y - (th / 2)) tw th;
          set_color black;
          moveto (mid_x - (tw / 2)) (mid_y - (th / 2));
          draw_string weight_str)
        else (
          (* Connexions désactivées en gris clair fin *)
          set_color (rgb 230 230 230);
          set_line_width 1;
          moveto x1 y1;
          lineto x2 y2)
      with Not_found -> ())
    genome.connections;

  (* 2. Dessiner les nœuds par dessus *)
  List.iter
    (fun node ->
      try
        let x, y = Hashtbl.find positions node.id in
        let c =
          match node.kind with
          | Sensor -> color_sensor
          | Output -> color_output
          | Hidden -> color_hidden
        in
        set_color c;
        fill_circle x y node_radius;
        set_color black;
        set_line_width 1;
        draw_circle x y node_radius;

        (* ID du nœud au centre *)
        let id_str = string_of_int node.id in
        let w_id, h_id = text_size id_str in
        moveto (x - (w_id / 2)) (y - (h_id / 2));
        draw_string id_str
      with Not_found -> ())
    genome.nodes;

  (* Info Fitness en bas *)
  set_color black;
  moveto 10 10;
  draw_string (Printf.sprintf "Network Fitness: %.4f" genome.fitness)
