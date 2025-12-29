open Graphics
open Neat
open Cartpole

(* --- Configuration de la Fenêtre --- *)
let win_width = 1000 (* Largeur de la fenêtre *)
let net_height = 500 (* Hauteur réservée au réseau (en bas) *)
let game_height = 400 (* Hauteur réservée au jeu (en haut) *)
let win_height = net_height + game_height

(* --- Couleurs et Styles --- *)
let color_sensor = rgb 100 200 100
let color_output = rgb 200 100 100
let color_hidden = rgb 150 150 255
let node_radius = 12

let init_window () =
  open_graph (Printf.sprintf " %dx%d" win_width win_height);
  set_window_title "NEAT Cartpole & Genome Visualization";
  auto_synchronize false

(* --- Partie 1 : Dessin du Jeu (En Haut) --- *)
let draw_cartpole state score =
  (* On dessine tout décalé de net_height vers le haut *)
  let base_y = net_height in

  (* Paramètres de dessin *)
  let scale_x = 100. in
  let cart_width = 50 in
  let cart_height = 30 in
  let pole_len = 100. in

  (* Le sol *)
  let ground_y = base_y + (game_height / 2) - 50 in
  set_color black;
  set_line_width 2;
  moveto 0 ground_y;
  lineto win_width ground_y;

  (* Position du chariot *)
  let center_x = win_width / 2 in
  let cart_draw_x = center_x + int_of_float (state.x *. scale_x) in
  let cart_draw_y = ground_y in

  (* Dessin Chariot *)
  set_color blue;
  fill_rect (cart_draw_x - (cart_width / 2)) cart_draw_y cart_width cart_height;

  (* Dessin Pôle *)
  let tip_x = cart_draw_x + int_of_float (pole_len *. sin state.theta) in
  let tip_y =
    cart_draw_y + cart_height + int_of_float (pole_len *. cos state.theta)
  in

  set_color red;
  set_line_width 6;
  moveto cart_draw_x (cart_draw_y + cart_height);
  lineto tip_x tip_y;

  (* Affichage du Score *)
  set_color black;
  moveto 10 (win_height - 30);
  set_text_size 20;
  (* Note: Graphics ignore souvent ça selon l'OS, mais on essaie *)
  draw_string (Printf.sprintf "Score: %d" score);

  (* Indicateur ECHEC *)
  if Cartpole.is_failed state then (
    set_color red;
    moveto (center_x - 40) (ground_y + 100);
    draw_string "--- FAILED ---")

(* --- Partie 2 : Dessin du Réseau (En Bas) --- *)
let calculate_positions (nodes : node_gene list) =
  let positions = Hashtbl.create (List.length nodes) in
  let sensors = List.filter (fun (n : node_gene) -> n.kind = Sensor) nodes in
  let outputs = List.filter (fun (n : node_gene) -> n.kind = Output) nodes in
  let hidden = List.filter (fun (n : node_gene) -> n.kind = Hidden) nodes in

  (* Fonction pour aligner verticalement les inputs et outputs *)
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
            int_of_float (min 6.0 (max 1.0 (abs_float conn.weight)))
          in
          set_line_width thickness;
          moveto x1 y1;
          lineto x2 y2)
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

(* --- Fonction Principale --- *)
let draw_all state score genome =
  clear_graph ();

  (* Ligne de séparation *)
  set_color (rgb 200 200 200);
  set_line_width 3;
  moveto 0 net_height;
  lineto win_width net_height;

  (* Dessin des deux parties *)
  draw_cartpole state score;
  draw_genome genome;

  synchronize ()

let draw_cart_simple state is_best =
  let base_y = net_height in
  let scale_x = 100. in
  let center_x = win_width / 2 in
  let ground_y = base_y + (game_height / 2) - 50 in

  let cart_draw_x = center_x + int_of_float (state.x *. scale_x) in

  (* Couleur : Rouge pour le leader, gris pour les autres *)
  if is_best then (
    set_color red;
    set_line_width 2)
  else (
    set_color (rgb 200 200 200);
    set_line_width 1);

  let cart_width = 50 in
  let cart_height = 30 in
  let pole_len = 100. in

  (* Chariot *)
  draw_rect (cart_draw_x - (cart_width / 2)) ground_y cart_width cart_height;

  (* Pôle *)
  let tip_x = cart_draw_x + int_of_float (pole_len *. sin state.theta) in
  let tip_y =
    ground_y + cart_height + int_of_float (pole_len *. cos state.theta)
  in

  moveto cart_draw_x (ground_y + cart_height);
  lineto tip_x tip_y

let draw_population states score best_genome =
  clear_graph ();

  (* 1. Dessiner le décor (une seule fois) *)
  let base_y = net_height in
  let ground_y = base_y + (game_height / 2) - 50 in
  set_color black;
  set_line_width 2;
  moveto 0 ground_y;
  lineto win_width ground_y;

  (* 2. Dessiner tous les agents encore en vie *)
  (* On dessine le premier de la liste (le meilleur) en dernier pour qu'il soit au dessus *)
  let rec draw_agents list idx =
    match list with
    | [] -> ()
    | state :: rest ->
        draw_agents rest (idx + 1);
        (* Appel récursif d'abord pour inverser l'ordre de dessin *)
        draw_cart_simple state (idx = 0)
    (* Le premier de la liste (idx 0) est le leader *)
  in
  draw_agents states 0;

  (* 3. Afficher le score *)
  set_color black;
  moveto 10 (win_height - 30);
  draw_string
    (Printf.sprintf "Live Agents: %d | Time: %d" (List.length states) score);

  (* 4. Dessiner le réseau du meilleur survivant (le premier de la liste) *)
  set_line_width 3;
  set_color (rgb 200 200 200);
  moveto 0 net_height;
  lineto win_width net_height;
  (* Séparateur *)
  draw_genome best_genome;

  synchronize ()
