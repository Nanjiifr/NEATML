open Neat
open Graphics

(* Dimensions de la fenêtre *)
let width = 800
let height = 600
let node_radius = 10

(* Couleurs *)
let color_sensor = rgb 100 200 100 (* Vert clair *)
let color_output = rgb 200 100 100 (* Rouge clair *)
let color_hidden = rgb 150 150 255 (* Bleu clair *)
let color_bg = white

(* --- CALCUL DES POSITIONS --- *)
(* Retourne une Hashtbl : Node_ID -> (x, y) *)
let calculate_positions nodes =
  let positions = Hashtbl.create (List.length nodes) in

  (* Séparer les types de noeuds *)
  let sensors = List.filter (fun n -> n.kind = Sensor) nodes in
  let outputs = List.filter (fun n -> n.kind = Output) nodes in
  let hidden = List.filter (fun n -> n.kind = Hidden) nodes in

  (* Fonction pour distribuer verticalement une liste de noeuds à une position X donnée *)
  let distribute_vertical x_pos node_list =
    let n = List.length node_list in
    let step = height / (n + 1) in
    List.iteri
      (fun i node -> Hashtbl.add positions node.id (x_pos, (i + 1) * step))
      node_list
  in

  (* 1. Placer les Sensors à Gauche (10% de la largeur) *)
  distribute_vertical (width / 10) sensors;

  (* 2. Placer les Outputs à Droite (90% de la largeur) *)
  distribute_vertical (width - (width / 10)) outputs;

  (* 3. Placer les Hidden au milieu (Aléatoire ou Grille) *)
  (* Pour un rendu plus stable, on pourrait utiliser l'innovation ID, 
       mais le random est acceptable pour commencer *)
  Random.init 42;
  (* Seed fixe pour que le dessin ne bouge pas à chaque refresh *)
  List.iter
    (fun node ->
      let min_x = width / 5 in
      let max_x = 4 * width / 5 in
      let min_y = 50 in
      let max_y = height - 50 in
      let x = min_x + Random.int (max_x - min_x) in
      let y = min_y + Random.int (max_y - min_y) in
      Hashtbl.add positions node.id (x, y))
    hidden;

  positions

(* --- DESSIN --- *)
let draw_genome genome =
  (* Initialisation graphiques *)
  open_graph (" " ^ string_of_int width ^ "x" ^ string_of_int height);
  set_window_title "NEAT Visualizer";
  set_color color_bg;
  fill_rect 0 0 width height;

  let positions = calculate_positions genome.nodes in

  (* 1. Dessiner les connexions *)
  List.iter
    (fun conn ->
      try
        let x1, y1 = Hashtbl.find positions conn.in_node in
        let x2, y2 = Hashtbl.find positions conn.out_node in

        if not conn.enabled then (
          (* Connexion désactivée en Gris *)
          set_color (rgb 200 200 200);
          set_line_width 1;
          moveto x1 y1;
          lineto x2 y2)
        else (
          (* Connexion active : Couleur selon le signe *)
          if conn.weight >= 0. then set_color (rgb 0 180 0) (* Vert foncé *)
          else set_color (rgb 180 0 0);

          (* Rouge foncé *)

          (* Épaisseur selon la force du poids (clampé entre 1 et 5 pixels) *)
          let thickness =
            int_of_float (min 5.0 (max 1.0 (abs_float conn.weight)))
          in
          set_line_width thickness;
          moveto x1 y1;
          lineto x2 y2)
      with Not_found ->
        () (* Si un noeud manque (ne devrait pas arriver), on ignore *))
    genome.connections;

  (* 2. Dessiner les noeuds *)
  List.iter
    (fun node ->
      try
        let x, y = Hashtbl.find positions node.id in

        (* Choix de la couleur *)
        let c =
          match node.kind with
          | Sensor -> color_sensor
          | Output -> color_output
          | Hidden -> color_hidden
        in

        (* Cercle rempli *)
        set_color c;
        fill_circle x y node_radius;

        (* Contour noir *)
        set_color black;
        set_line_width 1;
        draw_circle x y node_radius;

        (* ID du noeud (pour debug) *)
        moveto (x - 5) (y - 5);
        set_color black;
        draw_string (string_of_int node.id)
      with Not_found -> ())
    genome.nodes;

  (* Attendre une touche pour fermer *)
  Printf.printf
    "Appuyez sur une touche dans la fenetre graphique pour continuer...\n";
  flush stdout;
  ignore (read_key ());
  close_graph ()
