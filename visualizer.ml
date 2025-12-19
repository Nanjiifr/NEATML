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
let calculate_positions (nodes : node_gene list) =
  let positions = Hashtbl.create (List.length nodes) in

  (* Séparer les types de noeuds *)
  let (sensors : node_gene list) =
    List.filter (fun (n : node_gene) -> n.kind = Sensor) nodes
  in
  let outputs = List.filter (fun (n : node_gene) -> n.kind = Output) nodes in
  let hidden = List.filter (fun (n : node_gene) -> n.kind = Hidden) nodes in

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
  clear_graph ();
  let positions = calculate_positions genome.nodes in

  (* --- 1. DESSINER LES CONNEXIONS ET LES POIDS --- *)
  List.iter
    (fun conn ->
      try
        let x1, y1 = Hashtbl.find positions conn.in_node in
        let x2, y2 = Hashtbl.find positions conn.out_node in

        if not conn.enabled then (
          (* Connexion désactivée en Gris *)
          set_color (rgb 220 220 220);
          set_line_width 1;
          moveto x1 y1;
          lineto x2 y2)
        else (
          (* Couleur selon le signe *)
          if conn.weight >= 0. then set_color (rgb 0 180 0) (* Vert *)
          else set_color (rgb 180 0 0);

          (* Rouge *)

          (* Épaisseur *)
          let thickness =
            int_of_float (min 5.0 (max 1.0 (abs_float conn.weight)))
          in
          set_line_width thickness;
          moveto x1 y1;
          lineto x2 y2;

          (* Affichage du poids (avec fond blanc) *)
          let mid_x = (x1 + x2) / 2 in
          let mid_y = (y1 + y2) / 2 in
          let weight_str = Printf.sprintf "%.2f" conn.weight in
          let w_text, h_text = text_size weight_str in

          set_color white;
          fill_rect (mid_x - (w_text / 2)) (mid_y - (h_text / 2)) w_text h_text;

          set_color black;
          moveto (mid_x - (w_text / 2)) (mid_y - (h_text / 2));
          draw_string weight_str)
      with Not_found -> ())
    genome.connections;

  (* --- 2. DESSINER LES NOEUDS --- *)
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

        (* ID du noeud *)
        let id_str = string_of_int node.id in
        let w_id, h_id = text_size id_str in
        moveto (x - (w_id / 2)) (y - (h_id / 2));
        set_color black;
        draw_string id_str
      with Not_found -> ())
    genome.nodes;

  (* --- 3. NOUVEAU : DESSINER LE FITNESS EN HAUT --- *)
  set_color black;
  (* On formate le fitness avec 4 décimales pour voir la précision *)
  let fit_str = Printf.sprintf "Fitness: %.4f" genome.fitness in

  (* On calcule la taille du texte pour bien le centrer *)
  let w_fit, _ = text_size fit_str in

  (* Position : Centré horizontalement (width/2), en haut (height - 20) *)
  moveto ((width / 2) - (w_fit / 2)) (height - 30);

  (* On peut grossir un peu le texte artificiellement ou juste l'afficher tel quel *)
  draw_string fit_str;

  synchronize ()
