open Neat
open Evolution
open Visualizer
open Graphics
open Snake

(* ========================================================================== *)
(* 1. CONFIGURATION ET NAVIGATION                                             *)
(* ========================================================================== *)

type relative_move = TurnLeft | GoStraight | TurnRight

(* Convertit les sorties du réseau en décision *)
let decide_move outputs =
  let max_val = List.fold_left max (-1.0) outputs in
  let rec find_idx idx list =
    match list with
    | [] -> 1
    | h :: t -> if h = max_val then idx else find_idx (idx + 1) t
  in
  match find_idx 0 outputs with
  | 0 -> TurnLeft
  | 2 -> TurnRight
  | _ -> GoStraight

(* Traduction Vue Relative (Serpent) -> Vue Absolue (Carte) *)
let rel_to_abs current_dir move =
  match (current_dir, move) with
  | _, GoStraight -> current_dir
  | Up, TurnLeft -> Left
  | Up, TurnRight -> Right
  | Down, TurnLeft -> Right
  | Down, TurnRight -> Left
  | Left, TurnLeft -> Down
  | Left, TurnRight -> Up
  | Right, TurnLeft -> Up
  | Right, TurnRight -> Down

let get_current_direction snake =
  match snake with
  | head :: neck :: _ ->
      let hx, hy = head and nx, ny = neck in
      if hx = nx + 1 then Up
      else if hx = nx - 1 then Down
      else if hy = ny + 1 then Right
      else Left
  | _ -> Right

let dist_manhattan (x1, y1) (x2, y2) = abs (x1 - x2) + abs (y1 - y2)

(* ========================================================================== *)
(* 2. FITNESS (SIMPLE & EFFICACE)                                             *)
(* ========================================================================== *)

let is_wall_death state game_size =
  match state.snake with
  | (hx, hy) :: _ -> hx < 0 || hx >= game_size || hy < 0 || hy >= game_size
  | _ -> false

let evaluate_genome g =
  let phenotype = create_phenotype g in
  let game_size = 20 in
  let state = ref (Snake.init_game game_size) in
  let hunger_limit = 100 in
  let fitness = ref 0.0 in

  while !state.alive do
    (* Inputs : 1.0 (Biais) + 9 Capteurs *)
    let inputs = 1.0 :: Snake.get_inputs !state in

    let outputs = predict phenotype inputs in
    let decision = decide_move outputs in
    let current_dir = get_current_direction !state.snake in
    let next_dir = rel_to_abs current_dir decision in

    let old_dist = dist_manhattan (List.hd !state.snake) !state.food in

    state := Snake.step !state next_dir;

    if !state.alive then begin
      if !state.steps >= hunger_limit then
        state := { !state with alive = false }
      else begin
        (* Incitation à se rapprocher *)
        let new_dist = dist_manhattan (List.hd !state.snake) !state.food in
        if new_dist < old_dist then fitness := !fitness +. 1.0
        else fitness := !fitness -. 1.5;

        (* Grosse récompense pour manger *)
        if !state.steps = 0 then fitness := !fitness +. 100.0
      end
    end
  done;

  (* Pénalité mort mur *)
  if is_wall_death !state game_size then fitness := !fitness *. 0.5;
  float !state.score

(* ========================================================================== *)
(* 3. AFFICHAGE GRAPHIQUE (VUE SÉPARÉE)                                       *)
(* ========================================================================== *)

(* Dessine la grille Snake dans la moitié DROITE de la fenêtre *)
let draw_snake_panel state =
  let window_w = size_x () in
  let window_h = size_y () in

  (* --- CONFIGURATION DE LA ZONE DROITE --- *)
  let panel_x = window_w / 2 in
  (* Commence au milieu *)
  let panel_y = 0 in
  let panel_w = window_w / 2 in
  let panel_h = window_h in

  (* 1. Fond Noir pour le Panel Droit *)
  set_color (rgb 20 20 20);
  fill_rect panel_x panel_y panel_w panel_h;

  (* --- CALCUL DE L'ECHELLE --- *)
  (* On veut faire rentrer une grille de taille 'state.size' (ex: 20) dans ce panel *)
  (* On laisse une marge de 40px autour *)
  let margin = 40 in
  let available_size = min (panel_w - (2 * margin)) (panel_h - (2 * margin)) in

  (* C'EST ICI QUE L'ECHELLE EST FIXÉE : *)
  (* Taille d'un bloc (serpent ou pomme) en pixels *)
  let cell_size = available_size / state.size in

  (* Taille réelle de la grille affichée *)
  let board_size_px = state.size * cell_size in

  (* Offsets pour centrer la grille dans le panel droit *)
  let offset_x = panel_x + ((panel_w - board_size_px) / 2) in
  let offset_y = panel_y + ((panel_h - board_size_px) / 2) in

  (* 2. Dessin du Plateau (Cadre Blanc) *)
  set_color white;
  set_line_width 2;
  draw_rect (offset_x - 2) (offset_y - 2) (board_size_px + 4) (board_size_px + 4);
  set_line_width 1;

  (* --- FONCTION DE DESSIN UNITAIRE --- *)
  (* Convertit les coord (Row, Col) du jeu en pixels (X, Y) écran *)
  (* Snake.ml : x = Row (Vertical), y = Col (Horizontal) *)
  (* Graphics : x = Horizontal, y = Vertical *)
  let draw_block (r, c) color =
    (* Vérification de sécurité pour ne pas dessiner hors map *)
    if r >= 0 && r < state.size && c >= 0 && c < state.size then begin
      set_color color;
      (* Mapping : Graphics.X = Offset + Col * Size *)
      let gx = offset_x + (c * cell_size) in
      (* Mapping : Graphics.Y = Offset + Row * Size *)
      let gy = offset_y + (r * cell_size) in

      (* On remplit le carré (avec -1 pour laisser un micro-espace entre les blocs) *)
      fill_rect gx gy (cell_size - 1) (cell_size - 1)
    end
  in

  (* 3. La Pomme (Rouge) *)
  draw_block state.food (rgb 255 50 50);

  (* 4. Le Serpent (Vert) *)
  List.iteri
    (fun i pos ->
      if i = 0 then draw_block pos (rgb 0 255 0) (* Tête : Vert Fluo *)
      else draw_block pos (rgb 0 160 0) (* Corps : Vert Sombre *))
    state.snake;

  (* 5. Informations Textuelles *)
  set_color white;
  moveto offset_x (offset_y + board_size_px + 10);
  let info =
    Printf.sprintf "Score: %d | Energie: %d" state.score (100 - state.steps)
  in
  draw_string info

(* ========================================================================== *)
(* 4. BOUCLE PRINCIPALE                                                       *)
(* ========================================================================== *)

let main () =
  Random.self_init ();

  (try open_graph " 1200x600" with _ -> ());
  set_window_title "NEAT Snake - Training View";
  auto_synchronize false;

  let pop_size = 200 in
  let number_inputs = 9 in
  let number_outputs = 3 in

  let innov_global =
    InnovationManager.create (number_inputs + number_outputs + 5)
  in
  let pop =
    ref (create_pop pop_size number_inputs number_outputs innov_global)
  in
  let l_species = ref [] in

  let epoch = ref 0 in
  let best_score_global = ref 0 in

  while true do
    let new_pop, new_sp =
      generation !pop !l_species evaluate_genome innov_global
    in
    pop := new_pop;
    l_species := new_sp;

    let best_genome =
      List.fold_left
        (fun acc g -> if g.fitness > acc.fitness then g else acc)
        (List.hd !pop.genomes) !pop.genomes
    in
    print_pop_summary !pop !l_species !epoch;

    let phenotype = create_phenotype best_genome in
    let game_size = 10 in
    let state = ref (Snake.init_game game_size) in

    flush stdout;
    if !epoch mod 10 = 0 then
      while !state.alive do
        clear_graph ();

        draw_genome best_genome;

        draw_snake_panel !state;

        moveto ((size_x () / 2) - 40) 20;
        set_color (rgb 200 200 200);
        draw_string
          ("Gen: " ^ string_of_int !epoch ^ " | Record: "
          ^ string_of_int !best_score_global);

        synchronize ();

        let inputs = 1.0 :: Snake.get_inputs !state in
        let outputs = predict phenotype inputs in
        let decision = decide_move outputs in
        let current_dir = get_current_direction !state.snake in
        let next_dir = rel_to_abs current_dir decision in

        state := Snake.step !state next_dir;

        Unix.sleepf 0.005
      done;

    if !state.score > !best_score_global then best_score_global := !state.score;

    incr epoch
  done

let () = main ()
