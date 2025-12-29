open Graphics
open Cartpole

let width = 800
let height = 400
let scale_x = 100. (* 1 mètre = 100 pixels *)
let cart_width = 50
let cart_height = 30
let pole_len = 100. (* Longueur visuelle du pôle en pixels *)

(* Initialise la fenêtre graphique *)
let init_window () =
  open_graph (Printf.sprintf " %dx%d" width height);
  set_window_title "NEAT Cartpole Visualization";
  auto_synchronize false (* Pour éviter le scintillement (double buffering) *)

(* Dessine l'état actuel *)
let draw_state state score =
  clear_graph ();

  (* 1. Dessiner le sol *)
  set_color black;
  moveto 0 (height / 2);
  lineto width (height / 2);

  (* Calcul des coordonnées écran *)
  (* Le centre de l'écran est à (width/2, height/2) *)
  let center_x = width / 2 in
  let center_y = height / 2 in

  (* Position du chariot (state.x est en mètres) *)
  let cart_screen_x = center_x + int_of_float (state.x *. scale_x) in
  let cart_screen_y = center_y in

  (* 2. Dessiner le chariot *)
  set_color blue;
  fill_rect
    (cart_screen_x - (cart_width / 2))
    (cart_screen_y - (cart_height / 2))
    cart_width cart_height;

  (* 3. Dessiner le pôle *)
  (* Le pôle tourne autour du centre du chariot *)
  (* Attention : en math, 0 est à droite, ici 0 est en haut pour le pendule inversé généralement, 
     mais votre simulation utilise probablement 0 = vertical vers le haut *)
  let tip_x = cart_screen_x + int_of_float (pole_len *. sin state.theta) in
  let tip_y = cart_screen_y + int_of_float (pole_len *. cos state.theta) in

  set_color red;
  set_line_width 3;
  moveto cart_screen_x cart_screen_y;
  lineto tip_x tip_y;
  set_line_width 1;

  (* 4. Afficher le score *)
  set_color black;
  moveto 10 (height - 20);
  draw_string (Printf.sprintf "Score: %d" score);

  (* Si échec, afficher un message *)
  if Cartpole.is_failed state then begin
    set_color red;
    moveto (center_x - 30) (center_y + 100);
    draw_string "FAILED"
  end;

  synchronize () (* Affiche tout d'un coup *)
