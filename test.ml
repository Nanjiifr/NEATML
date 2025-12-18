(* test.ml *)
(* Assure-toi d'avoir compilé main.ml avant ou d'ouvrir le module si c'est un projet dune *)
open Neat

let run_test () =
  Printf.printf "=== DÉBUT DU TEST NEAT (Version Utilisateur) ===\n";

  (* 1. SETUP *)
  let innov_mgr = InnovationManager.create () in

  (* On crée un génome simple valide : 1 Sensor -> 2 Output *)
  let initial_genome =
    {
      nodes = [ { id = 1; kind = Sensor }; { id = 2; kind = Output } ];
      connections =
        [
          { in_node = 1; out_node = 2; weight = 0.5; enabled = true; innov = 1 };
        ];
      fitness = 0.0;
    }
  in

  (* ================================================================= *)
  (* TEST 1 : AJOUT DE NOEUD (ADD NODE)                                *)
  (* ================================================================= *)
  Printf.printf "\n[TEST 1] Mutation Add Node (Couper le lien 1->2)...\n";

  (* --- Simulation Génome A --- *)
  (* On fixe la graine aléatoire pour forcer le choix du lien *)
  Random.init 42;
  let genome_a = mutate_topology initial_genome Node innov_mgr in

  (* --- Simulation Génome B (Clone) --- *)
  (* On remet la MÊME graine : l'algo doit faire exactement les mêmes choix *)
  Random.init 42;
  let genome_b = mutate_topology initial_genome Node innov_mgr in

  Visualizer.draw_genome genome_a;

  (* Vérifications *)
  (* 1. Vérifier que les nouveaux nœuds ont le même ID *)
  (* Le dernier nœud ajouté est à la fin de la liste *)
  let node_a = List.hd (List.rev genome_a.nodes) in
  let node_b = List.hd (List.rev genome_b.nodes) in

  Printf.printf "  > ID Noeud A : %d\n" node_a.id;
  Printf.printf "  > ID Noeud B : %d\n" node_b.id;

  if node_a.id = node_b.id then
    Printf.printf "  [SUCCES] Les IDs de noeuds sont synchronisés.\n"
  else Printf.printf "  [ECHEC] IDs différents (%d vs %d)\n" node_a.id node_b.id;
  assert (node_a.id = node_b.id);

  (* 2. Vérifier les innovations des nouvelles connexions *)
  (* Les 2 dernières connexions sont les nouvelles *)
  let conns_a = List.rev genome_a.connections in
  let conn_out_a = List.hd conns_a in
  (* Lien vers Output *)
  let conn_in_a = List.nth conns_a 1 in
  (* Lien depuis Input *)

  let conns_b = List.rev genome_b.connections in
  let conn_out_b = List.hd conns_b in
  let conn_in_b = List.nth conns_b 1 in

  Printf.printf "  > Innov Lien 1 A: %d | B: %d\n" conn_in_a.innov
    conn_in_b.innov;
  Printf.printf "  > Innov Lien 2 A: %d | B: %d\n" conn_out_a.innov
    conn_out_b.innov;

  assert (conn_in_a.innov = conn_in_b.innov);
  assert (conn_out_a.innov = conn_out_b.innov);

  (* 3. Vérifier que l'ancienne connexion est bien désactivée (Test du bug A) *)
  (* L'ancienne connexion est celle qui a l'innov 1 *)
  let old_conn_a = List.find (fun c -> c.innov = 1) genome_a.connections in
  if not old_conn_a.enabled then
    Printf.printf "  [SUCCES] L'ancienne connexion est bien désactivée.\n"
  else
    Printf.printf
      "  [ECHEC] L'ancienne connexion est toujours active ! (Bug liste immuable)\n";

  (* ================================================================= *)
  (* TEST 2 : AJOUT DE CONNEXION (ADD LINK)                            *)
  (* ================================================================= *)
  (* Pour ce test, on a besoin d'un génome avec 3 nœuds pour avoir une possibilité de lien *)
  (* On repart du genome_a qui a maintenant 3 noeuds (1, 2, et le nouveau) *)
  Printf.printf "\n[TEST 2] Mutation Add Link...\n";

  (* Genome A fait une nouvelle connexion *)
  Random.init 123;
  let genome_a_linked = mutate_topology genome_a Connexion innov_mgr in

  (* Genome B fait la MEME connexion *)
  Random.init 123;
  let genome_b_linked = mutate_topology genome_a Connexion innov_mgr in
  (* Note: on repart de genome_a pour simuler le même état *)

  let new_link_a = List.hd (List.rev genome_a_linked.connections) in
  let new_link_b = List.hd (List.rev genome_b_linked.connections) in

  Printf.printf "  > Nouvelle Connexion A : %d -> %d (Innov %d)\n"
    new_link_a.in_node new_link_a.out_node new_link_a.innov;
  Printf.printf "  > Nouvelle Connexion B : %d -> %d (Innov %d)\n"
    new_link_b.in_node new_link_b.out_node new_link_b.innov;

  if new_link_a.innov = new_link_b.innov then
    Printf.printf
      "  [SUCCES] Les innovations de connexion sont synchronisées.\n"
  else Printf.printf "  [ECHEC] Innovations différentes !\n";

  assert (new_link_a.innov = new_link_b.innov);

  Printf.printf "\n=== TOUS LES TESTS PASSÉS AVEC SUCCÈS ===\n"

let () = run_test ()
