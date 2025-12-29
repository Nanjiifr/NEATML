open Types
open Phenotype

let compatibility_distance g1 g2 =
  let c1 = 1. and c2 = 1. and c3 = 0.4 in
  let n =
    float (max (List.length g1.connections) (List.length g2.connections))
  in
  let rec aux gene1 gene2 =
    match (gene1, gene2) with
    | [], [] -> 0.
    | _ :: t, [] -> (c1 /. n) +. aux t []
    | [], _ :: t -> (c1 /. n) +. aux [] t
    | h1 :: t1, h2 :: t2 ->
        if h1.innov = h2.innov then
          (c3 *. abs_float (h1.weight -. h2.weight)) +. aux t1 t2
        else if h1.innov < h2.innov then (c2 /. n) +. aux t1 gene2
        else (c2 /. n) +. aux gene1 t2
  in
  aux g1.connections g2.connections

let clear_species_members l_species =
  List.map
    (fun s ->
      {
        sp_id = s.sp_id;
        members = [];
        spawn_amount = s.spawn_amount;
        repr = s.repr;
        best_fitness = s.best_fitness;
        stagn_count = s.stagn_count;
      })
    l_species

let speciate_gemome l_species g =
  let treshold = 3.0 in
  let has_fit = ref false in
  let new_species =
    List.map
      (fun s ->
        let delta = compatibility_distance s.repr g in
        if delta < treshold && not !has_fit then begin
          has_fit := true;
          {
            sp_id = s.sp_id;
            members = g :: s.members;
            spawn_amount = s.spawn_amount;
            repr = s.repr;
            best_fitness = s.best_fitness;
            stagn_count = s.stagn_count;
          }
        end
        else s)
      l_species
  in
  if not !has_fit then
    let max_id =
      List.fold_left (fun max_id s -> max s.sp_id max_id) 0 l_species
    in
    {
      sp_id = max_id + 1;
      repr = g;
      members = [ g ];
      best_fitness = g.fitness;
      stagn_count = 0;
      spawn_amount = 0;
    }
    :: l_species
  else new_species

let get_fitness g dataset f =
  let n = float (Array.length dataset) in
  let phenotype = create_phenotype g in
  1. /. n
  *. Array.fold_left
       (fun acc (i, o) ->
         let fitness = f (predict phenotype i) o in
         fitness +. acc)
       0. dataset
