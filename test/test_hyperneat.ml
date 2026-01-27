open Neatml.Types
open Neatml.Phenotype
open Neatml.Evolution
open Neatml.Innovation

(* XOR training data *)
let xor_dataset = [|
  ([0.; 0.], [0.]);
  ([0.; 1.], [1.]);
  ([1.; 0.], [1.]);
  ([1.; 1.], [0.]);
|]

(* Fitness function for XOR *)
let xor_evaluator genome =
  let network = create_phenotype genome in
  let total_error = Array.fold_left
    (fun acc (inputs, expected) ->
      let output = predict network inputs in
      let error = abs_float (List.hd output -. List.hd expected) in
      acc +. error)
    0. xor_dataset
  in
  (* Convert error to fitness (lower error = higher fitness) *)
  4. -. total_error

(* Test NEAT mode *)
let test_neat () =
  Printf.printf "\n=== Testing NEAT (Classic Mode) ===\n";
  flush stdout;
  
  let innov = create 100 in
  let pop_size = 150 in
  let pop = create_pop pop_size 2 1 innov in
  let species = ref [] in
  let dynamic_threshold = ref 3.0 in
  
  for epoch = 1 to 50 do
    let new_pop, new_sp, _ = 
      generation pop !species xor_evaluator innov dynamic_threshold ()
    in
    species := new_sp;
    
    (* Find best genome *)
    let best = List.fold_left
      (fun acc g -> if g.fitness > acc.fitness then g else acc)
      (List.hd new_pop.genomes)
      new_pop.genomes
    in
    
    if epoch mod 10 = 0 then begin
      Printf.printf "Epoch %d: Best fitness = %.4f, Species = %d\n"
        epoch best.fitness (List.length !species);
      flush stdout
    end;
    
    (* Early stopping if solved *)
    if best.fitness >= 3.9 then begin
      Printf.printf "NEAT solved XOR at epoch %d!\n" epoch;
      flush stdout
    end
  done

(* Test HyperNEAT mode *)
let test_hyperneat () =
  Printf.printf "\n=== Testing HyperNEAT Mode ===\n";
  flush stdout;
  
  (* Define substrate configuration for XOR *)
  (* 2D coordinates: inputs at bottom, outputs at top *)
  let substrate_config = {
    input_coords = [[-1.; -1.]; [1.; -1.]]; (* 2 inputs *)
    hidden_coords = []; (* No hidden layer in substrate *)
    output_coords = [[0.; 1.]]; (* 1 output *)
  } in
  
  let hyperneat_config = {
    substrate = substrate_config;
    weight_threshold = 0.2;
    use_leo = true;
  } in
  
  let innov = create 100 in
  let pop_size = 150 in
  
  (* For HyperNEAT, we need CPPN networks *)
  (* CPPN inputs: x1, y1, x2, y2, distance = 5 inputs *)
  (* CPPN outputs: weight value = 1 output *)
  let pop = create_pop pop_size 5 1 innov in
  let species = ref [] in
  let dynamic_threshold = ref 3.0 in
  
  for epoch = 1 to 50 do
    let new_pop, new_sp, _ = 
      generation pop !species xor_evaluator innov dynamic_threshold ~hyperneat_config ()
    in
    species := new_sp;
    
    (* Find best CPPN genome *)
    let best = List.fold_left
      (fun acc g -> if g.fitness > acc.fitness then g else acc)
      (List.hd new_pop.genomes)
      new_pop.genomes
    in
    
    if epoch mod 10 = 0 then begin
      Printf.printf "Epoch %d: Best fitness = %.4f, Species = %d\n"
        epoch best.fitness (List.length !species);
      flush stdout
    end;
    
    (* Early stopping if solved *)
    if best.fitness >= 3.9 then begin
      Printf.printf "HyperNEAT solved XOR at epoch %d!\n" epoch;
      flush stdout
    end
  done

(* Main *)
let () =
  Random.self_init ();
  
  Printf.printf "XOR Problem - NEAT vs HyperNEAT Comparison\n";
  Printf.printf "==========================================\n";
  flush stdout;
  
  (* Test both modes *)
  test_neat ();
  test_hyperneat ();
  
  Printf.printf "\nTests completed!\n";
  flush stdout
