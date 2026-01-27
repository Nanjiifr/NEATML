# HyperNEAT dans NEATML

## Introduction

Cette implémentation ajoute le support de **HyperNEAT** (Hypercube-based NeuroEvolution of Augmenting Topologies) à NEATML. HyperNEAT est une extension de NEAT qui utilise un réseau CPPN (Compositional Pattern Producing Network) pour générer les poids d'un réseau substrat basé sur les positions géométriques des neurones.

## Différence entre NEAT et HyperNEAT

- **NEAT classique** : Évolue directement la topologie et les poids du réseau de neurones
- **HyperNEAT** : Évolue un CPPN qui génère les poids d'un réseau substrat à partir de coordonnées géométriques

## Types ajoutés

### Dans `types.ml` :

```ocaml
type substrate_node = {
  position : float list;    (* Coordonnées dans l'espace substrat *)
  layer : int;              (* 0=input, 1=hidden, 2=output *)
  node_id : int;
}

type substrate_config = {
  input_coords : float list list;   (* Coordonnées des neurones d'entrée *)
  hidden_coords : float list list;  (* Coordonnées des neurones cachés *)
  output_coords : float list list;  (* Coordonnées des neurones de sortie *)
}

type hyperneat_config = {
  substrate : substrate_config;
  weight_threshold : float;  (* Seuil minimum pour créer une connexion *)
  use_leo : bool;           (* Link Expression Output - exprime seulement les liens au-dessus du seuil *)
}
```

## Utilisation

### Mode NEAT classique (inchangé)

```ocaml
let innov = Innovation.create 100 in
let pop = Evolution.create_pop pop_size num_inputs num_outputs innov in
let species = ref [] in
let dynamic_threshold = ref 3.0 in

let new_pop, new_sp, _ = 
  Evolution.generation pop !species evaluator innov dynamic_threshold ()
in
```

### Mode HyperNEAT (nouveau)

```ocaml
(* 1. Définir la configuration du substrat *)
let substrate_config = {
  input_coords = [[-1.; -1.]; [1.; -1.]];  (* 2 neurones d'entrée *)
  hidden_coords = [];                       (* Pas de couche cachée *)
  output_coords = [[0.; 1.]];              (* 1 neurone de sortie *)
} in

(* 2. Créer la configuration HyperNEAT *)
let hyperneat_config = {
  substrate = substrate_config;
  weight_threshold = 0.2;  (* Connexions avec |poids| < 0.2 sont ignorées si use_leo=true *)
  use_leo = true;
} in

(* 3. Créer une population CPPN *)
let innov = Innovation.create 100 in
(* Pour HyperNEAT, le CPPN prend en entrée :
   - coordonnées source (x1, y1)
   - coordonnées cible (x2, y2)  
   - distance entre source et cible
   Total: 5 entrées pour des coordonnées 2D *)
let num_cppn_inputs = 5 in  (* 2*2 + 1 pour coordonnées 2D *)
let num_cppn_outputs = 1 in (* Valeur du poids *)
let pop = Evolution.create_pop pop_size num_cppn_inputs num_cppn_outputs innov in

(* 4. Évolution avec HyperNEAT *)
let species = ref [] in
let dynamic_threshold = ref 3.0 in

let new_pop, new_sp, _ = 
  Evolution.generation pop !species evaluator innov dynamic_threshold ~hyperneat_config ()
in
```

## Exemple complet

Voir le fichier `test/test_hyperneat.ml` pour un exemple complet qui compare NEAT et HyperNEAT sur le problème XOR.

Pour exécuter l'exemple :

```bash
dune build test/test_hyperneat.exe
_build/default/test/test_hyperneat.exe
```

## Notes importantes

1. **Fonction d'évaluation** : La fonction d'évaluation reçoit toujours un `genome` standard. En mode HyperNEAT, ce génome est le réseau substrat généré à partir du CPPN.

2. **Nombre d'entrées CPPN** : Pour un substrat avec des coordonnées de dimension `d`, le CPPN a besoin de `2*d + 1` entrées (coordonnées source + coordonnées cible + distance).

3. **Pas de changement aux fonctions existantes** : Les noms des fonctions existantes n'ont pas été modifiés. L'option HyperNEAT est ajoutée via un paramètre optionnel.

4. **Compatibilité** : Le code NEAT existant continue de fonctionner sans modification, il suffit d'ajouter `()` à la fin de l'appel à `generation`.

## Architecture technique

### Module `Hyperneat`

Le module `src/hyperneat.ml` fournit :

- `create_substrate_network` : Convertit un génome CPPN en réseau substrat
- `query_cppn` : Interroge le CPPN pour obtenir le poids d'une connexion
- `distance` : Calcule la distance euclidienne entre deux points

### Modification de `Evolution.generation`

La fonction `generation` accepte maintenant un paramètre optionnel `?hyperneat_config`. Quand ce paramètre est fourni, chaque génome CPPN est converti en réseau substrat avant l'évaluation.

```ocaml
let generation pop l_species evaluator innov_global dynamic_treshold ?hyperneat_config () =
  (* ... *)
  let eval_genome = 
    match hyperneat_config with
    | Some hn_config -> Hyperneat.create_substrate_network g hn_config
    | None -> g
  in
  let new_fitness = evaluator eval_genome in
  (* ... *)
```
