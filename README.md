# NEAT ML

Implementation OCaml de NEAT (NeuroEvolution of Augmenting Topologies) + HyperNEAT + quelques démos/tests.

## Fonctionnalités

- **NEAT** : Évolution de topologies de réseaux de neurones
- **HyperNEAT** : Extension de NEAT utilisant des CPPNs pour générer des réseaux substrats (voir [HYPERNEAT.md](HYPERNEAT.md))
- Spéciation dynamique
- Évolution parallèle avec Parmap
- Visualisation

## Structure

- `src/` : bibliothèque (NEAT, HyperNEAT, évolution, visualisation)
- `bin/` : exécutables (ex: `test_cartpole`)
- `test/` : petits programmes de test/validation

## Build

```sh
dune build
```

## Lancer la démo cartpole

```sh
dune exec bin/test_cartpole.exe
```

## Tests

Les programmes dans `test/` sont des exécutables.

```sh
dune exec test/test_xor.exe
```

### Test HyperNEAT

Pour tester HyperNEAT sur le problème XOR :

```sh
dune build test/test_hyperneat.exe
_build/default/test/test_hyperneat.exe
```

## Documentation HyperNEAT

Voir [HYPERNEAT.md](HYPERNEAT.md) pour la documentation complète sur l'utilisation de HyperNEAT.
