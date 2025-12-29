# NEAT ML

Implementation OCaml de NEAT (NeuroEvolution of Augmenting Topologies) + quelques démos/tests.

## Structure

- `src/` : bibliothèque (NEAT, évolution, cartpole, visualisation)
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
