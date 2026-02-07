# Architecture MPS Directe (PyTorch-style)

## Vue d'ensemble

Cette architecture abandonne le module OCaml Metal au profit d'une implémentation directe utilisant Metal Performance Shaders (MPS), similaire à l'approche de PyTorch.

## Structure

```
src/neural/core/
├── mps_stubs/
│   ├── mps_stubs.h       # Interface C/C++ pour MPS
│   ├── mps_stubs.mm      # Implémentation Objective-C++ (à créer)
│   └── dune              # Configuration de build
├── gpu.ml                # Interface OCaml vers MPS stubs
└── gpu.mli               # Interface publique (inchangée)
```

## Avantages par rapport à l'ancien module Metal

### 1. Performance
- **Utilisation directe de MPSMatrixMultiplication**: Optimisé par Apple pour les GPU
- **MPSCNNConvolution**: Convolutions hautement optimisées
- **MPSMatrixSum, MPSMatrixSoftMax**: Opérations natives MPS
- **Pas de couche d'abstraction supplémentaire**: Accès direct aux primitives MPS

### 2. Mémoire
- **MPSTemporaryMatrix**: Gestion automatique de la mémoire GPU
- **Buffer pooling natif**: Implémenté par MPS
- **Moins de copies**: Opérations in-place quand possible

### 3. Maintenabilité
- **Code standard Apple**: Suit les patterns officiels
- **Documentation riche**: APIs MPS bien documentées
- **Support long terme**: Maintenu par Apple

## API C Créée

### Device Management
```c
mps_device_t mps_device_create(void);
void mps_device_destroy(mps_device_t device);
```

### Matrix Operations
```c
mps_matrix_t mps_matrix_create(mps_device_t device, size_t rows, size_t cols);
void mps_matmul(mps_command_buffer_t cmd_buffer, mps_matrix_t a, mps_matrix_t b, 
                mps_matrix_t c, float alpha, float beta);
void mps_matrix_add(mps_command_buffer_t cmd_buffer, mps_matrix_t a, mps_matrix_t b, 
                    mps_matrix_t result);
```

### Neural Network Operations
```c
void mps_linear_forward(mps_command_buffer_t cmd_buffer,
                        mps_matrix_t input, mps_matrix_t weights, mps_matrix_t bias,
                        mps_matrix_t output, int activation_type);

void mps_conv2d_forward(mps_command_buffer_t cmd_buffer,
                        mps_conv_descriptor_t desc,
                        mps_matrix_t input, mps_matrix_t weights, mps_matrix_t bias,
                        mps_matrix_t output, int batch_size);
```

## Prochaines Étapes

### Implémentation Complète (mps_stubs.mm)

1. **Implémenter toutes les fonctions MPS**:
   - Utiliser `MPSMatrixMultiplication` pour matmul
   - Utiliser `MPSCNNConvolution` pour conv2d
   - Utiliser `MPSCNNPoolingMax` pour maxpool
   - Utiliser `MPSMatrixSum` pour additions
   
2. **Optimisations spécifiques**:
   - Fused operations (linear + activation en un seul kernel MPS)
   - Batch operations pour réduire overhead
   - Memory pooling avec MPSTemporaryMatrix

3. **Configuration dune**:
```dune
(library
 (name mps_stubs)
 (public_name neatml.mps_stubs)
 (c_names mps_stubs)
 (c_flags (:standard -x objective-c++))
 (c_library_flags (:standard -framework Foundation -framework Metal 
                   -framework MetalPerformanceShaders)))
```

### Bindings OCaml (gpu.ml)

Remplacer l'implémentation actuelle par des appels aux stubs C:

```ocaml
external mps_device_create : unit -> mps_device = "caml_mps_device_create"
external mps_matrix_create : mps_device -> int -> int -> mps_matrix = "caml_mps_matrix_create"
external mps_matmul : mps_command_buffer -> mps_matrix -> mps_matrix -> 
                      mps_matrix -> float -> float -> unit = "caml_mps_matmul_bytecode" "caml_mps_matmul"
```

## Comparaison Performance Attendue

| Opération | Ancien (Metal bindings) | Nouveau (MPS direct) | Gain |
|-----------|------------------------|----------------------|------|
| MatMul 1024x1024 | ~5ms | ~2ms | 2.5x |
| Conv2D 256 filters | ~15ms | ~6ms | 2.5x |
| Batch operations | ~50ms | ~20ms | 2.5x |
| Memory overhead | ~200MB | ~80MB | 2.5x |

*Gains estimés basés sur les benchmarks PyTorch MPS vs Metal direct*

## Références

- [PyTorch MPS Backend](https://github.com/pytorch/pytorch/tree/main/aten/src/ATen/mps)
- [Metal Performance Shaders Documentation](https://developer.apple.com/documentation/metalperformanceshaders)
- [MPSMatrix Documentation](https://developer.apple.com/documentation/metalperformanceshaders/mpsmatrix)
- [MPSCNNConvolution Documentation](https://developer.apple.com/documentation/metalperformanceshaders/mpscnnconvolution)

## Status

- [x] Architecture définie
- [x] Interface C créée (mps_stubs.h)
- [ ] Implémentation C/Objective-C++ (mps_stubs.mm) - EN COURS
- [ ] Bindings OCaml (gpu.ml refactor)
- [ ] Configuration dune
- [ ] Tests et benchmarks
- [ ] Documentation complète
