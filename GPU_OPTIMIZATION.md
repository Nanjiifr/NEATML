# GPU Optimization - Memory Management Improvements

## Problem Summary
The original GPU implementation had several critical issues causing crashes and memory problems on Mac:

1. **Memory Leaks**: Temporary buffers for int/float parameters were created but never freed
2. **Command Buffer Overflow**: No limit on command buffer accumulation leading to memory exhaustion
3. **Missing Shaders**: References to undefined shader functions (`mat_transpose`, `add_bias`)
4. **No Resource Cleanup**: No mechanism to properly release GPU resources

## Implemented Solutions

### 1. Buffer Pooling System
**Problem**: Every GPU operation created temporary buffers for parameters (int/float) that were never reused or freed.

**Solution**: Implemented a pooling system that reuses buffers:
- `int_buffer_pool`: Queue of reusable integer buffers (max 50)
- `float_buffer_pool`: Queue of reusable float buffers (max 50)
- Buffers are retrieved from pool when needed and returned after use
- Pool size is capped to prevent unbounded memory growth

### 2. Automatic Command Buffer Flushing
**Problem**: Commands accumulated in a single command buffer indefinitely, causing memory pressure.

**Solution**: 
- Track number of commands in buffer with `commands_in_buffer` counter
- Automatically commit and sync when reaching 100 commands (configurable via `max_commands_per_buffer`)
- Reset counter after each commit
- This prevents excessive memory buildup while maintaining good batching performance

### 3. Missing Metal Shaders
**Problem**: Code referenced `mat_transpose` and `add_bias` kernels that didn't exist.

**Solution**: Added complete Metal shader implementations:
```metal
kernel void mat_transpose(
    device const float *A [[buffer(0)]],
    device float *B [[buffer(1)]],
    constant int &rows [[buffer(2)]],
    constant int &cols [[buffer(3)]],
    uint2 id [[thread_position_in_grid]])
{
    if (id.x >= uint(cols) || id.y >= uint(rows)) return;
    B[id.x * rows + id.y] = A[id.y * cols + id.x];
}

kernel void add_bias(
    device float *X [[buffer(0)]],
    device const float *bias [[buffer(1)]],
    constant int &cols [[buffer(2)]],
    uint2 id [[thread_position_in_grid]])
{
    if (id.x >= uint(cols)) return;
    X[id.y * cols + id.x] += bias[id.x];
}
```

### 4. Resource Cleanup
**Problem**: No way to properly clean up GPU resources when done.

**Solution**: Added `cleanup()` function that:
- Syncs any pending commands
- Clears all buffer pools
- Resets command counter
- Should be called when GPU operations are complete

### 5. Command Counting
**Problem**: No visibility into command buffer state.

**Solution**: Added `increment_command_count()` calls to all GPU operations:
- `linear_fwd`, `linear_bwd`, `adam_step`, `mse_grad`
- `matmul`, `add`, `relu`, `sigmoid`, `tanh`
- `transpose`, `add_bias`, `copy_inplace`

## Performance Characteristics

### Memory Usage
- **Before**: Unbounded growth with each operation creating new temporary buffers
- **After**: Stable memory usage with buffer pooling (capped at ~50 buffers per type)

### Command Batching
- Operations are batched up to 100 commands before auto-flush
- This provides good performance while preventing memory overflow
- Manual `commit_batch()` still available for fine-grained control

### Resource Safety
- Automatic cleanup on buffer overflow prevents crashes
- Pool size limits prevent unbounded memory growth
- Proper sync in cleanup ensures no pending operations are lost

## Usage Recommendations

### Basic Usage
```ocaml
(* Enable GPU *)
Utils.enable_gpu ();

(* ... perform GPU operations ... *)

(* Cleanup when done (optional but recommended) *)
Gpu.cleanup ();
```

### For Training
The existing training loop already has proper sync points:
```ocaml
for epoch = 1 to epochs do
  for batch = 0 to n_batches - 1 do
    (* ... forward/backward pass ... *)
    Gpu.commit_batch ()  (* Batch commit after each iteration *)
  done;
  Gpu.sync ()  (* Full sync after each epoch *)
done
```

### Memory-Constrained Scenarios
For very large models or limited memory:
- Reduce `max_commands_per_buffer` (default: 100) for more frequent flushing
- Call `Gpu.sync()` more frequently to ensure commands complete
- Use `Gpu.cleanup()` between major operations to clear pools

## Configuration Constants

Located in `gpu.ml`:
- `max_commands_per_buffer = 100`: Commands before auto-flush
- `max_pool_size = 50`: Maximum buffers in each pool

These can be adjusted based on workload and available memory.

## Testing Notes

The implementation should be tested with:
1. Simple XOR test (`test/xor_nn.ml`)
2. Larger sine wave test (`test/sine.ml`)
3. Extended training runs to verify no memory leaks
4. Multiple sequential runs to verify cleanup works

## Technical Details

### Buffer Lifecycle
1. **Allocation**: Check pool, create new if empty
2. **Use**: Set parameter value and bind to encoder
3. **Return**: After encoding complete, return to pool
4. **Reuse**: Next operation retrieves from pool instead of allocating

### Command Buffer Lifecycle
1. **Creation**: On first operation or after commit
2. **Accumulation**: Commands added, counter incremented
3. **Auto-flush**: At 100 commands, commit and sync
4. **Manual flush**: `commit_batch()` or `sync()` can be called anytime

### Memory Safety
- Buffer pools are bounded (max 50 each)
- Command buffers auto-flush (max 100 commands)
- All temporary buffers are returned to pool
- Cleanup function clears all pools

## Conclusion

These optimizations address the core memory management issues in the GPU implementation:
- ✅ Prevents memory leaks through buffer pooling
- ✅ Prevents memory overflow through command batching
- ✅ Adds missing shader implementations
- ✅ Provides proper resource cleanup
- ✅ Maintains good performance through batching

The implementation should now be stable on Mac and other platforms without crashes or memory issues.
