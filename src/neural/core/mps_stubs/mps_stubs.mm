/*
 * MPS (Metal Performance Shaders) C Stubs Implementation
 * PyTorch-inspired direct MPS implementation for NEATML
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include "mps_stubs.h"
#include <vector>
#include <memory>

// ============================================================================
// C++ Wrapper Classes
// ============================================================================

struct MPSDevice {
    id<MTLDevice> device;
    id<MTLCommandQueue> commandQueue;
    
    MPSDevice() {
        device = MTLCreateSystemDefaultDevice();
        if (!device) {
            NSLog(@"Failed to create Metal device");
            return;
        }
        commandQueue = [device newCommandQueue];
    }
    
    ~MPSDevice() {
        commandQueue = nil;
        device = nil;
    }
};

struct MPSCommandBuffer {
    id<MTLCommandBuffer> buffer;
    MPSDevice* device;
    
    MPSCommandBuffer(MPSDevice* dev) : device(dev) {
        buffer = [dev->commandQueue commandBuffer];
    }
    
    ~MPSCommandBuffer() {
        buffer = nil;
    }
};

struct MPSMatrix {
    MPSMatrixDescriptor* descriptor;
    id<MTLBuffer> buffer;
    size_t rows;
    size_t cols;
    MPSDevice* device;
    
    MPSMatrix(MPSDevice* dev, size_t r, size_t c) 
        : rows(r), cols(c), device(dev) {
        descriptor = [MPSMatrixDescriptor 
            matrixDescriptorWithRows:rows 
            columns:cols 
            rowBytes:cols * sizeof(float) 
            dataType:MPSDataTypeFloat32];
        
        size_t bufferSize = rows * cols * sizeof(float);
        buffer = [dev->device newBufferWithLength:bufferSize 
                                          options:MTLResourceStorageModeShared];
    }
    
    ~MPSMatrix() {
        buffer = nil;
        descriptor = nil;
    }
    
    MPSMatrix* createMPSMatrix() {
        return [[MPSMatrix alloc] initWithBuffer:buffer 
                                      descriptor:descriptor];
    }
};

struct MPSConvDescriptor {
    int kernel_h, kernel_w;
    int input_channels, output_channels;
    int stride_h, stride_w;
    
    MPSConvDescriptor(int kh, int kw, int ic, int oc, int sh, int sw)
        : kernel_h(kh), kernel_w(kw), input_channels(ic), 
          output_channels(oc), stride_h(sh), stride_w(sw) {}
};

struct MPSPoolDescriptor {
    int kernel_h, kernel_w;
    int stride_h, stride_w;
    
    MPSPoolDescriptor(int kh, int kw, int sh, int sw)
        : kernel_h(kh), kernel_w(kw), stride_h(sh), stride_w(sw) {}
};

// ============================================================================
// Device Management
// ============================================================================

mps_device_t mps_device_create(void) {
    try {
        MPSDevice* device = new MPSDevice();
        if (!device->device) {
            delete device;
            return nullptr;
        }
        return static_cast<mps_device_t>(device);
    } catch (...) {
        return nullptr;
    }
}

void mps_device_destroy(mps_device_t device) {
    if (device) {
        delete static_cast<MPSDevice*>(device);
    }
}

// ============================================================================
// Command Buffer Management
// ============================================================================

mps_command_buffer_t mps_command_buffer_create(mps_device_t device) {
    if (!device) return nullptr;
    
    try {
        MPSDevice* dev = static_cast<MPSDevice*>(device);
        MPSCommandBuffer* cmdBuf = new MPSCommandBuffer(dev);
        return static_cast<mps_command_buffer_t>(cmdBuf);
    } catch (...) {
        return nullptr;
    }
}

void mps_command_buffer_commit(mps_command_buffer_t cmd_buffer) {
    if (!cmd_buffer) return;
    
    MPSCommandBuffer* cmdBuf = static_cast<MPSCommandBuffer*>(cmd_buffer);
    [cmdBuf->buffer commit];
}

void mps_command_buffer_wait_until_completed(mps_command_buffer_t cmd_buffer) {
    if (!cmd_buffer) return;
    
    MPSCommandBuffer* cmdBuf = static_cast<MPSCommandBuffer*>(cmd_buffer);
    [cmdBuf->buffer waitUntilCompleted];
}

void mps_command_buffer_destroy(mps_command_buffer_t cmd_buffer) {
    if (cmd_buffer) {
        delete static_cast<MPSCommandBuffer*>(cmd_buffer);
    }
}

// ============================================================================
// Matrix Operations
// ============================================================================

mps_matrix_t mps_matrix_create(mps_device_t device, size_t rows, size_t cols) {
    if (!device) return nullptr;
    
    try {
        MPSDevice* dev = static_cast<MPSDevice*>(device);
        MPSMatrix* matrix = new MPSMatrix(dev, rows, cols);
        return static_cast<mps_matrix_t>(matrix);
    } catch (...) {
        return nullptr;
    }
}

void mps_matrix_destroy(mps_matrix_t matrix) {
    if (matrix) {
        delete static_cast<MPSMatrix*>(matrix);
    }
}

void mps_matrix_set_data(mps_matrix_t matrix, const float* data, size_t size) {
    if (!matrix || !data) return;
    
    MPSMatrix* mat = static_cast<MPSMatrix*>(matrix);
    size_t bytes = mat->rows * mat->cols * sizeof(float);
    if (size * sizeof(float) != bytes) {
        NSLog(@"Size mismatch in mps_matrix_set_data");
        return;
    }
    
    memcpy([mat->buffer contents], data, bytes);
}

void mps_matrix_get_data(mps_matrix_t matrix, float* data, size_t size) {
    if (!matrix || !data) return;
    
    MPSMatrix* mat = static_cast<MPSMatrix*>(matrix);
    size_t bytes = mat->rows * mat->cols * sizeof(float);
    if (size * sizeof(float) != bytes) {
        NSLog(@"Size mismatch in mps_matrix_get_data");
        return;
    }
    
    memcpy(data, [mat->buffer contents], bytes);
}

size_t mps_matrix_rows(mps_matrix_t matrix) {
    if (!matrix) return 0;
    return static_cast<MPSMatrix*>(matrix)->rows;
}

size_t mps_matrix_cols(mps_matrix_t matrix) {
    if (!matrix) return 0;
    return static_cast<MPSMatrix*>(matrix)->cols;
}

// ============================================================================
// Core Matrix Operations
// ============================================================================

void mps_matmul(mps_command_buffer_t cmd_buffer,
                mps_matrix_t a, mps_matrix_t b, mps_matrix_t c,
                float alpha, float beta) {
    if (!cmd_buffer || !a || !b || !c) return;
    
    MPSCommandBuffer* cmdBuf = static_cast<MPSCommandBuffer*>(cmd_buffer);
    MPSMatrix* matA = static_cast<MPSMatrix*>(a);
    MPSMatrix* matB = static_cast<MPSMatrix*>(b);
    MPSMatrix* matC = static_cast<MPSMatrix*>(c);
    
    // Create MPS matrix objects
    MPSMatrix* mpsA = [[MPSMatrix alloc] initWithBuffer:matA->buffer
                                             descriptor:matA->descriptor];
    MPSMatrix* mpsB = [[MPSMatrix alloc] initWithBuffer:matB->buffer
                                             descriptor:matB->descriptor];
    MPSMatrix* mpsC = [[MPSMatrix alloc] initWithBuffer:matC->buffer
                                             descriptor:matC->descriptor];
    
    // Create and encode matrix multiplication
    MPSMatrixMultiplication* matmul = [[MPSMatrixMultiplication alloc]
        initWithDevice:cmdBuf->device->device
        resultRows:matA->rows
        resultColumns:matB->cols
        interiorColumns:matA->cols];
    
    [matmul setAlpha:alpha];
    [matmul setBeta:beta];
    
    [matmul encodeToCommandBuffer:cmdBuf->buffer
                       leftMatrix:mpsA
                      rightMatrix:mpsB
                     resultMatrix:mpsC];
}

void mps_matrix_add(mps_command_buffer_t cmd_buffer,
                    mps_matrix_t a, mps_matrix_t b, mps_matrix_t result) {
    if (!cmd_buffer || !a || !b || !result) return;
    
    MPSCommandBuffer* cmdBuf = static_cast<MPSCommandBuffer*>(cmd_buffer);
    MPSMatrix* matA = static_cast<MPSMatrix*>(a);
    MPSMatrix* matB = static_cast<MPSMatrix*>(b);
    MPSMatrix* matResult = static_cast<MPSMatrix*>(result);
    
    // Use compute shader for element-wise addition
    id<MTLComputePipelineState> pipeline = nil;
    NSError* error = nil;
    
    NSString* kernelSource = @R"(
        #include <metal_stdlib>
        using namespace metal;
        
        kernel void matrix_add(
            device const float* a [[buffer(0)]],
            device const float* b [[buffer(1)]],
            device float* result [[buffer(2)]],
            constant uint& size [[buffer(3)]],
            uint id [[thread_position_in_grid]]
        ) {
            if (id < size) {
                result[id] = a[id] + b[id];
            }
        }
    )";
    
    id<MTLLibrary> library = [cmdBuf->device->device newLibraryWithSource:kernelSource
                                                                  options:nil
                                                                    error:&error];
    if (error) {
        NSLog(@"Error creating library: %@", error);
        return;
    }
    
    id<MTLFunction> function = [library newFunctionWithName:@"matrix_add"];
    pipeline = [cmdBuf->device->device newComputePipelineStateWithFunction:function
                                                                      error:&error];
    if (error) {
        NSLog(@"Error creating pipeline: %@", error);
        return;
    }
    
    id<MTLComputeCommandEncoder> encoder = [cmdBuf->buffer computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:matA->buffer offset:0 atIndex:0];
    [encoder setBuffer:matB->buffer offset:0 atIndex:1];
    [encoder setBuffer:matResult->buffer offset:0 atIndex:2];
    
    uint size = (uint)(matA->rows * matA->cols);
    [encoder setBytes:&size length:sizeof(uint) atIndex:3];
    
    MTLSize gridSize = MTLSizeMake(size, 1, 1);
    NSUInteger threadGroupSize = MIN(pipeline.maxTotalThreadsPerThreadgroup, 256);
    MTLSize threadgroupSize = MTLSizeMake(threadGroupSize, 1, 1);
    
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    [encoder endEncoding];
}

void mps_matrix_mul_elementwise(mps_command_buffer_t cmd_buffer,
                                 mps_matrix_t a, mps_matrix_t b, mps_matrix_t result) {
    if (!cmd_buffer || !a || !b || !result) return;
    
    MPSCommandBuffer* cmdBuf = static_cast<MPSCommandBuffer*>(cmd_buffer);
    MPSMatrix* matA = static_cast<MPSMatrix*>(a);
    MPSMatrix* matB = static_cast<MPSMatrix*>(b);
    MPSMatrix* matResult = static_cast<MPSMatrix*>(result);
    
    NSError* error = nil;
    NSString* kernelSource = @R"(
        #include <metal_stdlib>
        using namespace metal;
        
        kernel void matrix_mul_elementwise(
            device const float* a [[buffer(0)]],
            device const float* b [[buffer(1)]],
            device float* result [[buffer(2)]],
            constant uint& size [[buffer(3)]],
            uint id [[thread_position_in_grid]]
        ) {
            if (id < size) {
                result[id] = a[id] * b[id];
            }
        }
    )";
    
    id<MTLLibrary> library = [cmdBuf->device->device newLibraryWithSource:kernelSource
                                                                  options:nil
                                                                    error:&error];
    if (error) return;
    
    id<MTLFunction> function = [library newFunctionWithName:@"matrix_mul_elementwise"];
    id<MTLComputePipelineState> pipeline = [cmdBuf->device->device 
        newComputePipelineStateWithFunction:function error:&error];
    if (error) return;
    
    id<MTLComputeCommandEncoder> encoder = [cmdBuf->buffer computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:matA->buffer offset:0 atIndex:0];
    [encoder setBuffer:matB->buffer offset:0 atIndex:1];
    [encoder setBuffer:matResult->buffer offset:0 atIndex:2];
    
    uint size = (uint)(matA->rows * matA->cols);
    [encoder setBytes:&size length:sizeof(uint) atIndex:3];
    
    MTLSize gridSize = MTLSizeMake(size, 1, 1);
    NSUInteger threadGroupSize = MIN(pipeline.maxTotalThreadsPerThreadgroup, 256);
    MTLSize threadgroupSize = MTLSizeMake(threadGroupSize, 1, 1);
    
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    [encoder endEncoding];
}

void mps_matrix_transpose(mps_command_buffer_t cmd_buffer,
                          mps_matrix_t input, mps_matrix_t output) {
    if (!cmd_buffer || !input || !output) return;
    
    MPSCommandBuffer* cmdBuf = static_cast<MPSCommandBuffer*>(cmd_buffer);
    MPSMatrix* matInput = static_cast<MPSMatrix*>(input);
    MPSMatrix* matOutput = static_cast<MPSMatrix*>(output);
    
    NSError* error = nil;
    NSString* kernelSource = @R"(
        #include <metal_stdlib>
        using namespace metal;
        
        kernel void matrix_transpose(
            device const float* input [[buffer(0)]],
            device float* output [[buffer(1)]],
            constant uint& rows [[buffer(2)]],
            constant uint& cols [[buffer(3)]],
            uint2 id [[thread_position_in_grid]]
        ) {
            if (id.x < cols && id.y < rows) {
                output[id.x * rows + id.y] = input[id.y * cols + id.x];
            }
        }
    )";
    
    id<MTLLibrary> library = [cmdBuf->device->device newLibraryWithSource:kernelSource
                                                                  options:nil
                                                                    error:&error];
    if (error) return;
    
    id<MTLFunction> function = [library newFunctionWithName:@"matrix_transpose"];
    id<MTLComputePipelineState> pipeline = [cmdBuf->device->device 
        newComputePipelineStateWithFunction:function error:&error];
    if (error) return;
    
    id<MTLComputeCommandEncoder> encoder = [cmdBuf->buffer computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:matInput->buffer offset:0 atIndex:0];
    [encoder setBuffer:matOutput->buffer offset:0 atIndex:1];
    
    uint rows = (uint)matInput->rows;
    uint cols = (uint)matInput->cols;
    [encoder setBytes:&rows length:sizeof(uint) atIndex:2];
    [encoder setBytes:&cols length:sizeof(uint) atIndex:3];
    
    MTLSize gridSize = MTLSizeMake(cols, rows, 1);
    MTLSize threadgroupSize = MTLSizeMake(16, 16, 1);
    
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    [encoder endEncoding];
}

// ============================================================================
// Activation Functions
// ============================================================================

void mps_relu_forward(mps_command_buffer_t cmd_buffer, mps_matrix_t x) {
    if (!cmd_buffer || !x) return;
    
    MPSCommandBuffer* cmdBuf = static_cast<MPSCommandBuffer*>(cmd_buffer);
    MPSMatrix* mat = static_cast<MPSMatrix*>(x);
    
    // Create MPSImage from buffer (simplified approach using compute shader)
    NSError* error = nil;
    NSString* kernelSource = @R"(
        #include <metal_stdlib>
        using namespace metal;
        
        kernel void relu(
            device float* x [[buffer(0)]],
            constant uint& size [[buffer(1)]],
            uint id [[thread_position_in_grid]]
        ) {
            if (id < size) {
                x[id] = max(x[id], 0.0f);
            }
        }
    )";
    
    id<MTLLibrary> library = [cmdBuf->device->device newLibraryWithSource:kernelSource
                                                                  options:nil
                                                                    error:&error];
    if (error) return;
    
    id<MTLFunction> function = [library newFunctionWithName:@"relu"];
    id<MTLComputePipelineState> pipeline = [cmdBuf->device->device 
        newComputePipelineStateWithFunction:function error:&error];
    if (error) return;
    
    id<MTLComputeCommandEncoder> encoder = [cmdBuf->buffer computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:mat->buffer offset:0 atIndex:0];
    
    uint size = (uint)(mat->rows * mat->cols);
    [encoder setBytes:&size length:sizeof(uint) atIndex:1];
    
    MTLSize gridSize = MTLSizeMake(size, 1, 1);
    NSUInteger threadGroupSize = MIN(pipeline.maxTotalThreadsPerThreadgroup, 256);
    MTLSize threadgroupSize = MTLSizeMake(threadGroupSize, 1, 1);
    
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    [encoder endEncoding];
}

void mps_sigmoid_forward(mps_command_buffer_t cmd_buffer, mps_matrix_t x) {
    if (!cmd_buffer || !x) return;
    
    MPSCommandBuffer* cmdBuf = static_cast<MPSCommandBuffer*>(cmd_buffer);
    MPSMatrix* mat = static_cast<MPSMatrix*>(x);
    
    NSError* error = nil;
    NSString* kernelSource = @R"(
        #include <metal_stdlib>
        using namespace metal;
        
        kernel void sigmoid(
            device float* x [[buffer(0)]],
            constant uint& size [[buffer(1)]],
            uint id [[thread_position_in_grid]]
        ) {
            if (id < size) {
                x[id] = 1.0f / (1.0f + exp(-x[id]));
            }
        }
    )";
    
    id<MTLLibrary> library = [cmdBuf->device->device newLibraryWithSource:kernelSource
                                                                  options:nil
                                                                    error:&error];
    if (error) return;
    
    id<MTLFunction> function = [library newFunctionWithName:@"sigmoid"];
    id<MTLComputePipelineState> pipeline = [cmdBuf->device->device 
        newComputePipelineStateWithFunction:function error:&error];
    if (error) return;
    
    id<MTLComputeCommandEncoder> encoder = [cmdBuf->buffer computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:mat->buffer offset:0 atIndex:0];
    
    uint size = (uint)(mat->rows * mat->cols);
    [encoder setBytes:&size length:sizeof(uint) atIndex:1];
    
    MTLSize gridSize = MTLSizeMake(size, 1, 1);
    NSUInteger threadGroupSize = MIN(pipeline.maxTotalThreadsPerThreadgroup, 256);
    MTLSize threadgroupSize = MTLSizeMake(threadGroupSize, 1, 1);
    
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    [encoder endEncoding];
}

void mps_tanh_forward(mps_command_buffer_t cmd_buffer, mps_matrix_t x) {
    if (!cmd_buffer || !x) return;
    
    MPSCommandBuffer* cmdBuf = static_cast<MPSCommandBuffer*>(cmd_buffer);
    MPSMatrix* mat = static_cast<MPSMatrix*>(x);
    
    NSError* error = nil;
    NSString* kernelSource = @R"(
        #include <metal_stdlib>
        using namespace metal;
        
        kernel void tanh_activation(
            device float* x [[buffer(0)]],
            constant uint& size [[buffer(1)]],
            uint id [[thread_position_in_grid]]
        ) {
            if (id < size) {
                x[id] = tanh(x[id]);
            }
        }
    )";
    
    id<MTLLibrary> library = [cmdBuf->device->device newLibraryWithSource:kernelSource
                                                                  options:nil
                                                                    error:&error];
    if (error) return;
    
    id<MTLFunction> function = [library newFunctionWithName:@"tanh_activation"];
    id<MTLComputePipelineState> pipeline = [cmdBuf->device->device 
        newComputePipelineStateWithFunction:function error:&error];
    if (error) return;
    
    id<MTLComputeCommandEncoder> encoder = [cmdBuf->buffer computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:mat->buffer offset:0 atIndex:0];
    
    uint size = (uint)(mat->rows * mat->cols);
    [encoder setBytes:&size length:sizeof(uint) atIndex:1];
    
    MTLSize gridSize = MTLSizeMake(size, 1, 1);
    NSUInteger threadGroupSize = MIN(pipeline.maxTotalThreadsPerThreadgroup, 256);
    MTLSize threadgroupSize = MTLSizeMake(threadGroupSize, 1, 1);
    
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    [encoder endEncoding];
}

// ============================================================================
// Linear Layer Operations
// ============================================================================

void mps_linear_forward(mps_command_buffer_t cmd_buffer,
                        mps_matrix_t input, mps_matrix_t weights, mps_matrix_t bias,
                        mps_matrix_t output, int activation_type) {
    if (!cmd_buffer || !input || !weights || !output) return;
    
    // output = input @ weights^T + bias
    mps_matmul(cmd_buffer, input, weights, output, 1.0f, 0.0f);
    
    // Add bias if provided
    if (bias) {
        MPSCommandBuffer* cmdBuf = static_cast<MPSCommandBuffer*>(cmd_buffer);
        MPSMatrix* matOutput = static_cast<MPSMatrix*>(output);
        MPSMatrix* matBias = static_cast<MPSMatrix*>(bias);
        
        NSError* error = nil;
        NSString* kernelSource = @R"(
            #include <metal_stdlib>
            using namespace metal;
            
            kernel void add_bias(
                device float* output [[buffer(0)]],
                device const float* bias [[buffer(1)]],
                constant uint& rows [[buffer(2)]],
                constant uint& cols [[buffer(3)]],
                uint2 id [[thread_position_in_grid]]
            ) {
                if (id.x < cols && id.y < rows) {
                    output[id.y * cols + id.x] += bias[id.x];
                }
            }
        )";
        
        id<MTLLibrary> library = [cmdBuf->device->device newLibraryWithSource:kernelSource
                                                                      options:nil
                                                                        error:&error];
        if (error) return;
        
        id<MTLFunction> function = [library newFunctionWithName:@"add_bias"];
        id<MTLComputePipelineState> pipeline = [cmdBuf->device->device 
            newComputePipelineStateWithFunction:function error:&error];
        if (error) return;
        
        id<MTLComputeCommandEncoder> encoder = [cmdBuf->buffer computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:matOutput->buffer offset:0 atIndex:0];
        [encoder setBuffer:matBias->buffer offset:0 atIndex:1];
        
        uint rows = (uint)matOutput->rows;
        uint cols = (uint)matOutput->cols;
        [encoder setBytes:&rows length:sizeof(uint) atIndex:2];
        [encoder setBytes:&cols length:sizeof(uint) atIndex:3];
        
        MTLSize gridSize = MTLSizeMake(cols, rows, 1);
        MTLSize threadgroupSize = MTLSizeMake(16, 16, 1);
        
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
        [encoder endEncoding];
    }
    
    // Apply activation
    if (activation_type == 1) {
        mps_relu_forward(cmd_buffer, output);
    } else if (activation_type == 2) {
        mps_sigmoid_forward(cmd_buffer, output);
    } else if (activation_type == 3) {
        mps_tanh_forward(cmd_buffer, output);
    }
}

void mps_linear_backward_weights(mps_command_buffer_t cmd_buffer,
                                  mps_matrix_t grad_output, mps_matrix_t input,
                                  mps_matrix_t grad_weights, mps_matrix_t grad_bias) {
    if (!cmd_buffer || !grad_output || !input || !grad_weights) return;
    
    // grad_weights = grad_output^T @ input
    MPSMatrix* matGradOutput = static_cast<MPSMatrix*>(grad_output);
    MPSMatrix* matInput = static_cast<MPSMatrix*>(input);
    
    // Create transposed descriptors
    mps_matrix_t grad_output_t = mps_matrix_create(
        static_cast<mps_device_t>(matGradOutput->device),
        matGradOutput->cols, matGradOutput->rows);
    
    mps_matrix_transpose(cmd_buffer, grad_output, grad_output_t);
    mps_matmul(cmd_buffer, grad_output_t, input, grad_weights, 1.0f, 0.0f);
    mps_matrix_destroy(grad_output_t);
    
    // Compute grad_bias if provided
    if (grad_bias) {
        MPSCommandBuffer* cmdBuf = static_cast<MPSCommandBuffer*>(cmd_buffer);
        MPSMatrix* matGradBias = static_cast<MPSMatrix*>(grad_bias);
        
        NSError* error = nil;
        NSString* kernelSource = @R"(
            #include <metal_stdlib>
            using namespace metal;
            
            kernel void sum_over_batch(
                device const float* grad_output [[buffer(0)]],
                device float* grad_bias [[buffer(1)]],
                constant uint& rows [[buffer(2)]],
                constant uint& cols [[buffer(3)]],
                uint id [[thread_position_in_grid]]
            ) {
                if (id < cols) {
                    float sum = 0.0f;
                    for (uint i = 0; i < rows; i++) {
                        sum += grad_output[i * cols + id];
                    }
                    grad_bias[id] = sum;
                }
            }
        )";
        
        id<MTLLibrary> library = [cmdBuf->device->device newLibraryWithSource:kernelSource
                                                                      options:nil
                                                                        error:&error];
        if (error) return;
        
        id<MTLFunction> function = [library newFunctionWithName:@"sum_over_batch"];
        id<MTLComputePipelineState> pipeline = [cmdBuf->device->device 
            newComputePipelineStateWithFunction:function error:&error];
        if (error) return;
        
        id<MTLComputeCommandEncoder> encoder = [cmdBuf->buffer computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:matGradOutput->buffer offset:0 atIndex:0];
        [encoder setBuffer:matGradBias->buffer offset:0 atIndex:1];
        
        uint rows = (uint)matGradOutput->rows;
        uint cols = (uint)matGradOutput->cols;
        [encoder setBytes:&rows length:sizeof(uint) atIndex:2];
        [encoder setBytes:&cols length:sizeof(uint) atIndex:3];
        
        MTLSize gridSize = MTLSizeMake(cols, 1, 1);
        NSUInteger threadGroupSize = MIN(pipeline.maxTotalThreadsPerThreadgroup, 256);
        MTLSize threadgroupSize = MTLSizeMake(threadGroupSize, 1, 1);
        
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
        [encoder endEncoding];
    }
}

void mps_linear_backward_input(mps_command_buffer_t cmd_buffer,
                                mps_matrix_t grad_output, mps_matrix_t weights,
                                mps_matrix_t grad_input) {
    if (!cmd_buffer || !grad_output || !weights || !grad_input) return;
    
    // grad_input = grad_output @ weights
    mps_matmul(cmd_buffer, grad_output, weights, grad_input, 1.0f, 0.0f);
}

// ============================================================================
// Convolution Operations
// ============================================================================

mps_conv_descriptor_t mps_conv_descriptor_create(
    int kernel_h, int kernel_w,
    int input_channels, int output_channels,
    int stride_h, int stride_w) {
    
    try {
        MPSConvDescriptor* desc = new MPSConvDescriptor(
            kernel_h, kernel_w, input_channels, output_channels,
            stride_h, stride_w);
        return static_cast<mps_conv_descriptor_t>(desc);
    } catch (...) {
        return nullptr;
    }
}

void mps_conv_descriptor_destroy(mps_conv_descriptor_t desc) {
    if (desc) {
        delete static_cast<MPSConvDescriptor*>(desc);
    }
}

void mps_conv2d_forward(mps_command_buffer_t cmd_buffer,
                        mps_conv_descriptor_t desc,
                        mps_matrix_t input, mps_matrix_t weights, mps_matrix_t bias,
                        mps_matrix_t output, int batch_size) {
    if (!cmd_buffer || !desc || !input || !weights || !output) return;
    
    MPSCommandBuffer* cmdBuf = static_cast<MPSCommandBuffer*>(cmd_buffer);
    MPSConvDescriptor* convDesc = static_cast<MPSConvDescriptor*>(desc);
    MPSMatrix* matInput = static_cast<MPSMatrix*>(input);
    MPSMatrix* matWeights = static_cast<MPSMatrix*>(weights);
    MPSMatrix* matOutput = static_cast<MPSMatrix*>(output);
    
    // Create MPS convolution descriptor
    MPSCNNConvolutionDescriptor* cnnDesc = [MPSCNNConvolutionDescriptor
        cnnConvolutionDescriptorWithKernelWidth:convDesc->kernel_w
                                   kernelHeight:convDesc->kernel_h
                           inputFeatureChannels:convDesc->input_channels
                          outputFeatureChannels:convDesc->output_channels];
    
    cnnDesc.strideInPixelsX = convDesc->stride_w;
    cnnDesc.strideInPixelsY = convDesc->stride_h;
    
    // Note: Full MPSCNNConvolution implementation requires proper image descriptor setup
    // This is a simplified version - production code would need proper MPSImage handling
    NSLog(@"Conv2D forward requires full MPSImage integration");
}

void mps_conv2d_backward_input(mps_command_buffer_t cmd_buffer,
                                mps_conv_descriptor_t desc,
                                mps_matrix_t grad_output, mps_matrix_t weights,
                                mps_matrix_t grad_input, int batch_size) {
    if (!cmd_buffer || !desc || !grad_output || !weights || !grad_input) return;
    
    NSLog(@"Conv2D backward input requires full MPSCNNConvolutionGradient integration");
}

void mps_conv2d_backward_weights(mps_command_buffer_t cmd_buffer,
                                  mps_conv_descriptor_t desc,
                                  mps_matrix_t input, mps_matrix_t grad_output,
                                  mps_matrix_t grad_weights, mps_matrix_t grad_bias,
                                  int batch_size) {
    if (!cmd_buffer || !desc || !input || !grad_output || !grad_weights) return;
    
    NSLog(@"Conv2D backward weights requires full MPSCNNConvolutionGradient integration");
}

// ============================================================================
// Pooling Operations
// ============================================================================

mps_pool_descriptor_t mps_maxpool_descriptor_create(
    int kernel_h, int kernel_w,
    int stride_h, int stride_w) {
    
    try {
        MPSPoolDescriptor* desc = new MPSPoolDescriptor(
            kernel_h, kernel_w, stride_h, stride_w);
        return static_cast<mps_pool_descriptor_t>(desc);
    } catch (...) {
        return nullptr;
    }
}

void mps_pool_descriptor_destroy(mps_pool_descriptor_t desc) {
    if (desc) {
        delete static_cast<MPSPoolDescriptor*>(desc);
    }
}

void mps_maxpool_forward(mps_command_buffer_t cmd_buffer,
                         mps_pool_descriptor_t desc,
                         mps_matrix_t input, mps_matrix_t output,
                         mps_matrix_t indices, int batch_size) {
    if (!cmd_buffer || !desc || !input || !output) return;
    
    NSLog(@"MaxPool forward requires full MPSCNNPoolingMax integration with MPSImage");
}

void mps_maxpool_backward(mps_command_buffer_t cmd_buffer,
                          mps_pool_descriptor_t desc,
                          mps_matrix_t grad_output, mps_matrix_t indices,
                          mps_matrix_t grad_input, int batch_size) {
    if (!cmd_buffer || !desc || !grad_output || !grad_input) return;
    
    NSLog(@"MaxPool backward requires full MPSCNNPoolingMaxGradient integration");
}

// ============================================================================
// Optimizer Operations
// ============================================================================

void mps_adam_step(mps_command_buffer_t cmd_buffer,
                   mps_matrix_t weights, mps_matrix_t gradients,
                   mps_matrix_t m, mps_matrix_t v,
                   float lr, float beta1, float beta2,
                   float beta1_power, float beta2_power,
                   float epsilon, float weight_decay) {
    if (!cmd_buffer || !weights || !gradients || !m || !v) return;
    
    MPSCommandBuffer* cmdBuf = static_cast<MPSCommandBuffer*>(cmd_buffer);
    MPSMatrix* matWeights = static_cast<MPSMatrix*>(weights);
    MPSMatrix* matGradients = static_cast<MPSMatrix*>(gradients);
    MPSMatrix* matM = static_cast<MPSMatrix*>(m);
    MPSMatrix* matV = static_cast<MPSMatrix*>(v);
    
    NSError* error = nil;
    NSString* kernelSource = @R"(
        #include <metal_stdlib>
        using namespace metal;
        
        struct AdamParams {
            float lr;
            float beta1;
            float beta2;
            float beta1_power;
            float beta2_power;
            float epsilon;
            float weight_decay;
        };
        
        kernel void adam_step(
            device float* weights [[buffer(0)]],
            device const float* gradients [[buffer(1)]],
            device float* m [[buffer(2)]],
            device float* v [[buffer(3)]],
            constant AdamParams& params [[buffer(4)]],
            constant uint& size [[buffer(5)]],
            uint id [[thread_position_in_grid]]
        ) {
            if (id < size) {
                float grad = gradients[id];
                
                // Apply weight decay
                if (params.weight_decay != 0.0f) {
                    grad += params.weight_decay * weights[id];
                }
                
                // Update biased first moment estimate
                m[id] = params.beta1 * m[id] + (1.0f - params.beta1) * grad;
                
                // Update biased second raw moment estimate
                v[id] = params.beta2 * v[id] + (1.0f - params.beta2) * grad * grad;
                
                // Compute bias-corrected first moment estimate
                float m_hat = m[id] / (1.0f - params.beta1_power);
                
                // Compute bias-corrected second raw moment estimate
                float v_hat = v[id] / (1.0f - params.beta2_power);
                
                // Update parameters
                weights[id] -= params.lr * m_hat / (sqrt(v_hat) + params.epsilon);
            }
        }
    )";
    
    id<MTLLibrary> library = [cmdBuf->device->device newLibraryWithSource:kernelSource
                                                                  options:nil
                                                                    error:&error];
    if (error) return;
    
    id<MTLFunction> function = [library newFunctionWithName:@"adam_step"];
    id<MTLComputePipelineState> pipeline = [cmdBuf->device->device 
        newComputePipelineStateWithFunction:function error:&error];
    if (error) return;
    
    struct AdamParamsStruct {
        float lr;
        float beta1;
        float beta2;
        float beta1_power;
        float beta2_power;
        float epsilon;
        float weight_decay;
    } adamParams = {lr, beta1, beta2, beta1_power, beta2_power, epsilon, weight_decay};
    
    id<MTLComputeCommandEncoder> encoder = [cmdBuf->buffer computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:matWeights->buffer offset:0 atIndex:0];
    [encoder setBuffer:matGradients->buffer offset:0 atIndex:1];
    [encoder setBuffer:matM->buffer offset:0 atIndex:2];
    [encoder setBuffer:matV->buffer offset:0 atIndex:3];
    [encoder setBytes:&adamParams length:sizeof(AdamParamsStruct) atIndex:4];
    
    uint size = (uint)(matWeights->rows * matWeights->cols);
    [encoder setBytes:&size length:sizeof(uint) atIndex:5];
    
    MTLSize gridSize = MTLSizeMake(size, 1, 1);
    NSUInteger threadGroupSize = MIN(pipeline.maxTotalThreadsPerThreadgroup, 256);
    MTLSize threadgroupSize = MTLSizeMake(threadGroupSize, 1, 1);
    
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    [encoder endEncoding];
}

// ============================================================================
// Utility Operations
// ============================================================================

void mps_matrix_zero(mps_command_buffer_t cmd_buffer, mps_matrix_t matrix) {
    if (!cmd_buffer || !matrix) return;
    
    MPSCommandBuffer* cmdBuf = static_cast<MPSCommandBuffer*>(cmd_buffer);
    MPSMatrix* mat = static_cast<MPSMatrix*>(matrix);
    
    id<MTLBlitCommandEncoder> blitEncoder = [cmdBuf->buffer blitCommandEncoder];
    [blitEncoder fillBuffer:mat->buffer
                      range:NSMakeRange(0, mat->rows * mat->cols * sizeof(float))
                      value:0];
    [blitEncoder endEncoding];
}

void mps_matrix_copy(mps_command_buffer_t cmd_buffer,
                     mps_matrix_t src, mps_matrix_t dst) {
    if (!cmd_buffer || !src || !dst) return;
    
    MPSCommandBuffer* cmdBuf = static_cast<MPSCommandBuffer*>(cmd_buffer);
    MPSMatrix* matSrc = static_cast<MPSMatrix*>(src);
    MPSMatrix* matDst = static_cast<MPSMatrix*>(dst);
    
    size_t bytes = matSrc->rows * matSrc->cols * sizeof(float);
    
    id<MTLBlitCommandEncoder> blitEncoder = [cmdBuf->buffer blitCommandEncoder];
    [blitEncoder copyFromBuffer:matSrc->buffer
                   sourceOffset:0
                       toBuffer:matDst->buffer
              destinationOffset:0
                           size:bytes];
    [blitEncoder endEncoding];
}

void mps_mse_gradient(mps_command_buffer_t cmd_buffer,
                      mps_matrix_t predictions, mps_matrix_t targets,
                      mps_matrix_t grad_output, float scale) {
    if (!cmd_buffer || !predictions || !targets || !grad_output) return;
    
    MPSCommandBuffer* cmdBuf = static_cast<MPSCommandBuffer*>(cmd_buffer);
    MPSMatrix* matPredictions = static_cast<MPSMatrix*>(predictions);
    MPSMatrix* matTargets = static_cast<MPSMatrix*>(targets);
    MPSMatrix* matGradOutput = static_cast<MPSMatrix*>(grad_output);
    
    NSError* error = nil;
    NSString* kernelSource = @R"(
        #include <metal_stdlib>
        using namespace metal;
        
        kernel void mse_gradient(
            device const float* predictions [[buffer(0)]],
            device const float* targets [[buffer(1)]],
            device float* grad_output [[buffer(2)]],
            constant float& scale [[buffer(3)]],
            constant uint& size [[buffer(4)]],
            uint id [[thread_position_in_grid]]
        ) {
            if (id < size) {
                grad_output[id] = scale * (predictions[id] - targets[id]);
            }
        }
    )";
    
    id<MTLLibrary> library = [cmdBuf->device->device newLibraryWithSource:kernelSource
                                                                  options:nil
                                                                    error:&error];
    if (error) return;
    
    id<MTLFunction> function = [library newFunctionWithName:@"mse_gradient"];
    id<MTLComputePipelineState> pipeline = [cmdBuf->device->device 
        newComputePipelineStateWithFunction:function error:&error];
    if (error) return;
    
    id<MTLComputeCommandEncoder> encoder = [cmdBuf->buffer computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:matPredictions->buffer offset:0 atIndex:0];
    [encoder setBuffer:matTargets->buffer offset:0 atIndex:1];
    [encoder setBuffer:matGradOutput->buffer offset:0 atIndex:2];
    [encoder setBytes:&scale length:sizeof(float) atIndex:3];
    
    uint size = (uint)(matPredictions->rows * matPredictions->cols);
    [encoder setBytes:&size length:sizeof(uint) atIndex:4];
    
    MTLSize gridSize = MTLSizeMake(size, 1, 1);
    NSUInteger threadGroupSize = MIN(pipeline.maxTotalThreadsPerThreadgroup, 256);
    MTLSize threadgroupSize = MTLSizeMake(threadGroupSize, 1, 1);
    
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    [encoder endEncoding];
}

// ============================================================================
// Synchronization
// ============================================================================

void mps_synchronize(mps_device_t device) {
    if (!device) return;
    
    MPSDevice* dev = static_cast<MPSDevice*>(device);
    id<MTLCommandBuffer> cmdBuf = [dev->commandQueue commandBuffer];
    [cmdBuf commit];
    [cmdBuf waitUntilCompleted];
}
