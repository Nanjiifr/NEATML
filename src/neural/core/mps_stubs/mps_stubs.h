/*
 * MPS (Metal Performance Shaders) C Stubs
 * PyTorch-inspired direct MPS implementation
 */

#ifndef MPS_STUBS_H
#define MPS_STUBS_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque handle types */
typedef void* mps_device_t;
typedef void* mps_command_buffer_t;
typedef void* mps_matrix_t;
typedef void* mps_vector_t;
typedef void* mps_kernel_t;
typedef void* mps_graph_t;

/* Device management */
mps_device_t mps_device_create(void);
void mps_device_destroy(mps_device_t device);

/* Command buffer management */
mps_command_buffer_t mps_command_buffer_create(mps_device_t device);
void mps_command_buffer_commit(mps_command_buffer_t cmd_buffer);
void mps_command_buffer_wait_until_completed(mps_command_buffer_t cmd_buffer);
void mps_command_buffer_destroy(mps_command_buffer_t cmd_buffer);

/* Matrix operations */
mps_matrix_t mps_matrix_create(mps_device_t device, size_t rows, size_t cols);
void mps_matrix_destroy(mps_matrix_t matrix);
void mps_matrix_set_data(mps_matrix_t matrix, const float* data, size_t size);
void mps_matrix_get_data(mps_matrix_t matrix, float* data, size_t size);
size_t mps_matrix_rows(mps_matrix_t matrix);
size_t mps_matrix_cols(mps_matrix_t matrix);

/* Core operations using MPS built-ins */
void mps_matmul(mps_command_buffer_t cmd_buffer,
                mps_matrix_t a, mps_matrix_t b, mps_matrix_t c,
                float alpha, float beta);

void mps_matrix_add(mps_command_buffer_t cmd_buffer,
                    mps_matrix_t a, mps_matrix_t b, mps_matrix_t result);

void mps_matrix_mul_elementwise(mps_command_buffer_t cmd_buffer,
                                 mps_matrix_t a, mps_matrix_t b, mps_matrix_t result);

void mps_matrix_transpose(mps_command_buffer_t cmd_buffer,
                          mps_matrix_t input, mps_matrix_t output);

/* Activation functions using MPSCNNNeuron* */
void mps_relu_forward(mps_command_buffer_t cmd_buffer, mps_matrix_t x);
void mps_sigmoid_forward(mps_command_buffer_t cmd_buffer, mps_matrix_t x);
void mps_tanh_forward(mps_command_buffer_t cmd_buffer, mps_matrix_t x);

/* Linear layer operations */
void mps_linear_forward(mps_command_buffer_t cmd_buffer,
                        mps_matrix_t input, mps_matrix_t weights, mps_matrix_t bias,
                        mps_matrix_t output, int activation_type);

void mps_linear_backward_weights(mps_command_buffer_t cmd_buffer,
                                  mps_matrix_t grad_output, mps_matrix_t input,
                                  mps_matrix_t grad_weights, mps_matrix_t grad_bias);

void mps_linear_backward_input(mps_command_buffer_t cmd_buffer,
                                mps_matrix_t grad_output, mps_matrix_t weights,
                                mps_matrix_t grad_input);

/* Convolution operations using MPSCNNConvolution */
typedef void* mps_conv_descriptor_t;

mps_conv_descriptor_t mps_conv_descriptor_create(
    int kernel_h, int kernel_w,
    int input_channels, int output_channels,
    int stride_h, int stride_w);

void mps_conv_descriptor_destroy(mps_conv_descriptor_t desc);

void mps_conv2d_forward(mps_command_buffer_t cmd_buffer,
                        mps_conv_descriptor_t desc,
                        mps_matrix_t input, mps_matrix_t weights, mps_matrix_t bias,
                        mps_matrix_t output, int batch_size);

void mps_conv2d_backward_input(mps_command_buffer_t cmd_buffer,
                                mps_conv_descriptor_t desc,
                                mps_matrix_t grad_output, mps_matrix_t weights,
                                mps_matrix_t grad_input, int batch_size);

void mps_conv2d_backward_weights(mps_command_buffer_t cmd_buffer,
                                  mps_conv_descriptor_t desc,
                                  mps_matrix_t input, mps_matrix_t grad_output,
                                  mps_matrix_t grad_weights, mps_matrix_t grad_bias,
                                  int batch_size);

/* Pooling operations using MPSCNNPooling */
typedef void* mps_pool_descriptor_t;

mps_pool_descriptor_t mps_maxpool_descriptor_create(
    int kernel_h, int kernel_w,
    int stride_h, int stride_w);

void mps_pool_descriptor_destroy(mps_pool_descriptor_t desc);

void mps_maxpool_forward(mps_command_buffer_t cmd_buffer,
                         mps_pool_descriptor_t desc,
                         mps_matrix_t input, mps_matrix_t output,
                         mps_matrix_t indices, int batch_size);

void mps_maxpool_backward(mps_command_buffer_t cmd_buffer,
                          mps_pool_descriptor_t desc,
                          mps_matrix_t grad_output, mps_matrix_t indices,
                          mps_matrix_t grad_input, int batch_size);

/* Optimizer operations */
void mps_adam_step(mps_command_buffer_t cmd_buffer,
                   mps_matrix_t weights, mps_matrix_t gradients,
                   mps_matrix_t m, mps_matrix_t v,
                   float lr, float beta1, float beta2,
                   float beta1_power, float beta2_power,
                   float epsilon, float weight_decay);

/* Utility operations */
void mps_matrix_zero(mps_command_buffer_t cmd_buffer, mps_matrix_t matrix);
void mps_matrix_copy(mps_command_buffer_t cmd_buffer,
                     mps_matrix_t src, mps_matrix_t dst);

void mps_mse_gradient(mps_command_buffer_t cmd_buffer,
                      mps_matrix_t predictions, mps_matrix_t targets,
                      mps_matrix_t grad_output, float scale);

/* Synchronization */
void mps_synchronize(mps_device_t device);

#ifdef __cplusplus
}
#endif

#endif /* MPS_STUBS_H */
