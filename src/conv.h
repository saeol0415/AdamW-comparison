#ifndef CONV_H
#define CONV_H

#include <memory>

#include "optimizer.h"

class ConvLayer {
  public:
    std::vector<std::vector<Matrix>> kernels, grad_kernels;
    double bias, grad_bias;
    std::vector<Matrix> input;
    std::vector<Matrix> last_output;
    std::unique_ptr<Optimizer> kernel_optimizer, bias_optimizer;
    int num_kernels;

    ConvLayer(int in_ch, int out_ch, std::unique_ptr<Optimizer> opt) : bias(0.0), grad_bias(0.0),
      kernel_optimizer(std::move(opt -> clone())), bias_optimizer(std::move(opt -> clone())), num_kernels(out_ch) {
        kernels.resize(num_kernels, std::vector<Matrix>(in_ch, Matrix(3, 3)));
        grad_kernels.resize(num_kernels, std::vector<Matrix>(in_ch, Matrix(3, 3)));
        for (auto& filter_kernels : kernels) { 
            for (auto& kernel : filter_kernels) {
                kernel.he_init(in_ch * 3 * 3);
            }
        }
    }

    std::vector<Matrix> forward(const std::vector<Matrix>& inputs) {
        input = inputs;
        std::vector<Matrix> outputs(num_kernels);

        for (int i = 0; i < num_kernels; i++) {
            outputs[i] = Matrix(inputs[0].rows, inputs[0].cols); // Initialize output feature map
            for (int c = 0; c < inputs.size(); c++) {
                Matrix padded_input = inputs[c].pad(1);
                outputs[i] += padded_input.correlate(kernels[i][c]); // Sum across channels
            }
            outputs[i] += bias; // Add bias to entire feature map
        }
        last_output = outputs; // Save for backward pass
        return outputs;
    }

    std::vector<Matrix> backward(const std::vector<Matrix>& grad_out) {
        std::vector<Matrix> grad_input(input.size(), Matrix(input[0].rows, input[0].cols));
        grad_kernels = std::vector<std::vector<Matrix>>(num_kernels, std::vector<Matrix>(input.size(), Matrix(3, 3)));
        grad_bias = 0.0;

        for (int i = 0; i < num_kernels; i++) {
            Matrix padded_grad_out = grad_out[i].pad(1); // Adjust padding
            for (int c = 0; c < input.size(); c++) {
                Matrix padded_input = input[c].pad(1);
                // Gradients w.r.t. kernel and input
                grad_kernels[i][c] = padded_input.correlate(grad_out[i]);
                grad_input[c] += padded_grad_out.correlate(kernels[i][c], true); // Full convolution for input grad
            }
            for (int r = 0; r < grad_out[i].rows; r++) {
                for (int c = 0; c < grad_out[i].cols; c++) {
                    // Gradients w.r.t. bias
                    grad_bias += grad_out[i].data[r][c];
                }
            }
        }
        return grad_input;
    }

    void update() {
        for (int i = 0; i < num_kernels; i++) {
            for (int c = 0; c < kernels[i].size(); c++) {
                kernel_optimizer->update(kernels[i][c], grad_kernels[i][c]);
            }
        }
        bias_optimizer->update(bias, grad_bias);
    }
};

#endif