#ifndef FC_H
#define FC_H

#include <memory>

#include "matrix.h"
#include "optimizer.h"

class FCLayer { 
  public:
    Matrix weights, grad_weights;
    Matrix bias, grad_bias;
    Matrix last_input;   // Save for backpropagation
    std::unique_ptr<Optimizer> weights_optimizer, bias_optimizer;

    FCLayer(int input_size, int output_size, std::unique_ptr<Optimizer> opt)
    : weights(Matrix(output_size, input_size)),
      grad_weights(Matrix(output_size, input_size)),
      bias(Matrix(output_size, 1)),
      grad_bias(Matrix(output_size, 1)),
      weights_optimizer(std::move(opt -> clone())),
      bias_optimizer(std::move(opt -> clone())) {
        weights.he_init(input_size);
    }

    Matrix forward(const Matrix& input) {    // Assume input is already flattened(column vector)
        last_input = input;
        Matrix result = weights * input + bias;
        return result;
    }

    Matrix backward(const Matrix& grad_out) {
        // Gradients w.r.t. weights, bias, and input
        grad_weights = grad_out * last_input.transpose();
        grad_bias = grad_out;
        Matrix grad_input = weights.transpose() * grad_out;
        return grad_input;
    }

    void update() {
        weights_optimizer -> update(weights, grad_weights);
        bias_optimizer -> update(bias, grad_bias);
    }
};

#endif