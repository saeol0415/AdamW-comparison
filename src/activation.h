#ifndef ACTIVATION_H
#define ACTIVATION_H

#include <cmath>

#include "matrix.h"

inline Matrix leakyReLU(const Matrix& input, double alpha = 0.01) {
    Matrix output = input;

    for (int i = 0; i < input.rows; i++) {
        for (int j = 0; j < input.cols; j++) {
            output.data[i][j] = (input.data[i][j] > 0) ? input.data[i][j] : alpha * input.data[i][j];
        }
    }
    return output;
}

inline Matrix leakyReLU_backward(const Matrix& input, const Matrix& grad_output, double alpha = 0.01) {
    if (input.rows != grad_output.rows || input.cols != grad_output.cols) printf("leakyReLU_backward > input=(%dx%d), grad_output=(%dx%d)\n", input.rows, input.cols, grad_output.rows, grad_output.cols);

    Matrix grad_input = grad_output;

    for (int i = 0; i < input.rows; i++) {
        for (int j = 0; j < input.cols; j++) {
            grad_input.data[i][j] *= (input.data[i][j] > 0) ? 1.0 : alpha;
        }
    }
    return grad_input;
}

inline Matrix softMax(const Matrix& input) {    // Assume input is a column vector
    Matrix output = input;
    double max_val = input.data[0][0];
    for (int i = 1; i < input.rows; i++) {
        max_val = std::max(max_val, input.data[i][0]);
    }

    double sum_exp = 0.0;
    for (int i = 0; i < input.rows; i++) {
        output.data[i][0] = exp(input.data[i][0] - max_val);    // Prevent overflow by dividing by e^(max_val)
        sum_exp += output.data[i][0];
    }

    for (int i = 0; i < input.rows; i++) {
        output.data[i][0] /= sum_exp;
    }

    return output;
}

#endif