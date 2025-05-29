#ifndef POOL_H
#define POOL_H

#include "matrix.h"

class PoolLayer {
  private:
    Matrix input;   // Save for backpropagation
    Matrix max_mask;    // Store Max Positions

  public:
    Matrix forward(const Matrix& input) {
        this -> input = input;
        int output_rows = input.rows / 2;
        int output_cols = input.cols / 2;
        Matrix pooled(output_rows, output_cols);
        max_mask = Matrix(input.rows, input.cols);

        for (int i = 0; i < output_rows; i++) {
            for (int j = 0; j < output_cols; j++) {
                double max_val = input.data[i*2 + 0][j*2 + 0];
                int max_row = i*2, max_col = j*2;
                for (int m = 0; m < 2; m++) {
                    for (int n = 0; n < 2; n++) {
                        int row = i * 2 + m;
                        int col = j * 2 + n;
                        if (input.data[row][col] > max_val) {
                            max_row = row;
                            max_col = col;
                            max_val = input.data[row][col];
                        }
                    }
                }
                pooled.data[i][j] = max_val;
                max_mask.data[max_row][max_col] = 1;
            }
        }
        return pooled;
    }

    Matrix backward(const Matrix& grad_output) {
        Matrix grad_input(input.rows, input.cols);
        for (int i = 0; i < grad_output.rows; i++) {
            for (int j = 0; j < grad_output.cols; j++) {
                for (int m = 0; m < 2; m++) {
                    for (int n = 0; n < 2; n++) {
                        int row = i * 2 + m;
                        int col = j * 2 + n;
                        if (row < input.rows && col < input.cols && max_mask.data[row][col] == 1) {
                            grad_input.data[row][col] = grad_output.data[i][j];
                        }
                    }
                }
            }
        }
        return grad_input;
    }

};

#endif