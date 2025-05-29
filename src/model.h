#ifndef MODEL_H
#define MODEL_H

#include <vector>
#include <fstream>
#include <memory>

#include "conv.h"
#include "pool.h"
#include "fc.h"
#include "activation.h"
#include "loss.h"
#include "optimizer.h"

class Model {
  private:
    ConvLayer conv1;
    ConvLayer conv2;
    PoolLayer pool1;
    PoolLayer pool2;
    FCLayer fc1;
    FCLayer fc2;
    std::vector<Matrix> intermediates;  // conv1_z(0-7), conv2_z(8-15), pool2_out(16-23), fc1_z(24)

  public:
    Model(std::unique_ptr<Optimizer> opt) 
    : conv1(1, 8, std::move(opt->clone())),
      pool1(),
      conv2(8, 8, std::move(opt->clone())),
      pool2(),
      fc1(8 * 7 * 7, 32, std::move(opt->clone())),
      fc2(32, 10, std::move(opt->clone())) {}

    std::pair<Matrix, double> forward(const Matrix& input, const Matrix& target) {
        intermediates.clear();
        std::vector<Matrix> x = {input};
        std::vector<Matrix> conv1_z = conv1.forward(x);
        intermediates.insert(intermediates.end(), conv1_z.begin(), conv1_z.end());

        for (auto& m : conv1_z) m = leakyReLU(m);
        for (auto& m : conv1_z) m = pool1.forward(m);

        std::vector<Matrix> conv2_z = conv2.forward(conv1_z);
        intermediates.insert(intermediates.end(), conv2_z.begin(), conv2_z.end());

        for (auto& m : conv2_z) m = leakyReLU(m);
        for (auto& m : conv2_z) {
            Matrix pooled = pool2.forward(m);
            intermediates.push_back(pooled);
        }

        Matrix flat(8 * 7 * 7, 1);
        int k = 0;
        for (int i = 16; i < 24; i++) {
            Matrix f = intermediates[i].flatten();
            for (int j = 0; j < f.rows; j++) flat.data[k++][0] = f.data[j][0];
        }

        Matrix fc1_z = fc1.forward(flat);
        intermediates.push_back(fc1_z);
        Matrix h1 = leakyReLU(fc1_z);

        Matrix logits = fc2.forward(h1);

        double loss = crossEntropyLoss(softMax(logits), target);
        return {logits, loss};
    }

    void backward(const Matrix& logits, const Matrix& target) {
        Matrix probs = softMax(logits);
        Matrix grad = probs - target;

        grad = fc2.backward(grad);
        grad = leakyReLU_backward(intermediates[24], grad);
        grad = fc1.backward(grad);

        // unflatten
        std::vector<Matrix> grad_x(8, Matrix(7, 7));
        int k = 0;
        for (int ch = 0; ch < 8; ++ch)
            for (int i = 0; i < 7; ++i)
                for (int j = 0; j < 7; ++j)
                    grad_x[ch].data[i][j] = grad.data[k++][0];

        // pool2 backward
        for (int i = 0; i < 8; ++i)
            grad_x[i] = pool2.backward(grad_x[i]);

        // ReLU backward from conv2
        for (int i = 0; i < 8; ++i)
            grad_x[i] = leakyReLU_backward(intermediates[8 + i], grad_x[i]);

        grad_x = conv2.backward(grad_x);

        // pool1 backward
        for (int i = 0; i < 8; ++i)
            grad_x[i] = pool1.backward(grad_x[i]);

        // ReLU backward from conv1
        for (int i = 0; i < 8; ++i)
            grad_x[i] = leakyReLU_backward(intermediates[i], grad_x[i]);

        grad_x = conv1.backward(grad_x);
    }

    void update() {
        conv1.update();
        conv2.update();
        fc1.update();
        fc2.update();
    }

    void save(const std::string& filename) const {
        std::ofstream ofs(filename);
        if (!ofs) throw std::runtime_error("Cannot open file for saving.");
    
        // Save convolutional layers
        ofs << "Convolutional Layers: 2" << "\n\n";

        ofs << "Layer 1 Filters: " << conv1.num_kernels << "\n";
        ofs << "Input Channels: " << conv1.kernels[0].size() << "\n\n";
    
        // Save kernels for each filter and channel
        for (int filter = 0; filter < conv1.num_kernels; filter++) {
            ofs << "Filter " << filter << " Kernels:\n";
            for (size_t channel = 0; channel < conv1.kernels[filter].size(); channel++) {
                ofs << "Channel " << channel << ":\n";
                const Matrix& kernel = conv1.kernels[filter][channel];
                for (int i = 0; i < kernel.rows; i++) {
                    for (int j = 0; j < kernel.cols; j++) {
                        ofs << kernel.data[i][j] << " ";
                    }
                    ofs << "\n";
                }
                ofs << "\n";
            }
        }
        ofs << "Bias: " << conv1.bias << "\n\n";

        ofs << "Layer 2 Filters: " << conv2.num_kernels << "\n";
        ofs << "Input Channels: " << conv2.kernels[0].size() << "\n\n";
    
        // Save kernels for each filter and channel
        for (int filter = 0; filter < conv2.num_kernels; filter++) {
            ofs << "Filter " << filter << " Kernels:\n";
            for (size_t channel = 0; channel < conv2.kernels[filter].size(); channel++) {
                ofs << "Channel " << channel << ":\n";
                const Matrix& kernel = conv2.kernels[filter][channel];
                for (int i = 0; i < kernel.rows; i++) {
                    for (int j = 0; j < kernel.cols; j++) {
                        ofs << kernel.data[i][j] << " ";
                    }
                    ofs << "\n";
                }
                ofs << "\n";
            }
        }
        ofs << "Bias: " << conv2.bias << "\n\n";
    

        // Save fully connected layer
        ofs << "FC Layers: 2" << "\n\n";
        
        ofs << "FC 1 Weights: " << fc1.weights.rows << "x" << fc1.weights.cols << "\n";
        for (int i = 0; i < fc1.weights.rows; i++) {
            for (int j = 0; j < fc1.weights.cols; j++) {
                ofs << fc1.weights.data[i][j] << " ";
            }
            ofs << "\n";
        }
    
        ofs << "\nFC 1 Bias: " << fc1.bias.rows << "\n";
        for (int i = 0; i < fc1.bias.rows; i++) {
            ofs << fc1.bias.data[i][0] << " ";
        }
        ofs << "\n";
    
        ofs << "FC 2 Weights: " << fc2.weights.rows << "x" << fc2.weights.cols << "\n";
        for (int i = 0; i < fc2.weights.rows; i++) {
            for (int j = 0; j < fc2.weights.cols; j++) {
                ofs << fc2.weights.data[i][j] << " ";
            }
            ofs << "\n";
        }
    
        ofs << "\nFC 2 Bias: " << fc2.bias.rows << "\n";
        for (int i = 0; i < fc2.bias.rows; i++) {
            ofs << fc2.bias.data[i][0] << " ";
        }
        ofs << "\n";

        ofs.close();
    }
};

#endif