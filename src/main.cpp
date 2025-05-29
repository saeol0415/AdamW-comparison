#include <iostream>

#include "matrix.h"
#include "model.h"
#include "dataset.h"
#include "training.h"

int argmax(const Matrix& mat) {
    double max_val = mat.data[0][0];
    int max_idx = 0;
    for (int i = 0; i < mat.rows; i++) {
        for (int j = 0; j < mat.cols; j++) {
            if (mat.data[i][j] > max_val) {
                max_val = mat.data[i][j];
                max_idx = i;
            }
        }
    }
    return max_idx;
}

int main() {
    Dataset train_data, test_data;
    train_data.loadCSV("C:\\Users\\saeol\\Desktop\\C Projects\\CNN\\data\\mnist_train_1k.csv", 28, 28, 10);
    test_data.loadCSV("C:\\Users\\saeol\\Desktop\\C Projects\\CNN\\data\\mnist_test.csv", 28, 28, 10);
    train_data.normalize_dataset();
    test_data.normalize_dataset();

    int epochs = 100;
    double lr = 0.01;
    auto sgd = std::make_unique<SGD>(lr);
    Model model(std::move(sgd));

    TrainingResult result = trainDataset(model, train_data, epochs, 0.05);
    model.save("trained_model.txt");

    int correct = 0;
    for (size_t i = 0; i < test_data.size(); i++) {
        const auto& [input, target] = test_data[i];
        auto [logits, loss] = model.forward(input, target);
        Matrix probs = softMax(logits);
        int guess = argmax(probs);
        int label = argmax(target);
        if (guess == label) correct++;
    }

    double accuracy = static_cast<double>(correct) / test_data.size();
    std::cout << "Test Accuracy: " << accuracy * 100.0 << "%\n";

    std::cout << "Completed" << std::endl;
}