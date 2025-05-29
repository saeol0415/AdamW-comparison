#ifndef LOSS_H
#define LOSS_H

#define EPSILON 1e-8

#include <cmath>

#include "matrix.h"

inline double crossEntropyLoss(const Matrix& y_pred, const Matrix& y_true) {
    double loss = 0.0;
    for (int i = 0; i < y_pred.rows; i++) {
        loss -= y_true.data[i][0] * log(y_pred.data[i][0] + EPSILON);  // Prevent log(0)
    }
    return loss;
}

#endif