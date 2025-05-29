#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#define EPSILON 1e-8

#include <cmath>
#include <memory>

class Optimizer {
  public:
    double learning_rate;
    Optimizer(double lr) : learning_rate(lr) {}
    virtual ~Optimizer() = default;

    virtual void update(Matrix& param, const Matrix& grad) = 0;
    virtual void update(double& param, double grad) = 0;
    virtual std::unique_ptr<Optimizer> clone() const = 0;
};

class SGD : public Optimizer {
  public:
    SGD(double lr = 0.01) : Optimizer(lr) {}
    
    void update(Matrix& param, const Matrix& grad) override {
        param -= learning_rate * grad;
    }

    void update(double& param, double grad) override {
        param -= learning_rate * grad;
    }

    std::unique_ptr<Optimizer> clone() const override {
        return std::make_unique<SGD>(learning_rate);
    }
};

class Momentum : public Optimizer {
  private:
    double alpha;
    Matrix velocity_matrix;
    double velocity_scalar;

  public:
    Momentum(double lr = 0.01, double alpha = 0.9) : Optimizer(lr), alpha(alpha), velocity_scalar(0.0) {}

    void update(Matrix& param, const Matrix& grad) override {
        if (velocity_matrix.rows == 0) velocity_matrix = Matrix(param.rows, param.cols);
        velocity_matrix = (alpha * velocity_matrix) - (learning_rate * grad);
        param += velocity_matrix;
    }
    
    void update(double& param, double grad) override {
        velocity_scalar = (alpha * velocity_scalar) - (learning_rate * grad);
        param += velocity_scalar;
    }

    std::unique_ptr<Optimizer> clone() const override {
        return std::make_unique<Momentum>(learning_rate, alpha);
    }
};

class AdaGrad : public Optimizer {
  private:
    Matrix G_matrix;
    double G_scalar;
  
  public:
    AdaGrad(double lr = 0.01) : Optimizer(lr), G_scalar(0.0) {}
    
    void update(Matrix& param, const Matrix& grad) override {
        if (G_matrix.rows == 0) G_matrix = Matrix(param.rows, param.cols);
        G_matrix += grad.hadamard_power(2);
        param -= (learning_rate / (G_matrix.hadamard_power(0.5) + EPSILON)) % grad;
    }

    void update(double& param, double grad) override {
        G_scalar += std::pow(grad, 2);
        param -= (learning_rate / (std::sqrt(G_scalar) + EPSILON)) * grad;
    }

    std::unique_ptr<Optimizer> clone() const override {
      return std::make_unique<AdaGrad>(learning_rate);
    }
};

class RMSProp : public Optimizer {
  private:
    double beta;
    Matrix EMA_G_matrix;
    double EMA_G_scalar;
  
  public:
    RMSProp(double lr = 0.01, double beta = 0.9) : Optimizer(lr), beta(beta), EMA_G_scalar(0.0) {}

    void update(Matrix& param, const Matrix& grad) override {
        if (EMA_G_matrix.rows == 0) EMA_G_matrix = Matrix(param.rows, param.cols);
        EMA_G_matrix = (beta * EMA_G_matrix) + ((1 - beta) * grad.hadamard_power(2));
        param -= (learning_rate / (EMA_G_matrix.hadamard_power(0.5) + EPSILON)) % grad;
    }

    void update(double& param, double grad) override {
        EMA_G_scalar = (beta * EMA_G_scalar) + ((1 - beta) * std::pow(grad, 2));
        param -= (learning_rate / (std::sqrt(EMA_G_scalar) + EPSILON)) * grad;
    }

    std::unique_ptr<Optimizer> clone() const override {
      return std::make_unique<RMSProp>(learning_rate, beta);
    }
};

class Adam : public Optimizer {
  private:
    double beta1, beta2;
    Matrix m_matrix, v_matrix, m_hat_matrix, v_hat_matrix;
    double m_scalar, v_scalar, m_hat_scalar, v_hat_scalar;
    int t;
  
  public:
    Adam(double lr = 0.01, double beta1 = 0.9, double beta2 = 0.999) : Optimizer(lr), beta1(beta1), beta2(beta2), m_scalar(0.0), v_scalar(0.0), t(0) {}

    void update(Matrix& param, const Matrix& grad) override {
        if (m_matrix.rows == 0) m_matrix = Matrix(param.rows, param.cols);
        if (v_matrix.rows == 0) v_matrix = Matrix(param.rows, param.cols);
        t++;
        m_matrix = (beta1 * m_matrix) + ((1 - beta1) * grad);
        v_matrix = (beta2 * v_matrix) + ((1 - beta2) * grad.hadamard_power(2));
        m_hat_matrix = m_matrix / (1 - std::pow(beta1, t));
        v_hat_matrix = v_matrix / (1 - std::pow(beta2, t));
        param -= (learning_rate / (v_hat_matrix.hadamard_power(0.5) + EPSILON)) % m_hat_matrix;
    }

    void update(double& param, double grad) override {
        t++;
        m_scalar = (beta1 * m_scalar) + ((1 - beta1) * grad);
        v_scalar = (beta2 * v_scalar) + ((1 - beta2) * std::pow(grad, 2));
        m_hat_scalar = m_scalar / (1 - std::pow(beta1, t));
        v_hat_scalar = v_scalar / (1 - std::pow(beta2, t));
        param -= (learning_rate / (std::sqrt(v_hat_scalar) + EPSILON)) * m_hat_scalar;
    }

    std::unique_ptr<Optimizer> clone() const override {
        return std::make_unique<Adam>(learning_rate, beta1, beta2);
    }
};

#endif