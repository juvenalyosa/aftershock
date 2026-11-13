#pragma once

#include <cstddef>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace pert {

class NumericError : public std::runtime_error {
public:
    explicit NumericError(const std::string& message) : std::runtime_error(message) {}
};

class Matrix {
public:
    Matrix() = default;
    Matrix(std::size_t rows, std::size_t cols, double value = 0.0);

    std::size_t rows() const { return rows_; }
    std::size_t cols() const { return cols_; }
    bool empty() const { return rows_ == 0 || cols_ == 0; }

    double& operator()(std::size_t row, std::size_t col);
    double operator()(std::size_t row, std::size_t col) const;

    const std::vector<double>& data() const { return data_; }
    std::vector<double>& data() { return data_; }

private:
    std::size_t rows_ = 0;
    std::size_t cols_ = 0;
    std::vector<double> data_;
};

struct TEConfig {
    int max_order = 6;
    std::vector<double> ridge_values = {1e-8, 1e-6, 1e-4, 1e-2};
    int whiteness_lag = 10;
    double whiteness_threshold = 0.2;
};

struct ModelCandidate {
    int p = 0;
    double ridge = 0.0;
    double te = 0.0;
    double bic = 0.0;
    double whiteness_max_abs_acf = 0.0;
    double spectral_radius_estimate = 0.0;
    bool accepted = false;
};

struct TEResult {
    int selected_order = 0;
    double selected_ridge = 0.0;
    double te_driver_to_target = 0.0;
    double te_target_to_driver = 0.0;
    double directionality_delta = 0.0;
    std::vector<ModelCandidate> model_candidates;
};

Matrix lag_matrix(const Matrix& series, int order);

Matrix hstack(const std::vector<Matrix>& matrices);

double te_with_fixed_model(
    const Matrix& target,
    const Matrix& driver,
    const std::optional<Matrix>& conditioning,
    int order,
    double ridge
);

TEResult conditional_te_linear_gaussian(
    const Matrix& target,
    const Matrix& driver,
    const std::optional<Matrix>& conditioning,
    const TEConfig& config
);

Matrix pairwise_te_matrix(const Matrix& signal, int lag = 1, double ridge = 1e-5);

}  // namespace pert
