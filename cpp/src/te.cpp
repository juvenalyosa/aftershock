#include "pert/te.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <sstream>

namespace pert {
namespace {

constexpr double kTiny = 1e-12;
constexpr double kPi = 3.141592653589793238462643383279502884;

void require_shape(bool ok, const std::string& message) {
    if (!ok) {
        throw NumericError(message);
    }
}

Matrix slice_rows(const Matrix& x, std::size_t start, std::size_t end) {
    require_shape(start <= end && end <= x.rows(), "Invalid row slice.");
    Matrix out(end - start, x.cols());
    for (std::size_t r = start; r < end; ++r) {
        for (std::size_t c = 0; c < x.cols(); ++c) {
            out(r - start, c) = x(r, c);
        }
    }
    return out;
}

Matrix transpose(const Matrix& x) {
    Matrix out(x.cols(), x.rows());
    for (std::size_t r = 0; r < x.rows(); ++r) {
        for (std::size_t c = 0; c < x.cols(); ++c) {
            out(c, r) = x(r, c);
        }
    }
    return out;
}

Matrix multiply(const Matrix& a, const Matrix& b) {
    require_shape(a.cols() == b.rows(), "Matrix multiply dimension mismatch.");
    Matrix out(a.rows(), b.cols());
    for (std::size_t i = 0; i < a.rows(); ++i) {
        for (std::size_t k = 0; k < a.cols(); ++k) {
            const double aik = a(i, k);
            if (std::abs(aik) <= 0.0) {
                continue;
            }
            for (std::size_t j = 0; j < b.cols(); ++j) {
                out(i, j) += aik * b(k, j);
            }
        }
    }
    return out;
}

Matrix subtract(const Matrix& a, const Matrix& b) {
    require_shape(a.rows() == b.rows() && a.cols() == b.cols(), "Matrix subtract dimension mismatch.");
    Matrix out(a.rows(), a.cols());
    for (std::size_t i = 0; i < a.data().size(); ++i) {
        out.data()[i] = a.data()[i] - b.data()[i];
    }
    return out;
}

Matrix solve_linear(Matrix a, Matrix b) {
    require_shape(a.rows() == a.cols(), "Linear solve requires a square coefficient matrix.");
    require_shape(a.rows() == b.rows(), "Linear solve RHS dimension mismatch.");

    const std::size_t n = a.rows();
    const std::size_t rhs_cols = b.cols();
    for (std::size_t col = 0; col < n; ++col) {
        std::size_t pivot = col;
        double best = std::abs(a(col, col));
        for (std::size_t r = col + 1; r < n; ++r) {
            const double value = std::abs(a(r, col));
            if (value > best) {
                best = value;
                pivot = r;
            }
        }
        if (best < kTiny) {
            throw NumericError("Singular matrix in linear solve.");
        }
        if (pivot != col) {
            for (std::size_t c = 0; c < n; ++c) {
                std::swap(a(col, c), a(pivot, c));
            }
            for (std::size_t c = 0; c < rhs_cols; ++c) {
                std::swap(b(col, c), b(pivot, c));
            }
        }

        const double diag = a(col, col);
        for (std::size_t c = col; c < n; ++c) {
            a(col, c) /= diag;
        }
        for (std::size_t c = 0; c < rhs_cols; ++c) {
            b(col, c) /= diag;
        }

        for (std::size_t r = 0; r < n; ++r) {
            if (r == col) {
                continue;
            }
            const double factor = a(r, col);
            if (std::abs(factor) <= 0.0) {
                continue;
            }
            for (std::size_t c = col; c < n; ++c) {
                a(r, c) -= factor * a(col, c);
            }
            for (std::size_t c = 0; c < rhs_cols; ++c) {
                b(r, c) -= factor * b(col, c);
            }
        }
    }
    return b;
}

std::pair<Matrix, Matrix> fit_ridge_multivariate(const Matrix& x, const Matrix& y, double ridge) {
    require_shape(x.rows() == y.rows(), "Ridge regression row mismatch.");
    Matrix xt = transpose(x);
    Matrix xtx = multiply(xt, x);
    for (std::size_t i = 0; i < std::min(xtx.rows(), xtx.cols()); ++i) {
        xtx(i, i) += ridge;
    }
    Matrix xty = multiply(xt, y);
    Matrix beta = solve_linear(xtx, xty);
    Matrix pred = multiply(x, beta);
    Matrix resid = subtract(y, pred);
    return {beta, resid};
}

double mean_column(const Matrix& x, std::size_t col) {
    double total = 0.0;
    for (std::size_t r = 0; r < x.rows(); ++r) {
        total += x(r, col);
    }
    return total / static_cast<double>(std::max<std::size_t>(1, x.rows()));
}

Matrix residual_cov(const Matrix& resid) {
    require_shape(resid.rows() >= 2, "At least two residual rows are required for covariance.");
    Matrix cov(resid.cols(), resid.cols());
    std::vector<double> means(resid.cols(), 0.0);
    for (std::size_t c = 0; c < resid.cols(); ++c) {
        means[c] = mean_column(resid, c);
    }
    const double denom = static_cast<double>(resid.rows() - 1);
    for (std::size_t i = 0; i < resid.cols(); ++i) {
        for (std::size_t j = i; j < resid.cols(); ++j) {
            double total = 0.0;
            for (std::size_t r = 0; r < resid.rows(); ++r) {
                total += (resid(r, i) - means[i]) * (resid(r, j) - means[j]);
            }
            const double value = total / denom;
            cov(i, j) = value;
            cov(j, i) = value;
        }
    }
    for (std::size_t i = 0; i < cov.rows(); ++i) {
        cov(i, i) += 1e-10;
    }
    return cov;
}

std::pair<int, double> slogdet(Matrix a) {
    require_shape(a.rows() == a.cols(), "slogdet requires a square matrix.");
    const std::size_t n = a.rows();
    int sign = 1;
    double logdet = 0.0;
    for (std::size_t col = 0; col < n; ++col) {
        std::size_t pivot = col;
        double best = std::abs(a(col, col));
        for (std::size_t r = col + 1; r < n; ++r) {
            const double value = std::abs(a(r, col));
            if (value > best) {
                best = value;
                pivot = r;
            }
        }
        if (best < kTiny) {
            return {0, -std::numeric_limits<double>::infinity()};
        }
        if (pivot != col) {
            for (std::size_t c = 0; c < n; ++c) {
                std::swap(a(col, c), a(pivot, c));
            }
            sign *= -1;
        }
        const double diag = a(col, col);
        if (diag < 0.0) {
            sign *= -1;
        }
        logdet += std::log(std::abs(diag));
        for (std::size_t r = col + 1; r < n; ++r) {
            const double factor = a(r, col) / diag;
            for (std::size_t c = col + 1; c < n; ++c) {
                a(r, c) -= factor * a(col, c);
            }
        }
    }
    return {sign, logdet};
}

double bic_from_residual_cov(const Matrix& resid, std::size_t n_params) {
    Matrix cov = residual_cov(resid);
    auto [sign, logdet] = slogdet(cov);
    if (sign <= 0 || !std::isfinite(logdet)) {
        logdet = 1e6;
    }
    const double n = static_cast<double>(resid.rows());
    const double m = static_cast<double>(resid.cols());
    const double ll = -0.5 * n * (m * std::log(2.0 * kPi) + logdet + m);
    return -2.0 * ll + static_cast<double>(n_params) * std::log(std::max(2.0, n));
}

double max_abs_resid_autocorr(const Matrix& resid, int lag_max) {
    double worst = 0.0;
    for (std::size_t c = 0; c < resid.cols(); ++c) {
        const double mean = mean_column(resid, c);
        double var = 0.0;
        for (std::size_t r = 0; r < resid.rows(); ++r) {
            const double centered = resid(r, c) - mean;
            var += centered * centered;
        }
        var /= static_cast<double>(std::max<std::size_t>(1, resid.rows()));
        if (var <= 1e-12) {
            continue;
        }
        const int max_lag = std::min<int>(lag_max, static_cast<int>(resid.rows()) - 1);
        for (int lag = 1; lag <= max_lag; ++lag) {
            double corr = 0.0;
            for (std::size_t r = 0; r + static_cast<std::size_t>(lag) < resid.rows(); ++r) {
                corr += (resid(r, c) - mean) * (resid(r + static_cast<std::size_t>(lag), c) - mean);
            }
            corr /= static_cast<double>(resid.rows() - static_cast<std::size_t>(lag));
            worst = std::max(worst, std::abs(corr / var));
        }
    }
    return worst;
}

double blocked_cv_mse(const Matrix& x, const Matrix& y, double ridge, int folds = 3) {
    const std::size_t n = x.rows();
    if (n < static_cast<std::size_t>(folds + 5)) {
        auto fitted = fit_ridge_multivariate(x, y, ridge);
        const Matrix& resid = fitted.second;
        double total = 0.0;
        for (double v : resid.data()) {
            total += v * v;
        }
        return total / static_cast<double>(std::max<std::size_t>(1, resid.data().size()));
    }

    double total_mse = 0.0;
    int used = 0;
    for (int fold = 0; fold < folds; ++fold) {
        const std::size_t a = static_cast<std::size_t>(fold) * n / static_cast<std::size_t>(folds);
        const std::size_t b = static_cast<std::size_t>(fold + 1) * n / static_cast<std::size_t>(folds);
        if (b <= a + 1) {
            continue;
        }

        Matrix x_train(n - (b - a), x.cols());
        Matrix y_train(n - (b - a), y.cols());
        Matrix x_valid(b - a, x.cols());
        Matrix y_valid(b - a, y.cols());

        std::size_t tr = 0;
        std::size_t va = 0;
        for (std::size_t r = 0; r < n; ++r) {
            const bool is_valid = (r >= a && r < b);
            Matrix& x_dst = is_valid ? x_valid : x_train;
            Matrix& y_dst = is_valid ? y_valid : y_train;
            std::size_t& dst_row = is_valid ? va : tr;
            for (std::size_t c = 0; c < x.cols(); ++c) {
                x_dst(dst_row, c) = x(r, c);
            }
            for (std::size_t c = 0; c < y.cols(); ++c) {
                y_dst(dst_row, c) = y(r, c);
            }
            ++dst_row;
        }

        auto fitted = fit_ridge_multivariate(x_train, y_train, ridge);
        Matrix pred = multiply(x_valid, fitted.first);
        Matrix resid = subtract(y_valid, pred);
        double mse = 0.0;
        for (double v : resid.data()) {
            mse += v * v;
        }
        total_mse += mse / static_cast<double>(std::max<std::size_t>(1, resid.data().size()));
        ++used;
    }
    if (used == 0) {
        auto fitted = fit_ridge_multivariate(x, y, ridge);
        double mse = 0.0;
        for (double v : fitted.second.data()) {
            mse += v * v;
        }
        return mse / static_cast<double>(std::max<std::size_t>(1, fitted.second.data().size()));
    }
    return total_mse / static_cast<double>(used);
}

double spectral_norm_estimate(const Matrix& beta_full, std::size_t y_dim, int p) {
    if (p < 1 || y_dim == 0) {
        return 0.0;
    }
    const std::size_t dim = y_dim * static_cast<std::size_t>(p);
    Matrix companion(dim, dim);
    for (int lag = 0; lag < p; ++lag) {
        const std::size_t row_offset = static_cast<std::size_t>(lag) * y_dim;
        for (std::size_t i = 0; i < y_dim; ++i) {
            for (std::size_t j = 0; j < y_dim; ++j) {
                if (row_offset + j < beta_full.rows() && i < beta_full.cols()) {
                    companion(i, row_offset + j) = beta_full(row_offset + j, i);
                }
            }
        }
    }
    if (p > 1) {
        for (std::size_t i = y_dim; i < dim; ++i) {
            companion(i, i - y_dim) = 1.0;
        }
    }

    std::vector<double> v(dim, 1.0 / std::sqrt(static_cast<double>(dim)));
    for (int iter = 0; iter < 60; ++iter) {
        std::vector<double> av(dim, 0.0);
        std::vector<double> atav(dim, 0.0);
        for (std::size_t r = 0; r < dim; ++r) {
            for (std::size_t c = 0; c < dim; ++c) {
                av[r] += companion(r, c) * v[c];
            }
        }
        for (std::size_t r = 0; r < dim; ++r) {
            for (std::size_t c = 0; c < dim; ++c) {
                atav[c] += companion(r, c) * av[r];
            }
        }
        double norm = 0.0;
        for (double value : atav) {
            norm += value * value;
        }
        norm = std::sqrt(norm);
        if (norm <= kTiny) {
            return 0.0;
        }
        for (std::size_t i = 0; i < dim; ++i) {
            v[i] = atav[i] / norm;
        }
    }
    std::vector<double> av(dim, 0.0);
    for (std::size_t r = 0; r < dim; ++r) {
        for (std::size_t c = 0; c < dim; ++c) {
            av[r] += companion(r, c) * v[c];
        }
    }
    double norm = 0.0;
    for (double value : av) {
        norm += value * value;
    }
    return std::sqrt(norm);
}

Matrix add_intercept(const Matrix& x) {
    Matrix out(x.rows(), x.cols() + 1);
    for (std::size_t r = 0; r < x.rows(); ++r) {
        out(r, 0) = 1.0;
        for (std::size_t c = 0; c < x.cols(); ++c) {
            out(r, c + 1) = x(r, c);
        }
    }
    return out;
}

double residual_variance_univariate(const Matrix& x, const Matrix& y, double ridge) {
    auto fitted = fit_ridge_multivariate(x, y, ridge);
    const Matrix& resid = fitted.second;
    double mean = 0.0;
    for (double value : resid.data()) {
        mean += value;
    }
    mean /= static_cast<double>(std::max<std::size_t>(1, resid.data().size()));
    double var = 0.0;
    for (double value : resid.data()) {
        const double centered = value - mean;
        var += centered * centered;
    }
    var /= static_cast<double>(std::max<std::size_t>(1, resid.data().size()));
    return std::max(var, 1e-10);
}

}  // namespace

Matrix::Matrix(std::size_t rows, std::size_t cols, double value)
    : rows_(rows), cols_(cols), data_(rows * cols, value) {}

double& Matrix::operator()(std::size_t row, std::size_t col) {
    return data_.at(row * cols_ + col);
}

double Matrix::operator()(std::size_t row, std::size_t col) const {
    return data_.at(row * cols_ + col);
}

Matrix lag_matrix(const Matrix& series, int order) {
    require_shape(order >= 1, "Lag order must be positive.");
    require_shape(series.rows() > static_cast<std::size_t>(order), "Not enough rows for lag matrix.");
    const std::size_t rows = series.rows() - static_cast<std::size_t>(order);
    Matrix out(rows, static_cast<std::size_t>(order) * series.cols());
    for (std::size_t r = 0; r < rows; ++r) {
        for (int lag = 1; lag <= order; ++lag) {
            const std::size_t src_row = static_cast<std::size_t>(order - lag) + r;
            const std::size_t col_offset = static_cast<std::size_t>(lag - 1) * series.cols();
            for (std::size_t c = 0; c < series.cols(); ++c) {
                out(r, col_offset + c) = series(src_row, c);
            }
        }
    }
    return out;
}

Matrix hstack(const std::vector<Matrix>& matrices) {
    if (matrices.empty()) {
        return Matrix();
    }
    const std::size_t rows = matrices.front().rows();
    std::size_t cols = 0;
    for (const Matrix& matrix : matrices) {
        require_shape(matrix.rows() == rows, "hstack row mismatch.");
        cols += matrix.cols();
    }
    Matrix out(rows, cols);
    std::size_t offset = 0;
    for (const Matrix& matrix : matrices) {
        for (std::size_t r = 0; r < rows; ++r) {
            for (std::size_t c = 0; c < matrix.cols(); ++c) {
                out(r, offset + c) = matrix(r, c);
            }
        }
        offset += matrix.cols();
    }
    return out;
}

double te_with_fixed_model(
    const Matrix& target,
    const Matrix& driver,
    const std::optional<Matrix>& conditioning,
    int order,
    double ridge
) {
    const std::size_t n = std::min(
        target.rows(),
        std::min(driver.rows(), conditioning ? conditioning->rows() : target.rows())
    );
    require_shape(n > static_cast<std::size_t>(order + 2), "Insufficient rows for TE model.");
    Matrix y = slice_rows(target, target.rows() - n, target.rows());
    Matrix x = slice_rows(driver, driver.rows() - n, driver.rows());
    std::optional<Matrix> w;
    if (conditioning) {
        w = slice_rows(*conditioning, conditioning->rows() - n, conditioning->rows());
    }

    Matrix y_target = slice_rows(y, static_cast<std::size_t>(order), y.rows());
    std::vector<Matrix> restricted_parts = {lag_matrix(y, order)};
    if (w && w->cols() > 0) {
        restricted_parts.push_back(lag_matrix(*w, order));
    }
    Matrix xr = hstack(restricted_parts);
    Matrix xf = hstack({xr, lag_matrix(x, order)});

    auto restricted = fit_ridge_multivariate(xr, y_target, ridge);
    auto full = fit_ridge_multivariate(xf, y_target, ridge);
    auto [sign_r, logdet_r] = slogdet(residual_cov(restricted.second));
    auto [sign_f, logdet_f] = slogdet(residual_cov(full.second));
    if (sign_r <= 0 || sign_f <= 0) {
        throw NumericError("Residual covariance determinant is non-positive.");
    }
    return 0.5 * (logdet_r - logdet_f);
}

TEResult conditional_te_linear_gaussian(
    const Matrix& target,
    const Matrix& driver,
    const std::optional<Matrix>& conditioning,
    const TEConfig& config
) {
    require_shape(!config.ridge_values.empty(), "At least one ridge value is required.");
    const std::size_t n = std::min(
        target.rows(),
        std::min(driver.rows(), conditioning ? conditioning->rows() : target.rows())
    );
    Matrix y = slice_rows(target, target.rows() - n, target.rows());
    Matrix x = slice_rows(driver, driver.rows() - n, driver.rows());
    std::optional<Matrix> w;
    if (conditioning) {
        w = slice_rows(*conditioning, conditioning->rows() - n, conditioning->rows());
    }

    std::vector<ModelCandidate> candidates;
    for (int p = 1; p <= config.max_order; ++p) {
        if (n <= static_cast<std::size_t>(p + 5)) {
            continue;
        }
        Matrix y_target = slice_rows(y, static_cast<std::size_t>(p), y.rows());
        std::vector<Matrix> restricted_parts = {lag_matrix(y, p)};
        if (w && w->cols() > 0) {
            restricted_parts.push_back(lag_matrix(*w, p));
        }
        Matrix xr = hstack(restricted_parts);
        Matrix xf = hstack({xr, lag_matrix(x, p)});

        double best_ridge = config.ridge_values.front();
        double best_cv = std::numeric_limits<double>::infinity();
        for (double ridge : config.ridge_values) {
            const double cv = blocked_cv_mse(xf, y_target, ridge, 3);
            if (cv < best_cv) {
                best_cv = cv;
                best_ridge = ridge;
            }
        }

        auto restricted = fit_ridge_multivariate(xr, y_target, best_ridge);
        auto full = fit_ridge_multivariate(xf, y_target, best_ridge);
        auto [sign_r, logdet_r] = slogdet(residual_cov(restricted.second));
        auto [sign_f, logdet_f] = slogdet(residual_cov(full.second));
        if (sign_r <= 0 || sign_f <= 0) {
            continue;
        }

        ModelCandidate candidate;
        candidate.p = p;
        candidate.ridge = best_ridge;
        candidate.te = 0.5 * (logdet_r - logdet_f);
        candidate.bic = bic_from_residual_cov(full.second, full.first.data().size());
        candidate.whiteness_max_abs_acf = max_abs_resid_autocorr(full.second, config.whiteness_lag);
        candidate.spectral_radius_estimate = spectral_norm_estimate(full.first, y.cols(), p);
        candidate.accepted =
            candidate.whiteness_max_abs_acf <= config.whiteness_threshold &&
            candidate.spectral_radius_estimate < 1.0;
        candidates.push_back(candidate);
    }

    if (candidates.empty()) {
        throw NumericError("Insufficient data for VAR/TE estimation.");
    }

    std::vector<ModelCandidate> accepted;
    for (const ModelCandidate& candidate : candidates) {
        if (candidate.accepted) {
            accepted.push_back(candidate);
        }
    }
    const std::vector<ModelCandidate>& pool = accepted.empty() ? candidates : accepted;
    const ModelCandidate& chosen = *std::min_element(
        pool.begin(),
        pool.end(),
        [](const ModelCandidate& a, const ModelCandidate& b) { return a.bic < b.bic; }
    );

    TEResult result;
    result.selected_order = chosen.p;
    result.selected_ridge = chosen.ridge;
    result.te_driver_to_target = te_with_fixed_model(y, x, w, chosen.p, chosen.ridge);
    result.te_target_to_driver = te_with_fixed_model(x, y, w, chosen.p, chosen.ridge);
    result.directionality_delta = result.te_driver_to_target - result.te_target_to_driver;
    result.model_candidates = std::move(candidates);
    return result;
}

Matrix pairwise_te_matrix(const Matrix& signal, int lag, double ridge) {
    require_shape(lag >= 1, "Pairwise TE lag must be positive.");
    const std::size_t t = signal.rows();
    const std::size_t n = signal.cols();
    Matrix te(n, n);
    if (t <= static_cast<std::size_t>(lag + 3)) {
        return te;
    }

    Matrix y_all = slice_rows(signal, static_cast<std::size_t>(lag), t);
    Matrix yp_all = slice_rows(signal, 0, t - static_cast<std::size_t>(lag));
    Matrix ones(yp_all.rows(), 1, 1.0);

    for (std::size_t target = 0; target < n; ++target) {
        Matrix y(yp_all.rows(), 1);
        Matrix yp(yp_all.rows(), 1);
        for (std::size_t r = 0; r < yp_all.rows(); ++r) {
            y(r, 0) = y_all(r, target);
            yp(r, 0) = yp_all(r, target);
        }
        Matrix xr = hstack({ones, yp});
        const double var_r = residual_variance_univariate(xr, y, ridge);
        for (std::size_t source = 0; source < n; ++source) {
            if (source == target) {
                continue;
            }
            Matrix xp(yp_all.rows(), 1);
            for (std::size_t r = 0; r < yp_all.rows(); ++r) {
                xp(r, 0) = yp_all(r, source);
            }
            Matrix xf = hstack({ones, yp, xp});
            const double var_f = residual_variance_univariate(xf, y, ridge);
            const double value = 0.5 * std::log(var_r / var_f);
            if (std::isfinite(value)) {
                te(source, target) = std::max(0.0, value);
            }
        }
    }

    return te;
}

}  // namespace pert
