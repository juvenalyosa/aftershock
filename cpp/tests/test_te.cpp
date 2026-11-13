#include "pert/te.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>

namespace {

double noise(int i, double scale) {
    return scale * std::sin(12.9898 * static_cast<double>(i) + 78.233);
}

void require_true(bool value, const std::string& label) {
    if (!value) {
        throw std::runtime_error(label);
    }
}

}  // namespace

int main() {
    constexpr int n = 220;
    pert::Matrix x(n, 1);
    pert::Matrix y(n, 1);
    x(0, 0) = 0.2;
    y(0, 0) = -0.1;
    for (int t = 1; t < n; ++t) {
        x(t, 0) = 0.75 * x(t - 1, 0) + noise(t, 0.12);
        y(t, 0) = 0.25 * y(t - 1, 0) + 0.9 * x(t - 1, 0) + noise(t + 500, 0.08);
    }

    pert::TEConfig cfg;
    cfg.max_order = 3;
    cfg.ridge_values = {1e-8, 1e-6, 1e-4};
    cfg.whiteness_threshold = 1.0;

    pert::TEResult result = pert::conditional_te_linear_gaussian(y, x, std::nullopt, cfg);
    require_true(result.selected_order >= 1, "TE model should select a positive order.");
    require_true(result.te_driver_to_target > 0.05, "Driver-to-target TE should be positive.");
    require_true(
        result.te_driver_to_target > result.te_target_to_driver,
        "Designed x->y system should have stronger x->y TE than y->x TE."
    );

    pert::Matrix signal(n, 2);
    for (int t = 0; t < n; ++t) {
        signal(t, 0) = x(t, 0);
        signal(t, 1) = y(t, 0);
    }
    pert::Matrix pairwise = pert::pairwise_te_matrix(signal, 1);
    require_true(pairwise(0, 1) > pairwise(1, 0), "Pairwise TE should recover x->y direction.");

    std::cout << "te tests passed\n";
    return 0;
}
