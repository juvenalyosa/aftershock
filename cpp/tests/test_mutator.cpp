#include "pert/mutator.hpp"

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require_equal(const std::string& got, const std::string& expected, const std::string& label) {
    if (got != expected) {
        throw std::runtime_error(label + ": expected '" + expected + "', got '" + got + "'");
    }
}

void require_true(bool value, const std::string& label) {
    if (!value) {
        throw std::runtime_error(label);
    }
}

}  // namespace

int main() {
    using pert::mutate_sequence;
    using pert::needleman_wunsch_affine;

    require_equal(mutate_sequence("AKST", {"K2R"}, 1), "ARST", "SNP");
    require_equal(mutate_sequence("ABCDE", {"C3del"}, 1), "ABDE", "single deletion");
    require_equal(mutate_sequence("ABCDEF", {"2_4del"}, 1), "AEF", "range deletion");
    require_equal(mutate_sequence("ABCD", {"B2_C3insXX"}, 1), "ABXXCD", "insertion");
    require_equal(mutate_sequence("ABCDE", {"2_4delinsXX"}, 1), "AXXE", "delins");
    require_equal(mutate_sequence("ABCDE", {"C3dup"}, 1), "ABCCDE", "single duplication");
    require_equal(mutate_sequence("ABCDE", {"2_3dup"}, 1), "ABCBCDE", "range duplication");

    bool threw = false;
    try {
        (void)mutate_sequence("AKST", {"A2R"}, 1);
    } catch (const pert::MutationError&) {
        threw = true;
    }
    require_true(threw, "SNP mismatch should throw");

    auto alignment = needleman_wunsch_affine("ABCD", "ABXXCD");
    require_equal(alignment.first, "AB--CD", "alignment original");
    require_equal(alignment.second, "ABXXCD", "alignment mutated");

    std::cout << "mutator tests passed\n";
    return 0;
}
