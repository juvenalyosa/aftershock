#include "pert/mutator.hpp"

#include <iostream>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " SEQUENCE START_POS MUTATION [MUTATION ...]\n";
        return 1;
    }

    const std::string sequence = argv[1];
    int start_pos = 1;
    try {
        start_pos = std::stoi(argv[2]);
    } catch (const std::exception&) {
        std::cerr << "Invalid START_POS: " << argv[2] << "\n";
        return 1;
    }

    std::vector<std::string> mutations;
    for (int i = 3; i < argc; ++i) {
        mutations.emplace_back(argv[i]);
    }

    try {
        const std::string mutated = pert::mutate_sequence(sequence, mutations, start_pos);
        std::cout << mutated << "\n\n";
        std::cout << pert::format_alignment_report(sequence, mutated);
    } catch (const pert::MutationError& exc) {
        std::cerr << "Mutation error: " << exc.what() << "\n";
        return 2;
    } catch (const std::exception& exc) {
        std::cerr << "Error: " << exc.what() << "\n";
        return 3;
    }

    return 0;
}
