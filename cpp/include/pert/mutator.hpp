#pragma once

#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace pert {

class MutationError : public std::runtime_error {
public:
    explicit MutationError(const std::string& message) : std::runtime_error(message) {}
};

std::map<int, char> convert_to_dict(const std::string& sequence, int start_pos);

std::string convert_to_sequence(const std::map<int, char>& aa_dict);

std::string mutate_sequence(
    const std::string& sequence,
    const std::vector<std::string>& mutations,
    int start_pos = 1
);

std::pair<std::string, std::string> needleman_wunsch_affine(
    const std::string& seq1,
    const std::string& seq2,
    int match_score = 1,
    int mismatch_score = -1,
    int gap_open = -10,
    int gap_extend = -1
);

std::string format_alignment_report(
    const std::string& original_seq,
    const std::string& mutated_seq,
    int context_chars = 20
);

}  // namespace pert
