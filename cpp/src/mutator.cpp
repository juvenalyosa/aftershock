#include "pert/mutator.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <regex>
#include <sstream>

namespace pert {
namespace {

bool contains(const std::string& haystack, const std::string& needle) {
    return haystack.find(needle) != std::string::npos;
}

int to_int(const std::string& text) {
    return std::stoi(text);
}

std::string found_expected_message(
    const std::string& prefix,
    const std::string& mutation,
    int pos,
    char found,
    char expected
) {
    std::ostringstream out;
    out << prefix << " " << mutation << " mismatch at position " << pos
        << ": found '" << found << "' but expected '" << expected << "'.";
    return out.str();
}

std::string found_optional(const std::map<int, char>& aa_dict, int pos) {
    auto it = aa_dict.find(pos);
    if (it == aa_dict.end()) {
        return "None";
    }
    return std::string(1, it->second);
}

}  // namespace

std::map<int, char> convert_to_dict(const std::string& sequence, int start_pos) {
    std::map<int, char> out;
    for (std::size_t i = 0; i < sequence.size(); ++i) {
        out[static_cast<int>(i) + start_pos] = sequence[i];
    }
    return out;
}

std::string convert_to_sequence(const std::map<int, char>& aa_dict) {
    if (aa_dict.empty()) {
        return "";
    }
    const int min_pos = aa_dict.begin()->first;
    const int max_pos = aa_dict.rbegin()->first;
    std::string out;
    out.reserve(static_cast<std::size_t>(max_pos - min_pos + 1));
    for (int pos = min_pos; pos <= max_pos; ++pos) {
        auto it = aa_dict.find(pos);
        if (it != aa_dict.end()) {
            out.push_back(it->second);
        }
    }
    return out;
}

std::string mutate_sequence(
    const std::string& sequence,
    const std::vector<std::string>& mutations,
    int start_pos
) {
    std::map<int, char> aa_dict = convert_to_dict(sequence, start_pos);

    const std::regex snp_re("^([A-Z])(\\d+)([A-Z])$");
    const std::regex range_del_re("([A-Z]?)(\\d+)_([A-Z]?)(\\d+)del");
    const std::regex single_del_re("([A-Z]?)(\\d+)del");
    const std::regex delins_re("([A-Z]?)(\\d+)(?:_([A-Z]?)(\\d+))?delins([A-Z]+)");
    const std::regex ins_re("([A-Z]?)(\\d+)_([A-Z]?)(\\d+)ins([A-Z]+)");
    const std::regex range_dup_re("([A-Z]?)(\\d+)_([A-Z]?)(\\d+)dup");
    const std::regex single_dup_re("([A-Z]?)(\\d+)dup");

    for (const std::string& mutation : mutations) {
        std::smatch match;

        if (std::regex_match(mutation, match, snp_re)) {
            const char old_aa = match[1].str()[0];
            const int pos = to_int(match[2].str());
            const char new_aa = match[3].str()[0];
            auto it = aa_dict.find(pos);
            if (it != aa_dict.end() && it->second == old_aa) {
                it->second = new_aa;
            } else {
                std::ostringstream out;
                out << "Substitution " << mutation << " could not be applied: "
                    << "mismatch or out-of-bounds at position " << pos
                    << " (found '" << found_optional(aa_dict, pos)
                    << "' but expected '" << old_aa << "').";
                throw MutationError(out.str());
            }
            continue;
        }

        if (contains(mutation, "del") && !contains(mutation, "delins")) {
            int start_del = 0;
            int end_del = 0;
            if (std::regex_match(mutation, match, range_del_re)) {
                start_del = to_int(match[2].str());
                end_del = to_int(match[4].str());
            } else if (std::regex_match(mutation, match, single_del_re)) {
                start_del = to_int(match[2].str());
                end_del = start_del;
            } else {
                throw MutationError("Could not parse deletion mutation: '" + mutation + "'.");
            }

            const int shift = (end_del - start_del) + 1;
            if (shift <= 0) {
                throw MutationError("Deletion " + mutation + " has non-positive length (start >= end).");
            }

            std::map<int, char> new_dict;
            for (const auto& [pos_key, aa] : aa_dict) {
                if (pos_key < start_del) {
                    new_dict[pos_key] = aa;
                } else if (pos_key > end_del) {
                    new_dict[pos_key - shift] = aa;
                }
            }
            aa_dict = std::move(new_dict);
            continue;
        }

        if (contains(mutation, "delins")) {
            if (!std::regex_match(mutation, match, delins_re)) {
                throw MutationError("Could not parse delins mutation: '" + mutation + "'.");
            }

            const std::string start_aa_text = match[1].str();
            const int start_pos_del = to_int(match[2].str());
            if (!start_aa_text.empty()) {
                const char expected = start_aa_text[0];
                auto it = aa_dict.find(start_pos_del);
                if (it == aa_dict.end() || it->second != expected) {
                    const char found = it == aa_dict.end() ? '\0' : it->second;
                    if (it == aa_dict.end()) {
                        std::ostringstream out;
                        out << "Delins " << mutation << " mismatch at start position "
                            << start_pos_del << ": found 'None' but expected '"
                            << expected << "'.";
                        throw MutationError(out.str());
                    }
                    throw MutationError(
                        found_expected_message("Delins", mutation, start_pos_del, found, expected)
                    );
                }
            }

            const int end_pos_del = match[4].matched ? to_int(match[4].str()) : start_pos_del;
            const std::string insertion = match[5].str();
            const int deleted_len = end_pos_del - start_pos_del + 1;
            const int net_shift = static_cast<int>(insertion.size()) - deleted_len;

            std::map<int, char> new_dict;
            for (const auto& [pos_key, aa] : aa_dict) {
                if (pos_key < start_pos_del) {
                    new_dict[pos_key] = aa;
                } else if (pos_key > end_pos_del) {
                    new_dict[pos_key + net_shift] = aa;
                }
            }
            for (std::size_t i = 0; i < insertion.size(); ++i) {
                new_dict[start_pos_del + static_cast<int>(i)] = insertion[i];
            }
            aa_dict = std::move(new_dict);
            continue;
        }

        if (contains(mutation, "ins")) {
            if (!std::regex_match(mutation, match, ins_re)) {
                throw MutationError("Could not parse insertion mutation: '" + mutation + "'.");
            }
            const int pos_after = to_int(match[2].str());
            const std::string insertion = match[5].str();
            const int insertion_len = static_cast<int>(insertion.size());

            std::map<int, char> new_dict;
            for (const auto& [pos_key, aa] : aa_dict) {
                if (pos_key <= pos_after) {
                    new_dict[pos_key] = aa;
                } else {
                    new_dict[pos_key + insertion_len] = aa;
                }
            }
            for (std::size_t i = 0; i < insertion.size(); ++i) {
                new_dict[pos_after + 1 + static_cast<int>(i)] = insertion[i];
            }
            aa_dict = std::move(new_dict);
            continue;
        }

        if (contains(mutation, "dup")) {
            int start_dup = 0;
            int end_dup = 0;
            if (std::regex_match(mutation, match, range_dup_re)) {
                start_dup = to_int(match[2].str());
                end_dup = to_int(match[4].str());
            } else if (std::regex_match(mutation, match, single_dup_re)) {
                start_dup = to_int(match[2].str());
                end_dup = start_dup;
            } else {
                throw MutationError("Could not parse duplication mutation: '" + mutation + "'.");
            }

            std::vector<char> duplicated_segment;
            for (int pos = start_dup; pos <= end_dup; ++pos) {
                auto it = aa_dict.find(pos);
                if (it != aa_dict.end()) {
                    duplicated_segment.push_back(it->second);
                }
            }
            if (duplicated_segment.empty()) {
                throw MutationError("Could not find sequence to duplicate for mutation '" + mutation + "'.");
            }

            const int dup_len = static_cast<int>(duplicated_segment.size());
            std::map<int, char> new_dict;
            for (const auto& [pos_key, aa] : aa_dict) {
                if (pos_key <= end_dup) {
                    new_dict[pos_key] = aa;
                } else {
                    new_dict[pos_key + dup_len] = aa;
                }
            }
            for (std::size_t i = 0; i < duplicated_segment.size(); ++i) {
                new_dict[end_dup + 1 + static_cast<int>(i)] = duplicated_segment[i];
            }
            aa_dict = std::move(new_dict);
            continue;
        }
    }

    return convert_to_sequence(aa_dict);
}

std::pair<std::string, std::string> needleman_wunsch_affine(
    const std::string& seq1,
    const std::string& seq2,
    int match_score,
    int mismatch_score,
    int gap_open,
    int gap_extend
) {
    const int n = static_cast<int>(seq1.size());
    const int m = static_cast<int>(seq2.size());
    const int neg_inf = std::numeric_limits<int>::min() / 4;

    std::vector<std::vector<int>> mat_m(n + 1, std::vector<int>(m + 1, 0));
    std::vector<std::vector<int>> mat_ix(n + 1, std::vector<int>(m + 1, 0));
    std::vector<std::vector<int>> mat_iy(n + 1, std::vector<int>(m + 1, 0));
    std::vector<std::vector<char>> ptr_m(n + 1, std::vector<char>(m + 1, '\0'));
    std::vector<std::vector<char>> ptr_ix(n + 1, std::vector<char>(m + 1, '\0'));
    std::vector<std::vector<char>> ptr_iy(n + 1, std::vector<char>(m + 1, '\0'));

    for (int i = 1; i <= n; ++i) {
        mat_m[i][0] = neg_inf;
        mat_iy[i][0] = neg_inf;
        mat_ix[i][0] = gap_open + (i - 1) * gap_extend;
    }
    for (int j = 1; j <= m; ++j) {
        mat_m[0][j] = neg_inf;
        mat_ix[0][j] = neg_inf;
        mat_iy[0][j] = gap_open + (j - 1) * gap_extend;
    }

    for (int i = 1; i <= n; ++i) {
        for (int j = 1; j <= m; ++j) {
            const int score = seq1[i - 1] == seq2[j - 1] ? match_score : mismatch_score;

            char best_prev = 'M';
            int best_score = mat_m[i - 1][j - 1];
            if (mat_ix[i - 1][j - 1] > best_score) {
                best_score = mat_ix[i - 1][j - 1];
                best_prev = 'X';
            }
            if (mat_iy[i - 1][j - 1] > best_score) {
                best_score = mat_iy[i - 1][j - 1];
                best_prev = 'Y';
            }
            mat_m[i][j] = score + best_score;
            ptr_m[i][j] = best_prev;

            const int score_open_x = mat_m[i - 1][j] + gap_open;
            const int score_extend_x = mat_ix[i - 1][j] + gap_extend;
            if (score_open_x > score_extend_x) {
                mat_ix[i][j] = score_open_x;
                ptr_ix[i][j] = 'M';
            } else {
                mat_ix[i][j] = score_extend_x;
                ptr_ix[i][j] = 'X';
            }

            const int score_open_y = mat_m[i][j - 1] + gap_open;
            const int score_extend_y = mat_iy[i][j - 1] + gap_extend;
            if (score_open_y > score_extend_y) {
                mat_iy[i][j] = score_open_y;
                ptr_iy[i][j] = 'M';
            } else {
                mat_iy[i][j] = score_extend_y;
                ptr_iy[i][j] = 'Y';
            }
        }
    }

    int i = n;
    int j = m;
    char matrix = 'M';
    int last_score = mat_m[i][j];
    if (mat_ix[i][j] > last_score) {
        last_score = mat_ix[i][j];
        matrix = 'X';
    }
    if (mat_iy[i][j] > last_score) {
        matrix = 'Y';
    }

    std::string align1;
    std::string align2;
    align1.reserve(seq1.size() + seq2.size());
    align2.reserve(seq1.size() + seq2.size());

    while (i > 0 || j > 0) {
        if (matrix == 'M') {
            align1.push_back(seq1[static_cast<std::size_t>(i - 1)]);
            align2.push_back(seq2[static_cast<std::size_t>(j - 1)]);
            matrix = ptr_m[i][j];
            --i;
            --j;
        } else if (matrix == 'X') {
            align1.push_back(seq1[static_cast<std::size_t>(i - 1)]);
            align2.push_back('-');
            matrix = ptr_ix[i][j];
            --i;
        } else {
            align1.push_back('-');
            align2.push_back(seq2[static_cast<std::size_t>(j - 1)]);
            matrix = ptr_iy[i][j];
            --j;
        }
    }

    std::reverse(align1.begin(), align1.end());
    std::reverse(align2.begin(), align2.end());
    return {align1, align2};
}

std::string format_alignment_report(
    const std::string& original_seq,
    const std::string& mutated_seq,
    int context_chars
) {
    auto [aligned_orig, aligned_mut] = needleman_wunsch_affine(original_seq, mutated_seq);
    std::vector<int> diff_indices;
    const int n = static_cast<int>(std::min(aligned_orig.size(), aligned_mut.size()));
    for (int i = 0; i < n; ++i) {
        if (aligned_orig[static_cast<std::size_t>(i)] != aligned_mut[static_cast<std::size_t>(i)]) {
            diff_indices.push_back(i);
        }
    }

    std::ostringstream out;
    out << "Alignment Verification (Original vs. Mutated):\n";
    if (diff_indices.empty()) {
        out << "Sequences are identical. No mutations were effectively applied.\n";
        return out.str();
    }

    const int start_index = std::max(0, diff_indices.front() - context_chars);
    const int end_index = std::min(n, diff_indices.back() + context_chars + 1);
    const std::string orig_snippet = aligned_orig.substr(
        static_cast<std::size_t>(start_index),
        static_cast<std::size_t>(end_index - start_index)
    );
    const std::string mut_snippet = aligned_mut.substr(
        static_cast<std::size_t>(start_index),
        static_cast<std::size_t>(end_index - start_index)
    );

    std::string match_bar;
    match_bar.reserve(orig_snippet.size());
    for (std::size_t k = 0; k < orig_snippet.size(); ++k) {
        match_bar.push_back(orig_snippet[k] == mut_snippet[k] ? '|' : ' ');
    }

    out << "Displaying alignment from index " << start_index << " to " << end_index << ":\n";
    out << "Original:  " << orig_snippet << "\n";
    out << "           " << match_bar << "\n";
    out << "Mutated:   " << mut_snippet << "\n";
    return out.str();
}

}  // namespace pert
