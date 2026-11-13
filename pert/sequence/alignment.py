"""Global sequence alignment (Needleman-Wunsch with affine gaps).

Used to (a) show the user what a mutation did to the sequence and (b) pair WT
and mutant residues when their lengths differ, which is the basis of every
"topology-flexible" comparison in the package.
"""

from __future__ import annotations

from typing import List, Tuple


def needleman_wunsch_affine(seq1, seq2, match_score=1, mismatch_score=-1, gap_open=-10, gap_extend=-1):
    """
    Performs global alignment using Needleman-Wunsch with an affine gap penalty.
    This is more biologically realistic than a simple linear gap penalty because it
    penalizes opening a new gap (e.g., an insertion/deletion event) more heavily
    than extending an existing gap.
    """
    n, m = len(seq1), len(seq2)
    
    # --- DYNAMIC PROGRAMMING (DP) TABLES ---
    # M:  Score ending with a match/mismatch between seq1[i] and seq2[j].
    # Ix: Score ending with a gap in seq2 (deletion from seq1).
    # Iy: Score ending with a gap in seq1 (insertion into seq1).
    M = [[0] * (m + 1) for _ in range(n + 1)]
    Ix = [[0] * (m + 1) for _ in range(n + 1)]
    Iy = [[0] * (m + 1) for _ in range(n + 1)]
    
    # --- POINTER TABLES (for backtracking) ---
    # These tables store the path taken to achieve the optimal score at each cell.
    Ptr_M = [[None] * (m + 1) for _ in range(n + 1)]
    Ptr_Ix = [[None] * (m + 1) for _ in range(n + 1)]
    Ptr_Iy = [[None] * (m + 1) for _ in range(n + 1)]

    # --- INITIALIZATION ---
    # Initialize the first row and column. A match (M) is impossible if one sequence is empty.
    for i in range(1, n + 1):
        M[i][0], Iy[i][0] = -float('inf'), -float('inf')
        Ix[i][0] = gap_open + (i - 1) * gap_extend # Cost of a gap of length i in seq2.
    for j in range(1, m + 1):
        M[0][j], Ix[0][j] = -float('inf'), -float('inf')
        Iy[0][j] = gap_open + (j - 1) * gap_extend # Cost of a gap of length j in seq1.

    # --- FILL DP TABLES ---
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            score = match_score if seq1[i-1] == seq2[j-1] else mismatch_score
            
            # Calculate score for M(i,j): coming from a previous match, insertion, or deletion.
            m_scores = { 'M': M[i-1][j-1], 'Ix': Ix[i-1][j-1], 'Iy': Iy[i-1][j-1] }
            best_prev_matrix = max(m_scores, key=m_scores.get)
            M[i][j] = score + m_scores[best_prev_matrix]
            Ptr_M[i][j] = best_prev_matrix

            # Calculate score for Ix(i,j): either opening a new gap or extending an old one.
            score_open_x = M[i-1][j] + gap_open
            score_extend_x = Ix[i-1][j] + gap_extend
            if score_open_x > score_extend_x:
                Ix[i][j] = score_open_x
                Ptr_Ix[i][j] = 'M' # Came from the Match matrix.
            else:
                Ix[i][j] = score_extend_x
                Ptr_Ix[i][j] = 'Ix' # Came from the Ix matrix (extending gap).

            # Calculate score for Iy(i,j): either opening a new gap or extending an old one.
            score_open_y = M[i][j-1] + gap_open
            score_extend_y = Iy[i][j-1] + gap_extend
            if score_open_y > score_extend_y:
                Iy[i][j] = score_open_y
                Ptr_Iy[i][j] = 'M' # Came from the Match matrix.
            else:
                Iy[i][j] = score_extend_y
                Ptr_Iy[i][j] = 'Iy' # Came from the Iy matrix (extending gap).

    # --- BACKTRACKING ---
    # Start from the bottom-right corner and trace back to the top-left using the pointers.
    align1, align2 = "", ""
    i, j = n, m
    
    # Determine which matrix (M, Ix, or Iy) had the highest score in the last cell.
    last_scores = {'M': M[i][j], 'Ix': Ix[i][j], 'Iy': Iy[i][j]}
    matrix = max(last_scores, key=last_scores.get)

    while i > 0 or j > 0:
        if matrix == 'M': # A match or mismatch step.
            align1 = seq1[i-1] + align1
            align2 = seq2[j-1] + align2
            matrix = Ptr_M[i][j] # Get the previous matrix from the pointer.
            i -= 1
            j -= 1
        elif matrix == 'Ix': # A deletion step (gap in seq2).
            align1 = seq1[i-1] + align1
            align2 = "-" + align2
            matrix = Ptr_Ix[i][j]
            i -= 1
        elif matrix == 'Iy': # An insertion step (gap in seq1).
            align1 = "-" + align1
            align2 = seq2[j-1] + align2
            matrix = Ptr_Iy[i][j]
            j -= 1
            
    return align1, align2


def verify_and_print_alignment(original_seq, mutated_seq, context_chars=20):
    """
    Aligns two sequences and prints a formatted, easy-to-read view of the differences,
    providing context around the mutated region.
    """
    print("\nAlignment Verification (Original vs. Mutated):")
    
    # Align the sequences to find the optimal alignment including gaps.
    aligned_orig, aligned_mut = needleman_wunsch_affine(original_seq, mutated_seq)
    
    # Find all indices where the aligned sequences differ.
    diff_indices = [i for i, (c1, c2) in enumerate(zip(aligned_orig, aligned_mut)) if c1 != c2]
            
    if not diff_indices:
        print("Sequences are identical. No mutations were effectively applied.")
        return

    # Determine a window around the differences to display for context.
    start_index = max(0, diff_indices[0] - context_chars)
    end_index = min(len(aligned_orig), diff_indices[-1] + context_chars + 1)

    # Extract the snippets of the alignment to be printed.
    orig_snippet = aligned_orig[start_index:end_index]
    mut_snippet = aligned_mut[start_index:end_index]
    
    # Create a "match bar" to visually highlight matches (|) and mismatches ( ).
    match_bar = "".join("|" if c1 == c2 else " " for c1, c2 in zip(orig_snippet, mut_snippet))

    # Print the final, formatted output.
    print(f"Displaying alignment from index {start_index} to {end_index}:")
    print(f"{'Original:':<10} {orig_snippet}")
    print(f"{'':<10} {match_bar}")
    print(f"{'Mutated:':<10} {mut_snippet}")


def align_index_pairs(seq_ref: str, seq_mut: str) -> List[Tuple[int, int]]:
    """Indices ``(i_ref, i_mut)`` of residues aligned to each other (gaps are skipped)."""
    aligned_ref, aligned_mut = needleman_wunsch_affine(seq_ref, seq_mut)
    pairs: List[Tuple[int, int]] = []
    idx_ref = 0
    idx_mut = 0
    for aa_ref, aa_mut in zip(aligned_ref, aligned_mut):
        ref_here = None
        mut_here = None
        if aa_ref != "-":
            ref_here = idx_ref
            idx_ref += 1
        if aa_mut != "-":
            mut_here = idx_mut
            idx_mut += 1
        if ref_here is not None and mut_here is not None:
            pairs.append((ref_here, mut_here))
    return pairs
