#!/usr/bin/env python3
# encoding: utf-8

import sys
import copy
import random
import re
import time
import bisect
import pickle
import numpy as np
import argparse
import pathlib
from collections import namedtuple, defaultdict
from source.SequenceContainer import SequenceContainer, ReadContainer, parse_input_mutation_model
import array
# Define ALLOWED_NUCL and NUCL (often the same)
ALLOWED_NUCL = ['A', 'C', 'G', 'T']
NUCL = ['A', 'C', 'G', 'T'] # Used in DiscreteDistribution for nucleotides
NUC_IND = {'A': 0, 'C': 1, 'G': 2, 'T': 3} # For indexing into error matrices
UNMODIFIABLE_BASES = ['N'] # Bases that we won't try to introduce errors on.


# Define a named tuple for BED records for better readability
BedRecord = namedtuple('BedRecord', ['chrom', 'start', 'end', 'name', 'strand'])

# --- DiscreteDistribution class ---
class DiscreteDistribution:
    def __init__(self, probabilities, values, degenerate_val=None):
        self.probabilities = np.array(probabilities, dtype=float)
        self.values = values
        self.degenerate_val = degenerate_val

        if np.sum(self.probabilities) > 0:
            self.probabilities = self.probabilities / np.sum(self.probabilities)
        else:
            # If probabilities are all zero, make it a uniform distribution over values
            # or a single value if degenerate_val is provided.
            if self.degenerate_val is not None:
                # If there's a degenerate value, force it
                if self.values and self.degenerate_val in self.values:
                    idx = self.values.index(self.degenerate_val)
                    self.probabilities = np.zeros_like(probabilities)
                    self.probabilities[idx] = 1.0
                elif self.values: # Degenerate not in values, pick first one
                     self.probabilities = np.zeros_like(probabilities)
                     self.probabilities[0] = 1.0
                else: # No values, no degenerate - this case shouldn't happen with how DD is used
                    self.probabilities = np.array([1.0]) # Fallback
            elif self.values: # This is the uniform fallback if no degenerate value and zero sum
                self.probabilities = np.ones_like(probabilities) / len(probabilities)
            else:
                self.probabilities = np.array([1.0]) # Empty values list fallback

    def sample(self):
        if self.degenerate_val is not None:
            return self.degenerate_val
        if not self.values:
            return None
        return np.random.choice(self.values, p=self.probabilities)


class ReadContainer:
    """
    Container for read data: computes quality scores and positions to insert errors
    """

    def __init__(self, read_len, error_model, rescaled_error, rescale_qual=False):
        
        self.read_len = read_len
        self.rescale_qual = rescale_qual
        

        model_path = pathlib.Path(error_model)
        try:
            # Use 'latin1' or 'iso-8859-1' for older Python 2 pickles if 'bytes' fails
            error_dat = pickle.load(open(model_path, 'rb'), encoding="bytes")
        except IOError:
            print("\nProblem opening the sequencing error model.\n")
            sys.exit(1)

        print(f"\nDEBUG: Loaded error model from: {model_path}")

        self.uniform = False

        q_scores = [] # Placeholder, will be overwritten

        # uniform-error SE reads (e.g., PacBio)
        if len(error_dat) == 4:
            self.uniform = True
            [q_scores, off_q, avg_error, error_params] = error_dat
            self.uniform_q_score = min([max(q_scores), int(-10. * np.log10(avg_error) + 0.5)])
            print('Reading in uniform sequencing error model... (q=' + str(self.uniform_q_score) + '+' + str(
                off_q) + ', p(err)={0:0.2f}%)'.format(100. * avg_error))

        # only 1 q-score model present, use same model for both strands
        elif len(error_dat) == 6:
            [init_q1, prob_q1, q_scores, off_q, avg_error, error_params] = error_dat
            self.pe_models = False

        # found a q-score model for both forward and reverse strands
        elif len(error_dat) == 8:
            [init_q1, prob_q1, init_q2, prob_q2, q_scores, off_q, avg_error, error_params] = error_dat
            self.pe_models = True
            if len(init_q1) != len(init_q2) or len(prob_q1) != len(prob_q2):
                print('\nError: R1 and R2 quality score models are of different length.\n')
                sys.exit(1)

        # This serves as a sanity check for the input model
        else:
            print('\nError: Something wrong with error model. Expected 4, 6, or 8 items. Got:', len(error_dat), '\n')
            sys.exit(1)

        self.q_scores = q_scores # Make q_scores an instance variable
        self.q_scores_map = {q: idx for idx, q in enumerate(self.q_scores)} # Map Q-score value to its index



        self.q_err_rate = [0.] * (max(self.q_scores) + 1)
        for q in self.q_scores:
            self.q_err_rate[q] = 10. ** (-q / 10.)
        self.off_q = off_q
        self.err_p = error_params
        self.err_sse = [DiscreteDistribution(n, NUCL) for n in self.err_p[0]]
        self.err_sie = DiscreteDistribution(self.err_p[2], self.err_p[3])
        self.err_sin = DiscreteDistribution(self.err_p[5], NUCL)


        if isinstance(rescaled_error, (int, float)): # Add this check
                    print(f"DEBUG: Desired rescaled_error (from argument): {rescaled_error:.6f} ({100.*rescaled_error:.3f}%)")
        if rescaled_error is None:
            self.error_scale = 1.0
        else:
            self.error_scale = rescaled_error / avg_error
            if not self.rescale_qual:
                print('Warning: Quality scores no longer exactly representative of error probability. '
                      'Error model scaled by {0:.3f} to match desired rate...'.format(self.error_scale))
            if self.uniform:
                if rescaled_error <= 0.:
                    self.uniform_q_score = max(self.q_scores)
                else:
                    self.uniform_q_score = min([max(self.q_scores), int(-10. * np.log10(rescaled_error) + 0.5)])
                print(' - Uniform quality score scaled to match specified error rate (q=' + str(
                    self.uniform_q_score) + '+' + str(self.off_q) + ', p(err)={0:0.2f}%)'.format(100. * rescaled_error))
        print(f"DEBUG: Calculated error_scale: {self.error_scale:.6f}")


        if not self.uniform:
            # adjust length to match desired read length
            if self.read_len == len(init_q1):
                self.q_ind_remap = range(self.read_len)
            else:
                print('Warning: Read length of error model (' + str(len(init_q1)) + ') does not match -R value (' + str(
                    self.read_len) + '), rescaling model...')
                self.q_ind_remap = [min(len(init_q1) - 1, max(0, int(len(init_q1) * n / read_len))) for n in range(read_len)]

            # --- DEBUGGING PRINTS ---
            print(f"DEBUG: init_q1 length (num positions in model): {len(init_q1)}")
            if len(init_q1) > 1: # Only if model has more than one position
                zero_sum_count = 0
                total_transitions = 0
                for i_pos in range(1, len(prob_q1)):
                    for j_prev_q_idx in range(len(prob_q1[i_pos])):
                        total_transitions += 1
                        if np.sum(prob_q1[i_pos][j_prev_q_idx]) <= 0.:
                            zero_sum_count += 1
                print(f"DEBUG: Number of zero-sum quality transition distributions: {zero_sum_count} out of {total_transitions}")
            # --- END DEBUGGING PRINTS ---



            # initialize probability distributions
            self.init_dist_by_pos_1 = []
            for i in range(len(init_q1)):
                # You could add similar modification logic for initial Q-scores here if desired
                self.init_dist_by_pos_1.append(DiscreteDistribution(init_q1[i], self.q_scores))
                # --- DEBUGGING: Print initial distribution probabilities ---
                if i == 0:
                    probs_str = [f"{p:.2e}" for p in self.init_dist_by_pos_1[-1].probabilities[:10]] + ["..."]
                    print(f"DEBUG_DD_INIT: init_dist_by_pos_1[{i}].probabilities (first 10): [{', '.join(probs_str)}]")
                    if len(self.init_dist_by_pos_1[-1].probabilities) > 10:
                        probs_str_last = [f"{p:.2e}" for p in self.init_dist_by_pos_1[-1].probabilities[-10:]]
                        print(f"DEBUG_DD_INIT: init_dist_by_pos_1[{i}].probabilities (last 10): [{', '.join(probs_str_last)}]")


            self.prob_dist_by_pos_by_prev_q1 = [None]
            for i in range(1, len(init_q1)):
                self.prob_dist_by_pos_by_prev_q1.append([])
                for j in range(len(init_q1[0])): # j is index for previous q-score

                    # IMPORTANT: Create a copy of the probabilities to modify them without affecting the original loaded data
                    current_probs = prob_q1[i][j].copy() 
                    sum_current_probs = np.sum(current_probs)

                    if sum_current_probs <= 0. or sum_current_probs < 1e-9: # Check for zero-sum or very small sum
                        # Fallback to a uniform distribution
                        fallback_probs = np.ones(len(self.q_scores)) / len(self.q_scores)
                        self.prob_dist_by_pos_by_prev_q1[-1].append(
                            DiscreteDistribution(fallback_probs, self.q_scores))
                        # --- DEBUGGING: Print uniform fallback distribution probabilities ---
                        if (i == 1 and j == 0) or (i == 5 and j == self.q_scores_map.get(34, 0)): # Example specific positions
                            probs_str = [f"{p:.2e}" for p in self.prob_dist_by_pos_by_prev_q1[-1][-1].probabilities[:5]] + ["..."]
                            print(f"DEBUG_DD_TRANS: pos={i}, prev_q_idx={j} (Q={self.q_scores[j]}), Type=Uniform Fallback. Probabilities (first 5): [{', '.join(probs_str)}]")
                    else:
                        # ### QUALITY MODIFICATION LOGIC START (Q34 delta) ###
                        # Only modify if:
                        # 1. Modification is enabled (--modify_q_dist)
                        # 2. The previous quality score (j) was Q34 (Q34_IDX)
                        # 3. Q34_IDX is valid (i.e., Q34 exists in the model)
                        # 4. The probability of transitioning to Q34 is sufficiently high initially (min_q34_prob_to_modify)
                        
                        self.prob_dist_by_pos_by_prev_q1[-1].append(
                            DiscreteDistribution(current_probs, self.q_scores))
                        # --- DEBUGGING: Print loaded distribution probabilities ---
                        if (i == 1 and j == self.q_scores_map.get(34, 0)) or (i == 5 and j == self.q_scores_map.get(34, 0)): # Example specific positions
                            probs_str = [f"{p:.2e}" for p in self.prob_dist_by_pos_by_prev_q1[-1][-1].probabilities[:10]] + ["..."]
                            print(f"DEBUG_DD_TRANS: pos={i}, prev_q_idx={j} (Q={self.q_scores[j]}), Type=Loaded. Probabilities (first 10): [{', '.join(probs_str)}]")
                            if len(self.prob_dist_by_pos_by_prev_q1[-1][-1].probabilities) > 10:
                                 probs_str_last = [f"{p:.2e}" for p in self.prob_dist_by_pos_by_prev_q1[-1][-1].probabilities[-10:]]
                                 print(f"DEBUG_DD_TRANS: pos={i}, prev_q_idx={j} (Q={self.q_scores[j]}), Type=Loaded. Probabilities (last 10): [{', '.join(probs_str_last)}]")


            # If paired-end, initialize probability distributions for the other strand
            if self.pe_models:
                # You'd apply the same modification logic here for R2 if needed
                self.init_dist_by_pos_2 = []
                for i in range(len(init_q2)):
                    self.init_dist_by_pos_2.append(DiscreteDistribution(init_q2[i], self.q_scores))
                self.prob_dist_by_pos_by_prev_q2 = [None]
                for i in range(1, len(init_q2)):
                    self.prob_dist_by_pos_by_prev_q2.append([])
                    for j in range(len(init_q2[0])):
                        current_probs_r2 = prob_q2[i][j].copy() # Copy for modification
                        sum_current_probs_r2 = np.sum(current_probs_r2)

                        if sum_current_probs_r2 <= 0. or sum_current_probs_r2 < 1e-9:
                            fallback_probs_r2 = np.ones(len(self.q_scores)) / len(self.q_scores)
                            self.prob_dist_by_pos_by_prev_q2[-1].append(
                                DiscreteDistribution(fallback_probs_r2, self.q_scores))
                        else:
                            # ### QUALITY MODIFICATION LOGIC FOR R2 (similar to R1) ###
                            if self.modify_q_dist and j == Q34_IDX and Q34_IDX != -1:
                                prob_to_q34_r2 = current_probs_r2[Q34_IDX]
                                if prob_to_q34_r2 >= self.min_q34_prob_to_modify:
                                    reduction_amount_r2 = prob_to_q34_r2 * self.q34_reduction_delta
                                    current_probs_r2[Q34_IDX] -= reduction_amount_r2
                                    other_q_indices_r2 = [k for k in range(len(self.q_scores)) if k != Q34_IDX]
                                    if other_q_indices_r2:
                                        if self.q_redistribution_method == "uniform":
                                            redistribution_per_other_r2 = reduction_amount_r2 / len(other_q_indices_r2)
                                            for k in other_q_indices_r2:
                                                current_probs_r2[k] += redistribution_per_other_r2
                                        elif self.q_redistribution_method == "proportional":
                                            sum_other_probs_r2 = np.sum([current_probs_r2[k] for k in other_q_indices_r2])
                                            if sum_other_probs_r2 > 0:
                                                for k in other_q_indices_r2:
                                                    current_probs_r2[k] += reduction_amount_r2 * (current_probs_r2[k] / sum_other_probs_r2)
                                            else:
                                                redistribution_per_other_r2 = reduction_amount_r2 / len(other_q_indices_r2)
                                                for k in other_q_indices_r2:
                                                    current_probs_r2[k] += redistribution_per_other_r2
                                    current_probs_r2 = current_probs_r2 / np.sum(current_probs_r2)
                                    # print(f"DEBUG_Q_MODIFY_R2: pos={i}, prev_q_idx={j} (Q=34). Modified R2 probs.") # Optional debug
                            # ### END QUALITY MODIFICATION LOGIC FOR R2 ###
                            self.prob_dist_by_pos_by_prev_q2[-1].append(DiscreteDistribution(current_probs_r2, self.q_scores))


    def get_sequencing_errors(self, read_data, is_reverse_strand=False):
        """
        Inserts errors of type substitution, insertion, or deletion into read_data, and assigns a quality score
        based on the container model.

        :param read_data: sequence to insert errors into
        :param is_reverse_strand: whether to treat this as the reverse strand or not
        :return: modified sequence and associate quality scores
        """

        q_out = [0] * self.read_len
        s_err = [] # list of error positions

        if self.uniform:
            my_q = [self.uniform_q_score + self.off_q] * self.read_len
            q_out = ''.join([chr(n) for n in my_q])
            for i in range(self.read_len):
                if random.random() < self.error_scale * self.q_err_rate[self.uniform_q_score]:
                    s_err.append(i)
        else:
            # Sample initial quality score for position 0
            if self.pe_models and is_reverse_strand:
                my_q = self.init_dist_by_pos_2[0].sample()
            else:
                my_q = self.init_dist_by_pos_1[0].sample()
            q_out[0] = my_q

            # Sample quality scores for subsequent positions based on previous quality score
            for i in range(1, self.read_len):
                prev_q_idx = self.q_scores_map[my_q] # Get index of previously sampled Q-score

                if self.pe_models and is_reverse_strand:
                    # Ensure index is within bounds of prob_dist_by_pos_by_prev_q2
                    if self.q_ind_remap[i] < len(self.prob_dist_by_pos_by_prev_q2) and \
                       prev_q_idx < len(self.prob_dist_by_pos_by_prev_q2[self.q_ind_remap[i]]):
                        my_q = self.prob_dist_by_pos_by_prev_q2[self.q_ind_remap[i]][prev_q_idx].sample()
                    else:
                        # Fallback if indices are out of bounds (should ideally not happen with correct models)
                        print(f"DEBUG_GET_ERR_WARN: R2 Q-score transition index out of bounds (pos={i}, prev_q_idx={prev_q_idx}). Resampling from init_dist_by_pos_2[0].")
                        my_q = self.init_dist_by_pos_2[0].sample() # Fallback to initial distribution
                else:
                    # Ensure index is within bounds of prob_dist_by_pos_by_prev_q1
                    if self.q_ind_remap[i] < len(self.prob_dist_by_pos_by_prev_q1) and \
                       prev_q_idx < len(self.prob_dist_by_pos_by_prev_q1[self.q_ind_remap[i]]):
                        my_q = self.prob_dist_by_pos_by_prev_q1[self.q_ind_remap[i]][prev_q_idx].sample()
                    else:
                        print(f"DEBUG_GET_ERR_WARN: R1 Q-score transition index out of bounds (pos={i}, prev_q_idx={prev_q_idx}). Resampling from init_dist_by_pos_1[0].")
                        my_q = self.init_dist_by_pos_1[0].sample() # Fallback to initial distribution
                
                q_out[i] = my_q

            # Quality scores are sampled based on the forward orientation of the model
            # If the read is reverse strand, the quality scores should also be reversed for output
            if is_reverse_strand:
                q_out = q_out[::-1]

            # --- DEBUGGING PRINTS FOR Q_OUT ---
            print(f"DEBUG_GET_ERR: Raw sampled Q-scores (int): {q_out[:min(10, len(q_out))]}...{q_out[max(0, len(q_out)-10):]} (len: {len(q_out)})")
            print(f"DEBUG_GET_ERR: self.rescale_qual: {self.rescale_qual}")
            print(f"DEBUG_GET_ERR: self.error_scale: {self.error_scale:.6f}")


            # Determine positions for errors based on sampled quality scores
            for i in range(self.read_len):
                # Ensure index `q_out[i]` is within bounds for `self.q_err_rate`
                if 0 <= q_out[i] < len(self.q_err_rate):
                    if random.random() < self.error_scale * self.q_err_rate[q_out[i]]:
                        s_err.append(i)
                else:
                    # Handle out-of-bounds Q-score, assign a default error rate
                    print(f"DEBUG_GET_ERR_WARN: Sampled Q-score {q_out[i]} is out of expected range for q_err_rate. Clamping to max Q for error rate calculation.")
                    valid_q_score = max(self.q_scores) # Use highest Q from model for error rate lookup
                    if random.random() < self.error_scale * self.q_err_rate[valid_q_score]:
                        s_err.append(i)


            # Convert quality scores from integers to Phred+33 ASCII characters
            if self.rescale_qual:
                rescaled_q_values_after_log = []
                for n_raw in q_out: # n_raw is the sampled Q-score from DiscreteDistribution
                    try:
                        error_rate_n = self.q_err_rate[n_raw]
                    except IndexError:
                        print(f"DEBUG_GET_ERR_ERROR: q_err_rate index out of bounds (n_raw={n_raw}). Max q_err_rate index: {len(self.q_err_rate) - 1}. Using max Q for error rate.")
                        error_rate_n = 10.**(-max(self.q_scores)/10.) # Effectively highest quality
                    
                    scaled_error = self.error_scale * error_rate_n
                    
                    # Prevent log(0) error for very high quality / zero scaled_error
                    if scaled_error <= 0:
                        new_q = self.max_output_q # Assign highest possible Q based on specified output range
                    else:
                        new_q = int(-10. * np.log10(scaled_error) + 0.5)


                    rescaled_q_values_after_log.append(new_q)
                
                print(f"DEBUG_GET_ERR: Q-scores after rescaling and clamping (int): {rescaled_q_values_after_log[:min(10, len(rescaled_q_values_after_log))]}...{rescaled_q_values_after_log[max(0, len(rescaled_q_values_after_log)-10):]}")
                q_out_final_int = rescaled_q_values_after_log # This will be the list of integers
                q_out = ''.join([chr(n + self.off_q) for n in q_out_final_int]) # Convert to ASCII
            else:
                q_out = ''.join([chr(n + self.off_q) for n in q_out])

        if self.error_scale == 0.0:
            return q_out, []

        s_out = []
        n_del_so_far = 0
        prev_indel = -2
        del_blacklist = []

        for ind in s_err[::-1]:

            if ind in del_blacklist:
                continue

            is_sub = True
            # Check conditions for indel: not at start, not too close to end (to allow for max indel len), not adjacent to previous indel
            if ind != 0 and ind < (self.read_len - 1 - max(self.err_p[3])) and abs(ind - prev_indel) > 1:
                if random.random() < self.err_p[1]: # err_p[1] is indel probability
                    is_sub = False

            if is_sub:
                my_nucl = str(read_data[ind])
                if my_nucl in NUC_IND: # Ensure base is modifiable
                    new_nucl = self.err_sse[NUC_IND[my_nucl]].sample()
                    s_out.append(('S', 1, ind, my_nucl, new_nucl))
            else:
                indel_len = int(self.err_sie.sample()) # Ensure indel_len is an integer

                if random.random() < self.err_p[4]: # err_p[4] is insertion probability within indel group
                    my_nucl = str(read_data[ind])
                    # If indel_len is 0, no insertion occurs, alt_base is just my_nucl
                    if indel_len > 0:
                        new_nucl = my_nucl + ''.join([self.err_sin.sample() for _ in range(indel_len)])
                        s_out.append(('I', len(new_nucl) - 1, ind, my_nucl, new_nucl))
                    else:
                        # If indel_len is 0, it's effectively no indel at all,
                        # but we still count it as an 'I' with 0 length change.
                        s_out.append(('I', 0, ind, my_nucl, my_nucl))


                elif ind < self.read_len - 1 - n_del_so_far - indel_len: # Deletion case
                    # Ensure the deletion doesn't go past the end of the original read or overlap with other deletions
                    # `ind + 1` is the start of the deletion, `indel_len` is how many to delete after that
                    del_start_idx_in_read = ind + 1
                    del_end_idx_in_read = ind + 1 + indel_len
                    
                    # Ensure we don't try to delete beyond the current length of read_data
                    if del_end_idx_in_read > len(read_data):
                        del_end_idx_in_read = len(read_data)
                        indel_len = del_end_idx_in_read - (ind + 1) # Adjust indel_len if clipped

                    if indel_len > 0: # Only if there's an actual deletion
                        my_nucl = str(read_data[ind : ind + indel_len + 1]) # Original sequence including pivot base + deleted bases
                        new_nucl = str(read_data[ind]) # Sequence after deletion: only the pivot base
                        n_del_so_far += len(my_nucl) - 1 # Accumulate length change for deletions
                        s_out.append(('D', len(my_nucl) - 1, ind, my_nucl, new_nucl))
                        # Blacklist the indices of the deleted bases so no new errors are put there
                        for i_del_pos in range(ind + 1, ind + indel_len + 1):
                            del_blacklist.append(i_del_pos)
                    else: # If indel_len is 0, it's a 'D' with 0 length change (no deletion)
                        my_nucl = str(read_data[ind])
                        s_out.append(('D', 0, ind, my_nucl, my_nucl))

                prev_indel = ind

        final_s_out = []
        del_blacklist_set = set(del_blacklist)

        for error_tuple in s_out:
            # Only add to final list if the error isn't on a base that was deleted as part of another indel
            if error_tuple[2] not in del_blacklist_set:
                final_s_out.append(error_tuple)

        return q_out, final_s_out

# --- FASTA Helper (Simplified) ---
def load_fasta_as_dict(fasta_path):
    sequences = {}
    current_chrom = None
    current_seq_lines = []
    with open(fasta_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_chrom:
                    sequences[current_chrom] = ''.join(current_seq_lines)
                current_chrom = line[1:].split()[0] # Get chrom name, often before first space
                current_seq_lines = []
            else:
                current_seq_lines.append(line.upper()) # Convert to uppercase
    if current_chrom: # Add the last sequence
        sequences[current_chrom] = ''.join(current_seq_lines)
    return sequences

# --- MODIFIED: BED File Loader for individual reads by name (now including strand) ---
def load_bed_file_by_name(bed_path):
    """
    Loads a BED file and stores BedRecord objects, keyed by their 'name' field.
    Assumes BED format has at least 6 columns (chrom, start, end, name, strand).
    Returns a dictionary mapping read names to BedRecord tuples.
    """
    read_coords_by_name = {}
    with open(bed_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('track'):
                continue
            parts = line.split('\t')
            if len(parts) < 5: # Need at least chrom, start, end, name, score, strand
                print(f"Warning: Malformed BED line (fewer than 6 fields): {line}. Skipping.")
                continue

            try:
                chrom = parts[0]
                start = int(parts[1]) # BED is 0-based start
                end = int(parts[2])   # BED is 0-based end (exclusive)
                name = parts[3]
                strand = parts[4] # Now capturing strand
                
                if name in read_coords_by_name:
                    print(f"Warning: Duplicate read name '{name}' found in {bed_path}. Overwriting previous entry.")
                
                read_coords_by_name[name] = BedRecord(chrom, start, end, name, strand)
            except ValueError:
                print(f"Warning: Could not parse coordinates or other fields from BED line: {line}. Skipping.")
                continue
            except IndexError: # For cases where parts[3], parts[4], parts[5] don't exist
                print(f"Warning: Malformed BED line (missing name, score, or strand): {line}. Skipping.")
                continue
    return read_coords_by_name

# --- NEW/RETAINED: BED File Loader for general mutation sites ---
def load_general_mutation_bed(bed_path):
    """
    Loads a BED file defining general mutation regions.
    Returns a dictionary mapping chromosome names to a list of (start, end) tuples (0-based).
    """
    sites = {}
    with open(bed_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('track'):
                continue
            parts = line.split('\t')
            if len(parts) < 3:
                continue
            chrom = parts[0]
            try:
                start = int(parts[1])
                end = int(parts[2])
            except ValueError:
                print(f"Warning: Could not parse coordinates from general mutation BED line: {line}. Skipping.")
                continue
            if chrom not in sites:
                sites[chrom] = []
            sites[chrom].append((start, end))
    for chrom in sites:
        sites[chrom].sort() # Ensure sorted for potential later efficiency
    return sites

# --- Original Reverse Complement Function ---
def reverse_complement(seq):
    complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', 'N': 'N', 'a': 't', 't': 'a', 'c': 'g', 'g': 'c'}
    return ''.join([complement[base] for base in seq[::-1]])

# --- NEW: Function to apply forced mutations to the reference sequence segment ---
def apply_forced_mutations(ref_sequence_segment, chrom, segment_start_0based, mutation_sites, mutation_rate):
    """
    Applies forced mutations to a segment of the reference sequence based on
    predefined mutation sites and a given mutation rate.
    
    :param ref_sequence_segment: The genomic sequence for the read/segment (e.g., from FASTA).
    :param chrom: Chromosome name of the segment.
    :param segment_start_0based: 0-based genomic start coordinate of the segment.
    :param mutation_sites: Dictionary of {chrom: [(start, end), ...]} for mutation hotspots.
    :param mutation_rate: Probability (0-1) of a mutation occurring within a hotspot.
    :return: Mutated sequence string.
    """
    if not mutation_sites or chrom not in mutation_sites:
        return ref_sequence_segment # No mutation sites for this chromosome

    mutated_sequence_list = list(ref_sequence_segment)
    
    # Iterate through each base in the current segment
    for i, base in enumerate(mutated_sequence_list):
        current_genomic_pos = segment_start_0based + i
        
        # Check if the current genomic position is within any mutation hotspot
        # Using bisect_left for efficient searching if mutation_sites lists are sorted (they are)
        mutation_hotspots_on_chrom = mutation_sites[chrom]
        
        # Find potential intervals that could contain current_genomic_pos
        # This is a linear scan after finding potential start, can be optimized further with interval trees
        # but for simple BED it's usually fine.
        
        # Binary search for interval whose end is >= current_genomic_pos
        # and whose start is <= current_genomic_pos
        
        # Using a simple linear scan over sorted intervals for demonstration,
        # more performant would be an interval tree or bisect_left/right logic.
        is_in_mutation_hotspot = False
        for m_start, m_end in mutation_hotspots_on_chrom:
            if m_start <= current_genomic_pos < m_end:
                is_in_mutation_hotspot = True
                break
        
        if is_in_mutation_hotspot and base.upper() in ALLOWED_NUCL: # Only mutate ATGC bases
            if random.random() < mutation_rate:
                # Perform a substitution mutation
                original_base = base.upper()
                possible_alts = [n for n in ALLOWED_NUCL if n != original_base]
                if possible_alts:
                    mutated_sequence_list[i] = random.choice(possible_alts)
    
    return "".join(mutated_sequence_list)


# --- Renamed & Modified: Function to apply sequencing errors and format read ---
import array # If you decide to use array.array('B') for mutable sequences
import copy # For deepcopy if you need to copy mutable structures

# Assuming original_sequence is a string of characters (e.g., "ATGC")
# qual_scores_char_str is a string of quality characters (e.g., "ABCD")
# sequencing_errors_list is the list of (type, len, pos, ref, alt) tuples

def apply_sequencing_errors_and_format_read(
    original_sequence,
    qual_scores_char_str,
    sequencing_errors_list,
    read_length, # This should be the target length, not necessarily the actual length of original_sequence
    read_container_instance # This is your 'se_class' or 'ReadContainer' object
):
    # Convert original_sequence to a mutable format (e.g., list of characters or bytearray)
    # If it's a string, it's immutable, so we must convert.
    current_read_chars = list(original_sequence)
    current_qual_chars = list(qual_scores_char_str)

    # Initialize CIGAR generation variables
    # sse_adj tracks how much sequence has shifted due to indels, affecting error application positions
    # Initialize sse_adj to zeros for the length of the potential sequence + some buffer for insertions
    # A safer max buffer is needed here, let's use a conservative estimate or look up from error model.
    # The original gen_reads.py uses a complex way to manage sequence length, so for simplicity
    # and to avoid index errors, we'll try to just track adjustments.
    MAX_POSSIBLE_INDEL_BUFFER = 100 # Adjust based on your expected max indel length/frequency
    sse_adj = [0] * (read_length + MAX_POSSIBLE_INDEL_BUFFER) # Use read_length as base

    # Sort errors for proper application order: Deletions, then Insertions, then Substitutions
    # Within each type, sort by position
    # The error[2] is 'pos' (0-indexed position within the read)
    sorted_errors = sorted(sequencing_errors_list, key=lambda x: (
        0 if x[0] == 'D' else 1 if x[0] == 'I' else 2, x[2]
    ))

    # --- CIGAR Construction Variables ---
    # This will store (length, operation_type) tuples
    cigar_operations = []
    last_ref_consumed_pos = 0 # Tracks the last original reference position covered by M/D

    for error_type, error_len, error_pos_orig_read, ref_allele, alt_allele in sorted_errors:
        # Calculate length of match/mismatch segment *before* this error
        # 'error_pos_orig_read' is the 0-indexed position in the *original* sequence.
        # This is where the CIGAR construction needs to be based on original reference positions.
        match_len = error_pos_orig_read - last_ref_consumed_pos
        if match_len > 0:
            cigar_operations.append((match_len, 'M')) # M for match/mismatch

        # Apply the error to the read sequence and qualities, and update sse_adj
        actual_pos_in_read = error_pos_orig_read + sse_adj[error_pos_orig_read]

        if error_type == 'D': # Deletion
            # Remove characters from the read sequence and quality string
            del current_read_chars[actual_pos_in_read : actual_pos_in_read + error_len]
            del current_qual_chars[actual_pos_in_read : actual_pos_in_read + error_len]

            # Add deletion to CIGAR
            cigar_operations.append((error_len, 'D'))
            last_ref_consumed_pos = error_pos_orig_read + error_len # Reference advances

            # Update sse_adj for positions after the deletion
            for i in range(error_pos_orig_read, len(sse_adj)):
                sse_adj[i] -= error_len

        elif error_type == 'I': # Insertion
            # Insert characters into the read sequence and quality string
            # For qualities, you'll need to insert default quality characters (e.g., 'F' or 'B')
            # or infer from the model if it provides insertion qualities.
            # For simplicity, using a placeholder quality for inserted bases.
            inserted_qual_char = chr(read_container_instance.min_qual + 33) if hasattr(read_container_instance, 'min_qual') else 'F' # Common placeholder for qualities

            for i in range(error_len): # Assuming error_len is length of alt_allele for simplicity
                current_read_chars.insert(actual_pos_in_read + i, alt_allele[i])
                current_qual_chars.insert(actual_pos_in_read + i, inserted_qual_char)

            # Add insertion to CIGAR
            cigar_operations.append((error_len, 'I'))
            # last_ref_consumed_pos does *not* advance for insertions, as they don't consume reference bases

            # Update sse_adj for positions after the insertion
            for i in range(error_pos_orig_read, len(sse_adj)):
                sse_adj[i] += error_len

        elif error_type == 'S': # Substitution (Mismatch)
            # Replace character in read sequence and quality string
            current_read_chars[actual_pos_in_read] = alt_allele # Assuming single base substitution
            # No change to sse_adj for substitutions
            # last_ref_consumed_pos advances by 1 (length of substitution)
            last_ref_consumed_pos = error_pos_orig_read + 1 # Advance by 1 for substitution

        # Apply N-max-qual: If a quality score is below n_max_qual, replace base with 'N'
        # This part should be applied *after* all other errors, on the final read_chars
        # Or you can do it here if it makes sense. The original gen_reads does it late.
        # Let's keep it separate for now, or put it after the main error loop.

    # After processing all errors, add any remaining match/mismatch segment
    remaining_match_len = read_length - last_ref_consumed_pos
    if remaining_match_len > 0:
        cigar_operations.append((remaining_match_len, 'M'))

    # Condense consecutive CIGAR operations of the same type
    condensed_cigar_ops = []
    if cigar_operations:
        current_len, current_type = cigar_operations[0]
        for i in range(1, len(cigar_operations)):
            next_len, next_type = cigar_operations[i]
            if next_type == current_type:
                current_len += next_len
            else:
                condensed_cigar_ops.append((current_len, current_type))
                current_len, current_type = next_len, next_type
        condensed_cigar_ops.append((current_len, current_type))

    # Format the CIGAR string (e.g., "70M1D29M")
    # This cigar string describes the alignment of the *modified* sequence to the *reference*.
    cigar_string = "".join([f"{length}{op_type}" for length, op_type in condensed_cigar_ops])

    # Now, handle the final clipping to 'read_length' and N-max-qual
    final_read_sequence = "".join(current_read_chars[:read_length]) # Clip to desired read_length
    final_qual_sequence = "".join(current_qual_chars[:read_length]) # Clip qualities too

    # Apply N-max-qual logic (if read_container_instance has n_max_qual set)
    if hasattr(read_container_instance, 'n_max_qual') and read_container_instance.n_max_qual != -1:
        # Quality scores are ASCII characters, so subtract 33 to get integer Phred score
        n_max_qual_int = read_container_instance.n_max_qual
        temp_read_list = list(final_read_sequence) # Convert back to list for mutable modification
        for i in range(len(final_qual_sequence)):
            if ord(final_qual_sequence[i]) - 33 < n_max_qual_int:
                temp_read_list[i] = 'N'
        final_read_sequence = "".join(temp_read_list)


    return final_read_sequence, final_qual_sequence, cigar_string
# --- Main simulation logic (MODIFIED for Option 3 with explicit strand and correct error model usage) ---
def main():
    parser = argparse.ArgumentParser(description='Generate sequencing reads with errors and specific site mutations based on provided BED coordinates.')
    parser.add_argument('-e', '--error_model', required=True,
                        help='Path to the sequencing error model file (pickle).')
    parser.add_argument('-R', '--read_length_model', type=int, default=98,
                        help='**CRITICAL:** This must match the length of reads in your BED files. This length is used for the internal error model (default: 100).')
    parser.add_argument('-r', '--rescale_error', type=float, default=None,
                        help='Rescale the overall error rate to this value. If not specified, uses model average. (e.g., 0.01 for 1%% error)')
    parser.add_argument('-s', '--rescale_qual', action='store_true',
                        help='If set, quality scores will be rescaled to reflect the rescaled error rate.')
    parser.add_argument('-f', '--fasta', required=True,
                        help='Path to the reference FASTA file.')
    parser.add_argument('-o1', '--output_fastq1', required=True,
                        help='Output FASTQ file for Read 1.')
    parser.add_argument('-o2', '--output_fastq2', required=True,
                        help='Output FASTQ file for Read 2.')
    parser.add_argument('-b1', '--bed_file_r1', required=True,
                        help='Path to a BED file specifying Read 1 coordinates (name field must match R2, includes strand).')
    parser.add_argument('-b2', '--bed_file_r2', required=True,
                        help='Path to a BED file specifying Read 2 coordinates (name field must match R1, includes strand).')
    parser.add_argument('-m', '--mutation_bed_file', default=None,
                        help='Optional: Path to a BED file specifying genomic regions for forced mutations (e.g., SNPs, indels).')
    parser.add_argument('-M', '--mutation_rate', type=float, default=0.02,
                        help='Mutation rate (0-1) for specific sites provided in mutation_bed_file. (default: 0.02)')
    
    args = parser.parse_args()

    # Load FASTA reference
    print(f"Loading reference FASTA from {args.fasta}...")
    reference_sequences = load_fasta_as_dict(args.fasta)
    if not reference_sequences:
        print("Error: No sequences found in the FASTA file.")
        sys.exit(1)
    print(f"Loaded {len(reference_sequences)} chromosomes.")

    # Load Read 1 coordinates from BED file
    print(f"Loading Read 1 coordinates from BED file: {args.bed_file_r1}...")
    r1_coords = load_bed_file_by_name(args.bed_file_r1)
    if not r1_coords:
        print("Error: No Read 1 coordinates found in the specified BED file. Exiting.")
        sys.exit(1)
    print(f"Loaded {len(r1_coords)} Read 1 entries.")

    # Load Read 2 coordinates from BED file
    print(f"Loading Read 2 coordinates from BED file: {args.bed_file_r2}...")
    r2_coords = load_bed_file_by_name(args.bed_file_r2)
    if not r2_coords:
        print("Error: No Read 2 coordinates found in the specified BED file. Exiting.")
        sys.exit(1)
    print(f"Loaded {len(r2_coords)} Read 2 entries.")

    # Load mutation BED file if provided (for additional mutation hotspots)
    mutation_sites = None
    if args.mutation_bed_file:
        print(f"Loading general mutation sites from BED file: {args.mutation_bed_file}...")
        mutation_sites = load_general_mutation_bed(args.mutation_bed_file)
        if not mutation_sites:
            print("Warning: No mutation sites found in the provided general mutation BED file.")
        else:
            print(f"Loaded general mutation sites for {len(mutation_sites)} chromosomes.")


    # Initialize ReadContainer
    read_container = ReadContainer(
        read_len=args.read_length_model, # This read_len is for model adaptation and internal loops
        error_model=args.error_model,
        rescaled_error=args.rescale_error,
        rescale_qual=args.rescale_qual,
    )

    # Open output FASTQ files
    try:
        f_out1 = open(args.output_fastq1, 'w')
        f_out2 = open(args.output_fastq2, 'w')
    except IOError as e:
        print(f"Error opening output FASTQ file(s): {e}")
        sys.exit(1)

    print(f"Generating paired reads based on BED coordinates. All reads are expected to be of length {args.read_length_model}.")
    start_time = time.time()
    
    processed_pairs = 0

    # Iterate through Read 1 entries and try to find a matching Read 2
    for read_name, r1_rec in r1_coords.items():
        if read_name not in r2_coords:
            print(f"Warning: No matching Read 2 found for '{read_name}'. Skipping this Read 1 entry.")
            continue
        
        r2_rec = r2_coords[read_name]

        # --- Process Read 1 ---
        r1_chrom = r1_rec.chrom
        r1_start_0based = r1_rec.start
        r1_end_0based = r1_rec.end
        r1_strand_char = r1_rec.strand
        current_read_length_r1 = r1_end_0based - r1_start_0based

        # CRITICAL CHECK: Ensure read length from BED matches model read length
        if current_read_length_r1 != args.read_length_model:
            print(f"Error: Read '{read_name}' (R1) has length {current_read_length_r1} from BED, but --read_length_model is {args.read_length_model}. All reads must match --read_length_model. Skipping.")
            continue

        ref_seq_r1 = reference_sequences.get(r1_chrom)
        if not ref_seq_r1:
            print(f"Warning: Chromosome {r1_chrom} not found in FASTA for read '{read_name}' (R1). Skipping pair.")
            continue
        
        if not (0 <= r1_start_0based < r1_end_0based <= len(ref_seq_r1)):
             print(f"Warning: R1 coordinates {r1_chrom}:{r1_start_0based}-{r1_end_0based} out of bounds for chromosome length {len(ref_seq_r1)}. Skipping read pair {read_name}.")
             continue

        read_sequence_r1_orig = ref_seq_r1[r1_start_0based:r1_end_0based]
        
        is_reverse_strand_r1 = (r1_strand_char == '-')
        if is_reverse_strand_r1:
            read_sequence_r1_orig = reverse_complement(read_sequence_r1_orig)
        
        # Apply forced mutations *before* sequencing errors
        mutated_sequence_r1 = apply_forced_mutations(
            read_sequence_r1_orig,
            r1_chrom,
            r1_start_0based,
            mutation_sites, # This is the external mutation_sites BED
            args.mutation_rate
        )
        # --- NEW STEP: Handle 'N' bases before passing to error model ---
        # Store original 'N' positions
        original_n_positions_r1 = [i for i, base in enumerate(mutated_sequence_r1) if base.upper() == 'N']
        # Temporarily replace 'N's with 'A' (or any other valid nucleotide)
        # It's important to use .upper() for consistency with NUC_IND
        sequence_for_error_model_r1 = "".join(['A' if b.upper() == 'N' else b.upper() for b in mutated_sequence_r1])


        # Generate sequencing errors and qualities using ReadContainer's method
        # Corrected call to get_sequencing_errors
        qual_scores_r1_char_str, sequencing_errors_r1_list = read_container.get_sequencing_errors(
            sequence_for_error_model_r1, # Pass the N-free sequence here
            is_reverse_strand_r1
        )
        print(f"DEBUG: R1 Qual String (first 10, last 10): {qual_scores_r1_char_str[:10]}...{qual_scores_r1_char_str[-10:]}")
        print(f"DEBUG: R1 Qual String Length: {len(qual_scores_r1_char_str)}")
        modified_read_sequence_r1, final_qual_r1, final_cigar_r1 = apply_sequencing_errors_and_format_read(
            mutated_sequence_r1, # Pass the original mutated_sequence_r1 for reference in apply_sequencing_errors... if needed for CIGAR/ref, but for *final read char generation* it takes the original sequence
            # IMPORTANT: The first argument to apply_sequencing_errors_and_format_read should be the *original sequence* before any sequencing errors (but after forced mutations).
            # The current `modified_read_sequence_r1` in the `apply_sequencing_errors_and_format_read` function will be built upon this.
            # So, you should pass `mutated_sequence_r1` (which still has its N's)
            qual_scores_r1_char_str, # These qualities are generated assuming A for N, but that's okay
            sequencing_errors_r1_list,
            current_read_length_r1,
            read_container
        )

        # --- NEW STEP: Restore 'N's based on original positions OR rely on n_max_qual ---
        # The apply_sequencing_errors_and_format_read function already handles n_max_qual.
        # If an 'N' in the original `mutated_sequence_r1` was changed to 'A' for error modeling,
        # and its quality was high enough, it would appear as 'A' in the final read.
        # If you *always* want 'N's to appear as 'N's and not be affected by sequencing errors or quality,
        # you need to explicitly put them back *after* apply_sequencing_errors_and_format_read.

        # Option A: Restore 'N's at original positions (if you want 'N's to always be 'N's regardless of quality)
        final_read_list_r1 = list(modified_read_sequence_r1)
        for pos in original_n_positions_r1:
            if pos < len(final_read_list_r1): # Ensure position is within bounds of final read
                final_read_list_r1[pos] = 'N'
        modified_read_sequence_r1 = "".join(final_read_list_r1)

        # --- Process Read 2 ---
        r2_chrom = r2_rec.chrom
        r2_start_0based = r2_rec.start
        r2_end_0based = r2_rec.end
        r2_strand_char = r2_rec.strand
        current_read_length_r2 = r2_end_0based - r2_start_0based

        # CRITICAL CHECK: Ensure read length from BED matches model read length
        if current_read_length_r2 != args.read_length_model:
            print(f"Error: Read '{read_name}' (R2) has length {current_read_length_r2} from BED, but --read_length_model is {args.read_length_model}. All reads must match --read_length_model. Skipping.")
            continue

        ref_seq_r2 = reference_sequences.get(r2_chrom)
        if not ref_seq_r2:
            print(f"Warning: Chromosome {r2_chrom} not found in FASTA for read '{read_name}' (R2). Skipping pair.")
            continue

        if not (0 <= r2_start_0based < r2_end_0based <= len(ref_seq_r2)):
             print(f"Warning: R2 coordinates {r2_chrom}:{r2_start_0based}-{r2_end_0based} out of bounds for chromosome length {len(ref_seq_r2)}. Skipping read pair {read_name}.")
             continue

        read_sequence_r2_orig = ref_seq_r2[r2_start_0based:r2_end_0based]
        
        is_reverse_strand_r2 = (r2_strand_char == '-')
        if is_reverse_strand_r2:
            read_sequence_r2_orig = reverse_complement(read_sequence_r2_orig)

        # Apply forced mutations *before* sequencing errors
        mutated_sequence_r2 = apply_forced_mutations(
            read_sequence_r2_orig,
            r2_chrom,
            r2_start_0based,
            mutation_sites,
            args.mutation_rate
        )

# --- NEW STEP: Handle 'N' bases before passing to error model ---
        original_n_positions_r2 = [i for i, base in enumerate(mutated_sequence_r2) if base.upper() == 'N']
        sequence_for_error_model_r2 = "".join(['A' if b.upper() == 'N' else b.upper() for b in mutated_sequence_r2])

        # Generate sequencing errors and qualities using ReadContainer's method
        # Corrected call to get_sequencing_errors
        qual_scores_r2_char_str, sequencing_errors_r2_list = read_container.get_sequencing_errors(
            sequence_for_error_model_r2, # Pass the N-free sequence here
            is_reverse_strand_r2
        )
        print(f"DEBUG: R2 Qual String (first 10, last 10): {qual_scores_r2_char_str[:10]}...{qual_scores_r2_char_str[-10:]}")
        print(f"DEBUG: R2 Qual String Length: {len(qual_scores_r2_char_str)}")
        modified_read_sequence_r2, final_qual_r2, final_cigar_r2 = apply_sequencing_errors_and_format_read(
            mutated_sequence_r2, # Pass the original mutated_sequence_r2
            qual_scores_r2_char_str,
            sequencing_errors_r2_list,
            current_read_length_r2,
            read_container
        )
        
        # --- NEW STEP: Restore 'N's based on original positions ---
        final_read_list_r2 = list(modified_read_sequence_r2)
        for pos in original_n_positions_r2:
            if pos < len(final_read_list_r2):
                final_read_list_r2[pos] = 'N'
        modified_read_sequence_r2 = "".join(final_read_list_r2)

        
        # --- Write Read 1 ---
        read_id_r1 = f"@{r1_rec.name}/1 {r1_chrom}:{r1_start_0based+1}-{r1_end_0based}_{r1_strand_char}"
        f_out1.write(f"{read_id_r1}\n{modified_read_sequence_r1}\n+\n{final_qual_r1}\n")

        # --- Write Read 2 ---
        read_id_r2 = f"@{r2_rec.name}/2 {r2_chrom}:{r2_start_0based+1}-{r2_end_0based}_{r2_strand_char}"
        f_out2.write(f"{read_id_r2}\n{modified_read_sequence_r2}\n+\n{final_qual_r2}\n")

        processed_pairs += 1

    end_time = time.time()
    print(f"\nGenerated {processed_pairs} paired reads in {end_time - start_time:.2f} seconds.")

    f_out1.close()
    f_out2.close()

if __name__ == '__main__':
    main()