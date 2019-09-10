// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

import os
import sys
from enum import Enum
from itertools import combinations
from itertools import product
import numpy as np
from dialogue.refer_resolution import ConflictLinker
import torch
from model.metric import evaluate_symbol_score, evaluate_bleu_score
from random import choice


class ConflictMode(Enum):
    # WARNING: NoConflict is just a placeholder for easy usage.
    # You should assign all other pairs as non-conflict pairs.
    NoConflict = 0
    ConflictAndReplace = 1


def permutation(arr, begin, end, out_arr):
    """
    permutation arr from begin to end, then put index pair into out_arr
    :param arr:
    :param begin:
    :param end:
    :param out_arr: [[arr_i, ],[]]
    :return:
    """
    if begin >= end:
        # ready to put pair into out_arr
        out_arr.append(arr.copy())
    else:
        i = begin
        for num in range(begin, end):
            arr[num], arr[i] = arr[i], arr[num]
            permutation(arr, begin + 1, end, out_arr)
            arr[num], arr[i] = arr[i], arr[num]


def prev_fol_permutation(prev_spans, fol_spans):
    """
    Given previous spans and follow-up spans, return the paired `index` array to indicate which ind_1 in previous
    is arranged to be conflict with ind_2 in follow-up spans.
    :return: paired index array, you should identify conflict mode and group them together out this function
    """
    # filter ind to permutation
    pre_inds = [ind for ind, span in enumerate(prev_spans) if span is not None]
    fol_inds = [ind for ind, span in enumerate(fol_spans) if span is not None]
    # get length of spans
    pre_len = len(pre_inds)
    fol_len = len(fol_inds)
    # take the least length
    min_len = min(pre_len, fol_len)

    pair_array = []
    if min_len == pre_len:
        combine_arr = []
        # C _ fol_len ^ min_len
        select_min_from_fol = [[fol_inds[ind] for ind in combine]
                               for combine in combinations(range(fol_len), min_len)]
        # A _ min_len ^ min_len
        for arr in select_min_from_fol:
            permutation(arr, 0, min_len, combine_arr)
        # traverse
        for decision in combine_arr:
            # (prev, fol)
            pair_array.append([(pre_inds[ind], fol_ind) for ind, fol_ind in enumerate(decision)])
    else:
        combine_arr = []
        # C _ pre_len ^ min_len
        select_min_from_prev = [[pre_inds[ind] for ind in combine]
                                for combine in combinations(range(pre_len), min_len)]
        # A _ min_len ^ min_len
        for arr in select_min_from_prev:
            permutation(arr, 0, min_len, combine_arr)
        # traverse
        for decision in combine_arr:
            # (prev, fol)
            pair_array.append([(pre_ind, fol_inds[ind]) for ind, pre_ind in enumerate(decision)])

    mode_array = [ind for ind in range(0, ConflictMode.__len__())]
    repeated_mode = [mode_array[:] for _ in range(min_len)]
    repeated_mode = product(*repeated_mode)
    pair_mode_array = []
    for mode_array in repeated_mode:
        for pair_seq in pair_array:
            pair_mode_array.append((pair_seq, mode_array))
    return pair_mode_array


class RewardCalculator(object):
    def __init__(self, pre_tokens, fol_tokens, restate_tokens,
                 pre_tags, fol_tags, restate_tags,
                 prev_spans, fol_spans, prev_valid_spans, fol_valid_spans, is_training):
        self.pre_tokens = pre_tokens
        self.fol_tokens = fol_tokens
        self.restate_tokens = " ".join(restate_tokens)

        self.pre_tags = pre_tags
        self.fol_tags = fol_tags
        self.restate_tags = restate_tags

        self.prev_spans = prev_spans
        self.fol_spans = fol_spans

        self.prev_valid_spans = prev_valid_spans
        self.fol_valid_spans = fol_valid_spans

        self.is_training = is_training

    def resolve_conflict(self, input_data):
        pair_seq, pair_mode = input_data
        conflict_map = np.zeros((len(self.fol_spans), len(self.prev_spans)), dtype=np.int)
        conflict_pair = []

        # find conflict and make decisions on fusion
        for conflict_tuple, conflict_mode in zip(pair_seq, pair_mode):
            pre_conflict_id, fol_conflict_id = conflict_tuple
            # make a decision to replace 0 with 1, or replace 1 with 0.
            # if has pronoun, should replace follow using prev
            if conflict_mode == ConflictMode.NoConflict.value:
                continue
            conflict_map[fol_conflict_id][pre_conflict_id] = 1
            conflict_pair.append((pre_conflict_id, fol_conflict_id, conflict_mode))

        linker = ConflictLinker(self.pre_tokens,
                                self.fol_tokens,
                                self.pre_tags,
                                self.fol_tags,
                                self.prev_spans,
                                self.fol_spans,
                                self.is_training)
        prev_fusion_tags, prev_fusion_tokens, fol_fusion_tags, fol_fusion_tokens, from_prev_to_fol = \
            linker.conflict_resolution(
                conflict_map,
                ""
            )
        return prev_fusion_tags, prev_fusion_tokens, fol_fusion_tags, fol_fusion_tokens, from_prev_to_fol, conflict_pair

    def calculate_conflict_prob(self, conflict_pair, conflict_prob_matrix):
        prob_base = 1.0
        for prev_conflict_id, fol_conflict_id, conflict_mode in conflict_pair:
            prob_base *= conflict_prob_matrix[prev_conflict_id, fol_conflict_id]

            # A1 conflicts with B1, so A1 doesn't conflict with others
            for fol_non_conflict_id, fol_non_conflict_span in enumerate(self.fol_spans):
                if fol_non_conflict_id != fol_conflict_id:
                    prob_base *= (1.0 - conflict_prob_matrix[prev_conflict_id, fol_non_conflict_id])

            # negative samples in follow-up
            for prev_non_conflict_id, _ in enumerate(self.prev_spans):
                if prev_non_conflict_id != prev_conflict_id:
                    prob_base *= (1.0 - conflict_prob_matrix[prev_non_conflict_id, fol_conflict_id])

        # default non conflict
        if len(conflict_pair) == 0:
            for prev_non_conflict_id, _ in enumerate(self.prev_spans):
                for fol_non_conflict_id, _ in enumerate(self.fol_spans):
                    prob_base *= (1.0 - conflict_prob_matrix[prev_non_conflict_id, fol_non_conflict_id])
        return prob_base

    def combination_reward_feedback(self, conflict_prob_matrix):
        # default return value
        default_ret = (0.0, 0.0, [])
        # permutation and combination
        # total iteration times = A _ max_len ^ min_len x possible mode
        conflict_index_pairs = prev_fol_permutation(self.prev_valid_spans, self.fol_valid_spans)

        if len(conflict_index_pairs) >= 5000:
            # print("Drop the large combinations")
            return default_ret

        # record random bleu score of this
        enum_bleu_scores = []
        enum_symbol_scores = []
        _debug_str_list = []

        predict_bleu_scores = []
        predict_symbol_scores = []
        predict_conflict_pair_list = []
        predict_conflict_prob_list = []

        fusion_results = map(self.resolve_conflict, conflict_index_pairs)

        for prev_fusion_tags, prev_fusion_tokens, fol_fusion_tags, fol_fusion_tokens, prev_to_fol, conflict_pair in fusion_results:
            # add previous fusion to bleu score
            prev_bleu_score = evaluate_bleu_score(prev_fusion_tokens, self.restate_tokens)
            fol_bleu_score = evaluate_bleu_score(fol_fusion_tokens, self.restate_tokens)

            prev_sym_score = evaluate_symbol_score(prev_fusion_tags, self.restate_tags)
            fol_sym_score = evaluate_symbol_score(fol_fusion_tags, self.restate_tags)

            enum_bleu_scores.append(prev_bleu_score)
            enum_symbol_scores.append(prev_sym_score)
            _debug_str_list.append(prev_fusion_tokens)

            # add follow fusion to bleu score
            enum_bleu_scores.append(fol_bleu_score)
            enum_symbol_scores.append(fol_sym_score)
            _debug_str_list.append(fol_fusion_tokens)

            predict_conflict_pair_list.append(conflict_pair)

            if prev_to_fol:
                predict_bleu_scores.append(fol_bleu_score)
                predict_symbol_scores.append(fol_sym_score)
            else:
                predict_bleu_scores.append(prev_bleu_score)
                predict_symbol_scores.append(prev_sym_score)

            if conflict_prob_matrix is not None:
                predict_conflict_prob_list.append(self.calculate_conflict_prob(conflict_pair, conflict_prob_matrix))

        if len(enum_bleu_scores) > 0:

            # find conflict pairs which satisfies max bleu score
            predict_scores = [0.5 * bleu_score + 0.5 * sym_score for bleu_score, sym_score in
                              zip(predict_bleu_scores, predict_symbol_scores)]
            max_pred_score = max(predict_scores)
            inds = [ind for ind, score in enumerate(predict_scores) if score >= max_pred_score - 0.001]
            best_conflict = []

            # choose one best to calculate conflicting probability
            rand_ind = choice(inds)

            for prev_conflict_id, fol_conflict_id, conflict_mode in predict_conflict_pair_list[rand_ind]:

                best_conflict.append((self.prev_spans[prev_conflict_id], self.fol_spans[fol_conflict_id], 1))

                # A1 conflicts with B1, so A1 doesn't conflict with B2/B3/B4 ...
                for fol_non_conflict_id, fol_non_conflict_span in enumerate(self.fol_spans):
                    if fol_non_conflict_id != fol_conflict_id:
                        best_conflict.append(
                            (self.prev_spans[prev_conflict_id], self.fol_spans[fol_non_conflict_id], 0))

                # negative samples in follow-up
                for prev_non_conflict_id, _ in enumerate(self.prev_spans):
                    if prev_non_conflict_id != prev_conflict_id:
                        best_conflict.append(
                            (self.prev_spans[prev_non_conflict_id], self.fol_spans[fol_conflict_id], 0))

            # default non conflict
            if len(predict_conflict_pair_list[rand_ind]) == 0:
                for prev_non_conflict_id, _ in enumerate(self.prev_spans):
                    for fol_non_conflict_id, _ in enumerate(self.fol_spans):
                        best_conflict.append(
                            (self.prev_spans[prev_non_conflict_id], self.fol_spans[fol_non_conflict_id], 0))

            if conflict_prob_matrix is not None:
                # enumerate bleu scores and find max
                mean_pred_symbol_bleu = sum([prob * score for prob, score in zip(predict_conflict_prob_list,
                                                                                 predict_scores)]) / sum(
                    predict_conflict_prob_list)
            else:
                mean_pred_symbol_bleu = np.mean(predict_scores)

            reinforce_reward = mean_pred_symbol_bleu

            return max_pred_score, reinforce_reward, best_conflict
        else:
            return default_ret
