# Find Long Common String between Precedent Query & Follow-up Query
import numpy as np
import json
import spacy
from model.util import RewardCalculator

nlp = spacy.load('en_core_web_sm')
stop_words = ["a", "an", "the", "so", "?", ",", "."]


def find_long_common_string(label_tokens, unlabel_tokens):
    """

    :param label_tokens: golden query tokens res which is labeled by workers
    :param unlabel_tokens: previous query + "@@Sep@@" + follow-up query
    :return:
    """
    len_r = len(label_tokens)
    len_n = len(unlabel_tokens)
    record_arr = np.zeros((len_r, len_n), dtype=np.int)

    ret = []
    z = 0

    i = 0
    while i < len_r:
        j = 0
        while j < len_n:
            if label_tokens[i] == unlabel_tokens[j]:
                if i == 0 or j == 0:
                    record_arr[i, j] = 1
                else:
                    record_arr[i, j] = record_arr[i - 1, j - 1] + 1

                if record_arr[i, j] > z:
                    z = record_arr[i, j]
                    ret = label_tokens[i - z + 1:i]
                elif record_arr[i, j] == z:
                    ret = ret + label_tokens[i - z + 1:i]
            else:
                record_arr[i, j] = 0
            j += 1
        i += 1

    # standard split position
    align_label_axis = [max(row) for row in record_arr]

    # align un_label data indicates how to split in source sentence
    sep_sym_ind = unlabel_tokens.index("@@Sep@@")

    # remove overlap
    for row_index, row in enumerate(record_arr):
        for col_index, val in enumerate(row):
            if val < align_label_axis[row_index]:
                record_arr[row_index, col_index] = 0

    record_arr = record_arr.T

    align_unlabel_axis = [max(col) for col in record_arr]

    pre_entities = np.zeros(sep_sym_ind, dtype=int)
    fol_entities = np.zeros((len_n - sep_sym_ind - 1), dtype=int)

    last_flag = 0
    for ind, flag in enumerate(align_unlabel_axis):
        # the first of follow-up should be divided
        if ind == sep_sym_ind or \
                ind == sep_sym_ind + 1 or \
                ind == 0:
            pass
        elif ind > sep_sym_ind:
            # follow
            if flag == 0 and last_flag != 0:
                fol_entities[ind - sep_sym_ind - 1] = 1
            elif flag == 1 and last_flag == 0:
                fol_entities[ind - sep_sym_ind - 1] = 1
            elif flag != 0 and flag != last_flag + 1:
                fol_entities[ind - sep_sym_ind - 1] = 1
        else:
            # pre
            if flag == 0 and last_flag != 0:
                pre_entities[ind] = 1
            elif flag == 1 and last_flag == 0:
                pre_entities[ind] = 1
            elif flag != 0 and flag != last_flag + 1:
                pre_entities[ind] = 1

        last_flag = flag
    return list(pre_entities), list(fol_entities)


def display(pre_snippet, pre_tokens, fol_snippet, fol_tokens):
    # print to debug
    print("Previous:")
    for ind, flag in enumerate(pre_snippet):
        if flag != 0:
            print("|", end=' ')
        print(pre_tokens[ind], end=' ')
    print("\nFollow-up:")
    for ind, flag in enumerate(fol_snippet):
        if flag != 0:
            print("|", end=' ')
        print(fol_tokens[ind], end=' ')
    print("\n--------------------------")


def remove_stopwords(query_obj):
    # remove stop words in query object
    ind = 0
    while ind < len(query_obj.utterance):
        token = query_obj.utterance[ind]
        if token in stop_words:
            del query_obj.utterance[ind]
            del query_obj.tags[ind]
        else:
            ind += 1


def find_common_snippet(single_obj):
    remove_stopwords(single_obj["prev"])
    remove_stopwords(single_obj["follow"])
    remove_stopwords(single_obj["restate"])

    un_label_tokens = single_obj["prev"].utterance + ["@@Sep@@"] + single_obj["follow"].utterance
    restate_tokens = single_obj["restate"].utterance
    sni_pre, sni_fol = find_long_common_string(restate_tokens, un_label_tokens)

    # visualize result
    # display(sni_pre, single_obj["prev"].utterance,
    #         sni_fol, single_obj["follow"].utterance)

    single_obj["prev_snippet"] = [str(tok) for tok in sni_pre]
    single_obj["follow_snippet"] = [str(tok) for tok in sni_fol]
    single_obj["conflicts"] = get_best_conflicts(single_obj)

    # if no conflicts, clear the snippet
    if len(single_obj["conflicts"]) == 0:
        single_obj["prev_snippet"] = ['0'] * len(sni_pre)
        single_obj["follow_snippet"] = ['0'] * len(sni_fol)
    return single_obj


def get_best_conflicts(single_obj):
    """
    export the training data for conflicting detection
    :return: training line for self tripe
    """
    prev_snippet, fol_snippet = single_obj["prev_snippet"], single_obj["follow_snippet"]
    prev_tokens, fol_tokens, restate_tokens = single_obj["prev"].utterance, single_obj["follow"].utterance, single_obj["restate"].utterance
    prev_tags, fol_tags, restate_tags = single_obj["prev"].tags, single_obj["follow"].tags, single_obj["restate"].tags

    # previous phrase list & follow-up phrase list
    prev_start_end = []
    fol_start_end = []

    start = 0
    # cut into span start/end representation
    for pos_ind in range(len(prev_snippet) + 1):
        # pos of SPLIT
        if pos_ind == len(prev_snippet):
            prev_start_end.append((start, pos_ind))
            break
        elif prev_snippet[pos_ind] != '0' and pos_ind != start:
            prev_start_end.append((start, pos_ind))
            start = pos_ind

    start = 0
    # cut into span start/end representation
    for pos_ind in range(len(fol_snippet) + 1):
        # pos of SPLIT
        if pos_ind == len(fol_snippet):
            fol_start_end.append((start, pos_ind))
            break
        elif fol_snippet[pos_ind] != '0' and pos_ind != start:
            fol_start_end.append((start, pos_ind))
            start = pos_ind

    #################################################################
    calculator = RewardCalculator(pre_tokens=prev_tokens,
                                  fol_tokens=fol_tokens,
                                  restate_tokens=restate_tokens,
                                  pre_tags=prev_tags,
                                  fol_tags=fol_tags,
                                  restate_tags=restate_tags,
                                  prev_spans=prev_start_end,
                                  fol_spans=fol_start_end,
                                  is_training=True)

    # mean reward
    conflict_confidence, _, best_conflict = calculator.combination_reward_feedback(None)
    # best conflict could be none
    if best_conflict is None:
        best_conflict = []

    # use genetic algorithm to get better pre-training result.
    return best_conflict
