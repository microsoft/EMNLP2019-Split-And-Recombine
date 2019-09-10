// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

import itertools
import json
import re
import string
from copy import deepcopy
import spacy
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
from nltk.stem.lancaster import LancasterStemmer
from spacy.matcher import PhraseMatcher
from spacy.symbols import ORTH, LEMMA, POS
from typing import Dict
from dialogue.bind_define import StandardSpan
from dialogue.constant import *
from dialogue.knowledge_bind import KnowledgeBinder
from nltk.corpus import stopwords
from typing import Optional
from collections import Counter

punc = string.punctuation

nlp = spacy.load('en_core_web_sm', disable=["tagger", "parser"])
stemmer = LancasterStemmer()

# if strict, disable fuzzy matching
strict_match = False


def get_val_actual_type(col_index, header_types):
    """
    get actual type of column
    :param col_index:
    :param header_types:
    :return:
    """
    # True means ClassType
    if header_types[col_index] == SubType.measure:
        return Bind.ValueNumber, header_types[col_index]
    elif header_types[col_index] == SubType.date:
        return Bind.ValueDate, header_types[col_index]
    else:
        return Bind.ValueText, header_types[col_index]


def get_col_actual_type(col_index, header_types):
    """
    get actual type of column
    :param col_index:
    :param header_types:
    :return:
    """
    # True means ClassType
    if header_types[col_index] == SubType.measure:
        return Bind.ColumnNumber, header_types[col_index]
    elif header_types[col_index] == SubType.date:
        return Bind.ColumnDate, header_types[col_index]
    else:
        return Bind.ColumnText, header_types[col_index]


def get_entities_out_table(doc, stop_flag, header_types, table_headers):
    """
    get date/number entities which has never occurred in the table
    :param doc:
    :param stop_flag:
    :param header_types:
    :param table_headers:
    :return:
    """
    entities = []

    # alignment from tokens and text
    for entity in doc.ents:
        entity_type = SubType(entity.label_)
        candidate_headers = [header for index, header in enumerate(table_headers)
                             if header_types[index] == entity_type]

        # get rid of JJR (more/less than)
        pos_tags = [doc[index].tag_ for index in range(entity.start, entity.end)]

        start = entity.start
        end = entity.end

        if "JJR" in pos_tags or "RBR" in pos_tags:
            if "IN" in pos_tags:
                # more than ?
                # more expensive than ?
                # After preposition
                start = entity.start + pos_tags.index("IN") + 1
            else:
                for token in doc:
                    print(token.text)

        origin_text = doc.text[entity.start_char:entity.end_char]

        # no number, continue
        if len(re.findall("\d", origin_text)) == 0:
            continue

        data = {
            "origin": doc.text[entity.start_char:entity.end_char],  # no mapping
            "header": candidate_headers,  # find headers
            "entity": entity_type.value,

            "type": None,
            "start": start,
            "len": end - start
        }

        if True not in stop_flag[start:end]:
            data_type = None

            if entity_type == SubType.date:
                # get value date
                data_type = Bind.ValueDate

            elif entity_type == SubType.cardinal or \
                    entity_type == SubType.ordinal or \
                    entity_type == SubType.money or \
                    entity_type == SubType.quantity or \
                    entity_type == SubType.percent:
                # get value number
                data_type = Bind.ValueNumber

            if data_type is not None:
                for index in range(start, end):
                    stop_flag[index] = True
                data["type"] = data_type.value
                entities.append([data])

    for token_index, token in enumerate(doc):
        if stop_flag[token_index] is True:
            continue
        data = None
        content = token.text
        is_date = False
        if re.fullmatch("^[12]\d{3}$", content) is not None:
            candidate_headers = [header for index, header in enumerate(table_headers)
                                 if header_types[index] == SubType.date]
            if len(candidate_headers) > 0:
                data = {
                    "origin": token.text,  # no mapping
                    "header": candidate_headers,  # find headers
                    "entity": SubType.date.value,
                    "type": Bind.ValueDate.value,
                    "start": token_index,
                    "len": 1
                }
                is_date = True
        if not is_date and re.fullmatch("^[-–]?[0-9]([0-9])*((\.[0-9]+)|((,\d{3})+))?$", content) is not None:
            candidate_headers = [header for index, header in enumerate(table_headers)
                                 if header_types[index] == SubType.cardinal]
            data = {
                "origin": token.text,  # no mapping
                "header": candidate_headers,  # find headers
                "entity": SubType.cardinal.value,
                "type": Bind.ValueNumber.value,
                "start": token_index,
                "len": 1
            }
        if data is not None:
            entities.append([data])
            stop_flag[token_index] = True

    if len(entities) > 0:
        return entities
    else:
        return None


def fuzzy_matching_tokens(candidate_tokens, origin_tokens, choices, origin_choices, common_words):
    # stemming
    # candidate_tokens = [stemmer.stem(token) for token in candidate_tokens]
    match_scores = []

    # remove repeat data in origin choices
    np_origin_choices = []
    for o_choice in origin_choices:
        if o_choice not in np_origin_choices:
            np_origin_choices.append(o_choice)
        else:
            np_origin_choices.append("")

    for token_index, token in enumerate(candidate_tokens):
        if token not in string.punctuation and \
                token not in stopwords.words('english') and \
                not re.fullmatch("^\d+$", token):

            if token in common_words:
                best_matches = process.extractBests(origin_tokens[token_index], np_origin_choices,
                                                    scorer=fuzz.token_set_ratio,
                                                    score_cutoff=90)
            else:
                token = stemmer.stem(token)
                # take best match of token, no order considering
                best_matches = process.extractBests(token, choices,
                                                    scorer=fuzz.token_set_ratio,
                                                    score_cutoff=90)
                # restore original token
                for match_idx, match in enumerate(best_matches):
                    choice_index = choices.index(match[0])
                    origin_choice = np_origin_choices[choice_index]
                    best_matches[match_idx] = (origin_choice, match[1])

            if 0 < len(best_matches) <= 4:
                match_scores.append(best_matches)
            else:
                match_scores.append([('', 0)])
        else:
            match_scores.append([('', 0)])

    # calculate match score
    all_combinations = [list(combination) for combination in (itertools.product(*match_scores))]

    # select which has the maximum score (less tag, longest cover)
    best_score = - float("inf")
    best_combinations = []

    # simple strategy for taking over the one has the most probability
    for combination in all_combinations:
        tag_list = [tag[0] for tag in combination if tag is not None]
        cover_len = len(tag_list)
        category_len = len(set(tag_list))
        # measure similarity in cover, cover more means more possible
        score = cover_len - category_len
        if score > best_score:
            best_score = score
        best_combinations.append((combination, score))

    best_combinations = [comb[0] for comb in best_combinations if comb[1] == best_score]

    best_dict_combinations = []
    for combination in best_combinations:
        temp_list = []
        for index, ele_tag in enumerate(combination):
            if ele_tag[1] != 0:
                temp_obj = {
                    "start": index,
                    "len": 1,
                    "match": ele_tag[0]
                }
                temp_list.append(temp_obj)
        best_dict_combinations.append(temp_list)

    #  merge continuous tags
    for combination in best_dict_combinations:
        start = 0
        limit = len(combination)
        while start < limit:
            current_ele = combination[start]
            if start + 1 < limit:
                next_ele = combination[start + 1]
                # string match
                if current_ele["match"] == next_ele["match"]:
                    # char distance less than 3
                    next_ele["len"] += next_ele["start"] - current_ele["start"]
                    next_ele["start"] = current_ele["start"]
                    combination[start] = None
            start += 1

        while None in combination:
            combination.remove(None)

    best_combinations = {}
    # calculating length matching score in every row of best_dict_combinations
    for combination in best_dict_combinations:
        for ele in combination:
            start = ele["start"]
            end = start + ele["len"]
            candidate_length = len(" ".join(candidate_tokens[start:end]))
            match_length = len(ele["match"])
            ele_score = abs(match_length - candidate_length)
            # if start in best_combinations
            if start not in best_combinations:
                best_combinations[start] = [(ele, ele_score)]
            else:
                if ele_score == best_combinations[start][0][1] and \
                        ele["match"] not in [tup[0]["match"] for tup in best_combinations[start]]:
                    best_combinations[start].append((ele, ele_score))
                elif ele_score < best_combinations[start][0][1]:
                    best_combinations[start] = [(ele, ele_score)]

    # record the best one
    best_dict_combinations = []
    for combination in best_combinations.values():
        best_dict_combinations.append([tup[0] for tup in combination])

    return best_dict_combinations


def transfer_to_tags(binding_data):
    # TODO: here we only take one
    span: Optional[StandardSpan] = None
    for match in binding_data["matches"]:
        utterance = deepcopy(binding_data["tokens"])
        mask = [0] * len(binding_data["tokens"])
        # binding sequence
        for tag_span in match:
            start = tag_span["start"]
            if mask[start] == 0:
                mask[start] = 1
                utterance[start] = " ".join(utterance[start:start + tag_span["len"]])
                begin = 1
                while begin < tag_span["len"]:
                    mask[start + begin] = 2
                    begin += 1

        tag_span_index = 0
        tokens = []
        tags = []
        for i in range(len(mask)):
            if mask[i] == 0:
                tags.append(None)
                tokens.append(utterance[i])
            elif mask[i] == 1:
                old_tag = match[tag_span_index]
                tag_span_index += 1
                new_tag = StandardSymbol(
                    origin=old_tag["origin"],
                    header=old_tag["header"],
                    sub_type=old_tag["entity"],
                    class_type=old_tag["type"]
                )
                tags.append(new_tag)
                tokens.append(utterance[i])
        span = StandardSpan(utterance=tokens,
                            tags=tags)
        break
    return span


class BindingExecutor(object):

    def __init__(self, table_headers, header_types, table_rows, table_title_line):
        """
        :param table_headers: columns of table
        :param table_rows: values of table
        :param header_types: types of header
        :param table_title_line: the title of table, which seems equal to COL in table
        :return: match list
        """
        # add all values of table to choices
        header_len = len(table_headers)
        choices = [header.lower() for header in table_headers]
        # all choices
        self.val_to_col = {}
        if table_rows is not None:
            records = {}
            for row in table_rows:
                for col_ind, value in enumerate(row):
                    value = str(value).lower()
                    if value not in records:
                        val_ind = len(choices)
                        choices.append(value)
                        # place holder
                        records[value] = val_ind
                        self.val_to_col[val_ind] = [col_ind]
                    else:
                        val_ind = records[value]
                        self.val_to_col[val_ind].append(col_ind)

        self.header_types = header_types
        self.header_len = header_len
        self.table_headers = [header.lower() for header in table_headers]
        self.original_tokens = choices
        self.fuzz_choices = []
        # title just for fuzzywuzzy
        self.table_title = table_title_line.strip().lower()

        self.stop_vocab = list(set(stopwords.words('english')))

        non_stop_words = []

        # define own non stop words
        matcher_tokens = []
        matcher_mapping_type = {}

        max_len = 10
        special_cases = []
        # token position in choices
        for index, word in enumerate(choices):
            doc = nlp(word)
            tokens = [w.lemma_.lower() if w.lemma_ != '-PRON-' else w.text.lower() for w in doc]

            # add lemma not original token
            # self.fuzz_choices.append(" ".join([porter_stemmer.stem(token) for token in tokens]))
            token_text = " ".join([stemmer.stem(token) for token in tokens])

            if token_text not in self.fuzz_choices:
                self.fuzz_choices.append(token_text)
            else:
                self.fuzz_choices.append("")

            for token in tokens:
                non_stop_words.append(token)

            if len(tokens) >= max_len and len(choices[index - 1]) < 20:
                # too long, to match it, we should add special case in vocab
                # actual length is not long
                # TODO: space in words can't match
                special_cases.append((word, [{ORTH: word, LEMMA: word, POS: u"NOUN"}]))

            # transfer tokens to text, lemma all text
            text = ""
            end = 0
            for token_index, token in enumerate(doc):
                while end < token.idx:
                    text += " "
                    end += 1
                # remove all punctuations
                append_text = "".join([char for char in tokens[token_index] if char not in punc])
                if append_text != string.whitespace:
                    text += tokens[token_index]
                end = token.idx + len(token.text)

            # plus 1 to avoid starting from 0
            # use tokens instead of word to avoid lemma miss
            if text != word:
                matcher_tokens.append((index + 1, text))
            matcher_tokens.append((index + 1, word))

            # set tag for binding tags
            if index < header_len:
                # default text
                matcher_mapping_type[index + 1] = Bind.ColumnText
            else:
                matcher_mapping_type[index + 1] = Bind.ValueText

        # add special case in nlp
        for word, data in special_cases:
            nlp.tokenizer.add_special_case(word, data)

        self.matcher_mapping_type = matcher_mapping_type

        # sentence
        self.matcher = PhraseMatcher(nlp.vocab, max_length=max_len)
        for index, word in matcher_tokens:
            tokens = nlp(word)
            if max_len > len(tokens) > 0:
                self.matcher.add(index, None, tokens)

        # use sub word to match, limit is 3
        if self.table_title != "":
            title_tokens = [w.lemma_.lower() if w.lemma_ != '-PRON-' else w.text.lower() for w in nlp(self.table_title)]

            for token in title_tokens:
                if token not in punc:
                    non_stop_words.append(token)

            if len(title_tokens) < max_len:
                title_nlp = [nlp(" ".join(title_tokens))]
            else:
                title_nlp = []
            # add whole token into title nlp
            limit = 3
            for start in range(len(title_tokens)):
                end = min(start + limit, len(title_tokens))
                while start < end:
                    has_special_word = False
                    temp_tokens = title_tokens[start:end]
                    for token in temp_tokens:
                        if token not in self.stop_vocab and \
                                token not in punc and \
                                not re.fullmatch("^\d+$", token):
                            has_special_word = True
                            break
                    # no special, not added
                    if has_special_word:
                        sub_title_span = " ".join(temp_tokens)
                        if sub_title_span not in non_stop_words:
                            title_nlp.append(nlp(sub_title_span))
                    start += 1
                if end == len(title_tokens) and start + limit > len(title_tokens):
                    break

            if len(title_tokens) > 0:
                self.title_tokens = title_tokens

                choices.append(self.table_title)
                # add Table type
                self.matcher.add(len(choices), None, *title_nlp)
                self.matcher_mapping_type[len(choices)] = Bind.Table
                self.fuzz_choices.append(" ".join(title_tokens))

        self.non_stop_words = non_stop_words
        self.knowledge_binder = KnowledgeBinder(nlp)

    def binding_sequence(self, sentence):
        # use phrase match
        nlp_tokens = nlp(sentence)
        stop_flag = [False] * len(nlp_tokens)
        knowledge_stop_flag = [False] * len(nlp_tokens)

        pure_tokens = [''] * len(nlp_tokens)
        # for exact match
        sen_tokens = [''] * len(pure_tokens)

        # replace tokens with pure string
        tokens = [token.text.lower() for token in nlp_tokens]
        # data structure
        data = {
            "tokens": tokens,
            "matches": []
        }

        # tokenizer sentence and add them into
        for match_id, token in enumerate(nlp_tokens):
            if token.text not in self.non_stop_words:
                stop_flag[match_id] = token.is_stop
            pure_tokens[match_id] = token.lemma_.lower() if token.lemma_ != '-PRON-' else token.text.lower()
            sen_tokens[match_id] = token.text.lower()

        # define maximal match length
        match_candidates = []
        temp_candidate = []
        temp_start = 0

        for match_id, token in enumerate(pure_tokens):
            if stop_flag[match_id] is False:
                temp_candidate.append(token)
            elif len(temp_candidate) > 0:
                match_candidates.append((temp_start, temp_candidate))
                temp_candidate = []
                temp_start = match_id + 1
            else:
                temp_start = match_id + 1

        if len(temp_candidate) > 0:
            match_candidates.append((temp_start, temp_candidate))

        # match exact words
        for start_index, candidate_tokens in match_candidates:
            doc = nlp(" ".join(candidate_tokens))
            match_flag = [False] * len(doc)

            matches = self.matcher(doc)

            # sort them using start index and length, ascending for start, descending for length
            matches = sorted(matches, key=lambda x: (x[1], -x[2]))

            if len(matches) > 0:
                for match_id, m_start, m_end in matches:
                    # if value, we should know its column
                    match_type = self.matcher_mapping_type[match_id]

                    if match_type == Bind.ValueText:
                        col_index = self.val_to_col[match_id - 1][0]
                        value_type, entity_type = get_val_actual_type(col_index, self.header_types)

                        append_data = []
                        col_names = set()
                        for col_index in self.val_to_col[match_id - 1]:
                            col_names.add(self.original_tokens[col_index])

                        # if value_type == Bind.ValueNumber or value_type == Bind.ValueDate:
                        append_data.append({
                            "origin": self.original_tokens[match_id - 1],
                            "header": sorted(list(col_names)),
                            "entity": entity_type.value,
                            "type": value_type.value,
                            "start": start_index + m_start,
                            "len": m_end - m_start
                        })
                    elif match_type == Bind.ColumnText:
                        col_type, _ = get_col_actual_type(match_id - 1, self.header_types)
                        append_data = {
                            "origin": self.original_tokens[match_id - 1],
                            "header": self.original_tokens[match_id - 1],
                            "entity": self.header_types[match_id - 1].value,
                            "type": col_type.value,
                            "start": start_index + m_start,
                            "len": m_end - m_start
                        }
                    else:
                        append_data = {
                            "origin": self.original_tokens[match_id - 1],
                            "header": None,
                            "entity": None,
                            "type": match_type.value,
                            "start": start_index + m_start,
                            "len": m_end - m_start
                        }

                    # if match exactly
                    if True not in match_flag[m_start:m_end]:
                        if isinstance(append_data, list):
                            data["matches"].append(append_data)
                        else:
                            data["matches"].append([append_data])

                    for flag_index in range(m_start, m_end):
                        match_flag[flag_index] = True

                    for ass_index in range(start_index + m_start, start_index + m_end):
                        knowledge_stop_flag[ass_index] = True

        knowledge_tags = self.knowledge_binder.get_knowledge_bindings(nlp_tokens, knowledge_stop_flag)
        if knowledge_tags is not None:
            data["matches"].extend(knowledge_tags)

        # add extra rule to mask common words
        for match_id, token in enumerate(tokens):
            stop_flag[match_id] |= knowledge_stop_flag[match_id]
            if token in self.non_stop_words:
                continue
            elif token in self.stop_vocab:
                # is stop word
                stop_flag[match_id] = True

        # find others which should use data binding
        entities = get_entities_out_table(nlp_tokens, stop_flag, self.header_types, self.table_headers)
        if entities is not None:
            data["matches"].extend(entities)

        # combinations for all ?
        exact_combinations = data["matches"]

        # add extra rule to mask common words
        for match_id, token in enumerate(tokens):
            if token in self.non_stop_words:
                continue
            elif token in self.stop_vocab:
                # is stop word
                stop_flag[match_id] = True

        # reset zero
        fuzz_candidates = []
        exact_candidates = []
        temp_candidate = []
        temp_exact_candidate = []
        temp_start = 0

        # WARNING: segment candidates again, different record row
        for match_id, token in enumerate(pure_tokens):
            if stop_flag[match_id] is False:
                temp_candidate.append(token)
                temp_exact_candidate.append(sen_tokens[match_id])

            elif len(temp_candidate) > 0:
                # record start and end
                fuzz_candidates.append((temp_start, len(temp_candidate), temp_candidate))
                exact_candidates.append(temp_exact_candidate)

                temp_candidate = []
                temp_exact_candidate = []
                temp_start = match_id + 1
            else:
                temp_start = match_id + 1

        if len(temp_candidate) > 0:
            fuzz_candidates.append((temp_start, len(temp_candidate), temp_candidate))
            exact_candidates.append(temp_exact_candidate)

        if not strict_match:
            candidate_index = -1
            # for match candidates
            for start, length, candidate_fuzz_tokens in fuzz_candidates:
                candidate_index += 1
                # get best match phrase
                best_multi_matches = fuzzy_matching_tokens(candidate_fuzz_tokens,
                                                           exact_candidates[candidate_index],
                                                           self.fuzz_choices,
                                                           self.original_tokens,
                                                           self.stop_vocab)
                fuzz_partial_combinations = []
                if best_multi_matches is None:
                    continue

                for best_multi_match in best_multi_matches:
                    # every match list is a partial combination of tags
                    match_list = []
                    for matches in best_multi_match:
                        fuzz_start = start + matches["start"]
                        length = matches["len"]

                        # at least the word length more than 2
                        if len(matches["match"]) > 2:
                            # index of choice, using original token as standard
                            match_id = self.original_tokens.index(matches["match"]) + 1
                            match_type = self.matcher_mapping_type[match_id]
                            # just text for Value !
                            if match_type == Bind.ValueText:
                                col_indices = self.val_to_col[match_id - 1]
                                col_index = col_indices[0]
                                # got different types of value
                                value_type, entity_type = get_val_actual_type(col_index, self.header_types)
                                # is main type
                                append_data = {
                                    "origin": self.original_tokens[match_id - 1],
                                    "header": [self.original_tokens[col_index] for col_index in col_indices],
                                    "entity": entity_type.value,
                                    "type": value_type.value,
                                    "start": fuzz_start,
                                    "len": length
                                }
                            elif match_type == Bind.ColumnText:
                                col_type, _ = get_col_actual_type(match_id - 1, self.header_types)
                                append_data = {
                                    "origin": self.original_tokens[match_id - 1],
                                    "header": self.original_tokens[match_id - 1],
                                    "entity": self.header_types[match_id - 1].value,
                                    "type": col_type.value,
                                    "start": fuzz_start,
                                    "len": length
                                }
                            else:
                                append_data = {
                                    "origin": self.original_tokens[match_id - 1],
                                    "header": None,
                                    "entity": None,
                                    "type": match_type.value,
                                    "start": fuzz_start,
                                    "len": length
                                }

                            match_list.append(append_data)

                    if len(match_list) >= 1:
                        fuzz_partial_combinations.append(match_list)
                    exact_combinations.extend(fuzz_partial_combinations)

        if len(exact_combinations) == 1:
            all_combinations = exact_combinations
        else:
            all_combinations = [list(match) for match in (itertools.product(*exact_combinations))]

        for index, sentence_match in enumerate(all_combinations):
            sorted_matches = sorted(sentence_match, key=lambda k: (k["start"], -k["len"]))

            #  merge continuous tags
            start = 0
            limit = len(sentence_match)
            while start < limit:
                current = sorted_matches[start]
                if start + 1 < limit and \
                        current["origin"] == sorted_matches[start + 1]["origin"] and \
                        current["type"] == sorted_matches[start + 1]["type"] and \
                        len(" ".join(
                            tokens[current["start"] + current["len"]:sorted_matches[start + 1]["start"]])) <= 3:
                    # char distance less than 3
                    sorted_matches[start + 1]["len"] += sorted_matches[start + 1]["start"] - current["start"]
                    sorted_matches[start + 1]["start"] = current["start"]
                    sorted_matches[start] = None
                start += 1

            sentence_match = sorted_matches

            while None in sentence_match:
                sentence_match.remove(None)

            all_combinations[index] = sentence_match

        data["matches"] = all_combinations
        return data


def load_tables(table_file) -> Dict:
    """
    loading table objects from file, the file is formatted as json-line format, where each line is a json object.
    :param table_file: default is in `data/tables.jsonl`.
    :return:
    """
    # table index starting from 1
    table_list = {ind + 1: json.loads(line) for ind, line in
                  enumerate(open(table_file, "r", encoding="utf8").readlines())}
    return table_list


def construct_table_entities(table_obj):
    """
    recognize the entity type for each column in one table. Here we use the related values to get the entity type itself.
    The entity type is useful especially for the scenario where the pronoun refers to one person or time.
    :param table_obj: object corresponding to each table
    :return:
    """
    # fetch parameters of table
    table_rows = table_obj["rows"]
    header_types = [SubType.dimension] * len(table_rows[0])
    # header_len x row_len
    types = [[SubType.dimension for _ in range(len(table_rows))] for _ in range(len(table_rows[0]))]
    for row_index, row in enumerate(table_rows):
        # skip the large row
        if row_index >= 1000:
            break
        for header_index, cell in enumerate(row):
            value = str(cell)
            doc = nlp(value)
            if len(doc.ents) == 1 and doc.ents[0].start_char == 0 and \
                    (len(value) * 3 / 4) <= doc.ents[0].end_char <= len(value):
                types[header_index][row_index] = SubType(doc.ents[0].label_)
            else:
                if re.fullmatch("^[12]\d{3}$", value) is not None:
                    types[header_index][row_index] = SubType.date
                elif re.fullmatch("^[-–]?[0-9]([0-9,\.])*$", value) is not None:
                    types[header_index][row_index] = SubType.cardinal
    for header_index in range(len(header_types)):
        header_types[header_index] = Counter(types[header_index]).most_common(1)[0][0]

    table_obj["types"] = [e for e in header_types]
    return table_obj


class Binder(object):
    """
    Binder is designed to do rule-based entity linking based on simple string matching.
    """
    def __init__(self, table_file):
        self.tables = load_tables(table_file)
        self.cache = {}

    def interactive_binding(self, db_id: int, query: str) -> StandardSpan:
        """
        Given the databsed/table id and its corresponding question, return the corresponding StandardSpan data structure.
        :param db_id: range from 1 to 120, staring from 1.
        :param query: question on the table.
        """
        if db_id not in self.cache:
            table_obj = self.tables[db_id]
            # construct entity for table_obj
            table_obj = construct_table_entities(table_obj)
            # execute simple rule-based binding
            table_title = table_obj["page_title"] if "page_title" in table_obj else ""
            table_title += " " + table_obj["section_title"] if "section_title" in table_obj else ""

            executor = BindingExecutor(table_obj["header"], table_obj["types"], table_obj["rows"], table_title)
            self.cache[db_id] = executor
        executor = self.cache[db_id]
        data = executor.binding_sequence(query)
        span_tag = transfer_to_tags(data)
        return span_tag
