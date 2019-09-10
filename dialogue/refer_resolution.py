// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

from copy import deepcopy
from enum import Enum

import numpy as np

from dialogue.constant import StandardSymbol, Bind


class FusionMode(Enum):
    IgnoreSelf = 0
    TakePrev = 1
    TakeSelf = 2
    TakePrevPlusSelf = 3


def all_subclasses(cls):
    return cls.__subclasses__() + [g for s in cls.__subclasses__()
                                   for g in all_subclasses(s)]


class PlainSegment(object):
    match_patterns = []
    # filling in logic trie
    actual_patterns = []
    standard_pattern = []

    def __init__(self, symbol_seq, token_seq):
        # symbols belonging to this segment
        self.symbols = symbol_seq
        # tokens belonging to this segment
        self.tokens = token_seq
        # no special significances
        self.stand_symbols = []
        self.has_tag = len([ele for ele in symbol_seq if ele is not None]) > 0

    @staticmethod
    def construct_segment(symbol_seq, token_seq, is_training):
        # try to identify a span into specific segment type
        tag_pattern = [symbol.class_type for symbol in symbol_seq if symbol is not None]

        # when in eval mode, greedy search
        if (is_training and (len(tag_pattern) >= 1 and tag_pattern[0] is not None and tag_pattern[-1] is not None))\
                or (not is_training):
            for cls in all_subclasses(PronounSegment):
                # traverse all match patterns
                for cls_rule in cls.match_patterns:
                    if tag_pattern == cls_rule:
                        return cls(symbol_seq, token_seq)
        # tag as col or val, or plain
        if set(tag_pattern).issubset({Bind.ValueText, Bind.ValueNumber, Bind.ValueDate}):
            return ValSegment(symbol_seq, token_seq)
        # elif Bind.ValueText in tag_pattern or \
        #         Bind.ValueNumber in tag_pattern or \
        #         Bind.ValueDate in tag_pattern:
        #     # TODO: how to snippet col & val together
        #     pass
        elif Bind.ColumnText in tag_pattern or \
                Bind.ColumnNumber in tag_pattern or \
                Bind.ColumnDate in tag_pattern:
            return ColSegment(symbol_seq, token_seq)
        else:
            return PlainSegment(symbol_seq, token_seq)

    @staticmethod
    def union_other_segments(other_segs):
        # construct new symbols and tokens
        symbols = []
        tokens = []

        for ind, segment in enumerate(other_segs):
            # Add symbol
            symbols.extend(segment.symbols)

            # Add tokens
            tokens.extend(segment.tokens)

            # if with context, context will be the conjunction
            if ind != len(other_segs) - 1:
                # Conjunction
                tokens.append("and")
        return symbols, tokens

    def direct_replace(self, other_segs, is_compare):
        """
        Merge other segments and append to this
        """
        symbols, tokens = self.union_other_segments(other_segs)

        if is_compare:
            self.symbols += symbols
            self.tokens += ["and"] + tokens
        else:
            self.symbols = symbols
            self.tokens = tokens

    # use information from other terms to update logic information
    def apply_changes(self, other_segments, is_compare):
        self.direct_replace(other_segments, is_compare)

    # clear symbols and tokens of self
    def clear(self):
        self.symbols = []
        self.tokens = []

    def organize_stand_symbols(self):
        """
        Transform the match symbols into standard symbol as the order of standard pattern
        """
        # clear stand symbols
        stand_symbols = [None] * len(self.standard_pattern)
        # record whether used
        stand_slot_record = [False] * len(self.standard_pattern)
        symbol_slot_record = [False] * len(self.symbols)
        # fill into slot
        for symbol_index, symbol in enumerate(self.symbols):
            for stand_index, expected_type in enumerate(self.standard_pattern):
                # ignore
                if symbol is None or stand_slot_record[stand_index] is True \
                        or symbol_slot_record[symbol_index] is True:
                    continue
                if (expected_type == Bind.Column and
                    (symbol.class_type == Bind.ColumnNumber or
                     symbol.class_type == Bind.ColumnText or
                     symbol.class_type == Bind.ColumnDate)) or \
                        (expected_type == Bind.Value and
                         (symbol.class_type == Bind.ValueText or
                          symbol.class_type == Bind.ValueNumber or
                          symbol.class_type == Bind.ValueDate)) or \
                        (expected_type == Bind.Deter and
                         (symbol.class_type == Bind.PreDeter or
                          symbol.class_type == Bind.NextDeter or
                          symbol.class_type == Bind.ThatDeter or
                          symbol.class_type == Bind.OtherDeter or
                          symbol.class_type == Bind.AllDeter)) or \
                        (expected_type == symbol.class_type):
                    symbol_slot_record[symbol_index] = True
                    stand_slot_record[stand_index] = True
                    stand_symbols[stand_index] = deepcopy(symbol)
        return stand_symbols

    def export_symbols(self):
        """
        return all non-null symbols of self
        :return:
        """
        return [sym for sym in self.symbols if sym]

    def __repr__(self):
        return "Class:" + self.__class__.__name__ + " Tokens:" + " ".join(self.tokens)

    def __str__(self):
        return " ".join(self.tokens)


class ColSegment(PlainSegment):
    match_patterns = [
        [Bind.ColumnText],
        [Bind.ColumnNumber],
        [Bind.ColumnDate]
    ]
    standard_pattern = [Bind.Column]

    def __init__(self, symbol_seq, token_seq):
        # assign standard symbols, in the order of standard pattern
        super().__init__(symbol_seq, token_seq)
        self.stand_symbols = self.organize_stand_symbols()


class ValSegment(PlainSegment):
    match_patterns = [
        [Bind.ValueDate],
        [Bind.ValueText],
        [Bind.ValueNumber]
    ]
    standard_pattern = [Bind.Value, Bind.Column]

    def __init__(self, symbol_seq, token_seq):
        # assign standard symbols, in the order of standard pattern
        super().__init__(symbol_seq, token_seq)
        self.stand_symbols = self.organize_stand_symbols()
        # TODO: constraint on standard symbols
        if self.stand_symbols[0] is not None:
            # value header re-assign
            for sym in self.stand_symbols[2:]:
                if sym is None:
                    continue
                if self.stand_symbols[0].header in sym.header:
                    sym.header = [self.stand_symbols[0].header]
                else:
                    sym.header = []


class PronounSegment(PlainSegment):

    def coarse_conflict(self, other_segment):
        """
        Item segment could be conflicting with any one
        :return: None because item reference should be resolved by its surrounding context
        """
        return False

    def reference_resolution(self, other_segments, conflict_state, intent_is_compare):
        """
        resolve the reference using other segments.
        :param other_segments: the one which should replace the segment
        :param conflict_state: None or True.
        :param intent_is_compare: if intent is compare, do nothing with original segment
        """
        # abstract function, do nothing
        return


class ItemSegment(PronounSegment):
    match_patterns = [
        [Bind.ThatDeter]
    ]
    standard_pattern = [Bind.ThatDeter]

    def __init__(self, symbol_seq, token_seq):
        # assign standard symbols, in the order of standard pattern
        super().__init__(symbol_seq, token_seq)
        self.stand_symbols = self.organize_stand_symbols()

    def coarse_conflict(self, other_segment):
        """
        Item segment could be conflicting with any one
        :return: None because item reference should be resolved by its surrounding context
        """
        # if isinstance(other_segment, ColSegment):
        #     return None
        # else:
        #     return False
        # conflict means replacement
        return True

    def reference_resolution(self, other_segments, conflict_state, intent_is_compare):
        # conflict state must be None
        symbols, tokens = self.union_other_segments(other_segments)
        self.symbols = symbols
        self.tokens = tokens


class EntitySegment(PronounSegment):
    match_patterns = [
        [Bind.ItemEntity]
    ]
    standard_pattern = [Bind.ItemEntity]

    def __init__(self, symbol_seq, token_seq):
        # assign standard symbols, in the order of standard pattern
        super().__init__(symbol_seq, token_seq)
        self.stand_symbols = self.organize_stand_symbols()

    def coarse_conflict(self, other_segment):
        try:
            if isinstance(other_segment, ValSegment) and \
                    other_segment.stand_symbols[0].sub_type in self.stand_symbols[0].sub_type:
                return True
        except:
            pass
        try:
            # decision by global
            if isinstance(other_segment, ColSegment) and \
                    other_segment.stand_symbols[0].sub_type in self.stand_symbols[0].sub_type:
                return True
        except:
            pass
        # impossible conflict with others
        return False

    def reference_resolution(self, other_segments, conflict_state, intent_is_compare):
        # if is None, means
        if conflict_state is None:
            # remove aggregation from select segment
            union_segments = []
            for seg in other_segments:
                # simplify the reference one
                if isinstance(seg, ColSegment) and \
                        seg.stand_symbols[0].sub_type in self.stand_symbols[0].sub_type:
                    union_segments.append(seg)
                else:
                    union_segments.append(seg)
            symbols, tokens = self.union_other_segments(union_segments)
        else:
            symbols, tokens = self.union_other_segments(other_segments)

        self.symbols = symbols
        self.tokens = tokens


class SetDeterminerSegment(PronounSegment):
    match_patterns = [
        [Bind.ThatDeter, Bind.ColumnDate],
        [Bind.ThatDeter, Bind.ColumnNumber],
        [Bind.ThatDeter, Bind.ColumnText],

        [Bind.AllDeter, Bind.ColumnDate],
        [Bind.AllDeter, Bind.ColumnNumber],
        [Bind.AllDeter, Bind.ColumnText],

        [Bind.OtherDeter, Bind.ColumnDate],
        [Bind.OtherDeter, Bind.ColumnNumber],
        [Bind.OtherDeter, Bind.ColumnText]
    ]
    standard_pattern = [Bind.Deter, Bind.Column]

    def __init__(self, symbol_seq, token_seq):
        # assign standard symbols, in the order of standard pattern
        super().__init__(symbol_seq, token_seq)
        self.stand_symbols = self.organize_stand_symbols()

    def coarse_conflict(self, other_segment):
        try:
            if isinstance(other_segment, ValSegment) and \
                    self.stand_symbols[1].header in other_segment.stand_symbols[0].header:
                return True
        except:
            pass
        try:
            # decision by global
            if isinstance(other_segment, ColSegment) and \
                    self.stand_symbols[1].header == other_segment.stand_symbols[0].header:
                return None
        except:
            pass

        return False

    def reference_resolution(self, other_segments, conflict_state, intent_is_compare):
        if self.stand_symbols[0].class_type == Bind.ThatDeter:
            # if None, fetch all context
            # e.g: how many champions are from China / show these champions
            if conflict_state is None:
                # remove aggregation from select segment
                union_segments = []
                for seg in other_segments:
                    if self.coarse_conflict(seg) is not False:
                        union_segments.append(seg)
                    else:
                        union_segments.append(seg)
                symbols, tokens = self.union_other_segments(union_segments)
            # else only fetch the symbol content
            # e.g: how many champions are from China / show score of that country
            else:
                symbols, tokens = self.union_other_segments(other_segments)
            self.symbols = symbols
            self.tokens = tokens
        elif self.stand_symbols[0].class_type == Bind.OtherDeter:
            # show me players in front / show me other players
            if conflict_state is None:
                # head words
                select_segments = []
                # modifiers
                modify_segments = []
                for seg in other_segments:
                    if self.coarse_conflict(seg) is not False:
                        select_segments.append(seg)
                    else:
                        modify_segments.append(seg)
                # merge modifiers
                modify_symbols, modify_tokens = self.union_other_segments(modify_segments)
                # merge headers
                head_symbols, head_tokens = self.union_other_segments(select_segments)
                # append not keyword
                self.symbols = head_symbols + [StandardSymbol("exclude", None, Bind.Exclude)] + modify_symbols
                self.tokens = head_tokens + ["not"] + modify_tokens

            # show me sales in the season Australia  / how about other countries
            else:
                symbols, tokens = self.union_other_segments(other_segments)
                # insert countries excluding ...
                self.symbols = [self.stand_symbols[1],
                                StandardSymbol("excluding", None, Bind.Exclude)] + symbols
                self.tokens = [self.stand_symbols[1].origin, "not"] + tokens

            if not intent_is_compare:
                # others clear self
                for other in other_segments:
                    other.clear()

                # modify original sentence, append all information to the first ele
                other_segments[0].symbols = self.symbols
                other_segments[0].tokens = self.tokens
            # if intent is compare, append to original token sequence
            else:
                # append to the last ele of other segments
                other_segments[-1].symbols += self.symbols
                other_segments[-1].tokens += ["and"] + self.tokens

        elif self.stand_symbols[0].class_type == Bind.AllDeter:
            # show me players in front / show all players
            if not intent_is_compare:
                if conflict_state is None:
                    # clear all other segments
                    for other in other_segments:
                        other.clear()

                    # modify original sentence, append all information to the first ele
                    other_segments[0].symbols = self.symbols
                    other_segments[0].tokens = self.tokens

                # show me sales in Australia  / for all countries
                else:
                    # clear all other segments
                    for other in other_segments:
                        other.clear()
            # if intent is compare, append to original token sequence
            else:
                # append to the last ele of other segments
                other_segments[-1].symbols += self.symbols
                other_segments[-1].tokens += ["and"] + self.tokens


class CalcDeterminerSegment(PronounSegment):
    match_patterns = [
        [Bind.NextDeter, Bind.ColumnNumber],
        [Bind.NextDeter, Bind.ColumnDate],

        [Bind.PreDeter, Bind.ColumnNumber],
        [Bind.PreDeter, Bind.ColumnDate],
    ]
    standard_pattern = [Bind.Deter, Bind.Column]

    def __init__(self, symbol_seq, token_seq):
        # assign standard symbols, in the order of standard pattern
        super().__init__(symbol_seq, token_seq)
        self.stand_symbols = self.organize_stand_symbols()

    def coarse_conflict(self, other_segment):
        try:
            if isinstance(other_segment, ValSegment) and \
                    self.stand_symbols[1].header in other_segment.stand_symbols[0].header:
                return True
        except:
            pass

        return False

    def reference_resolution(self, other_segments, conflict_state, intent_is_compare):
        if len(other_segments) == 1:
            value_seg = other_segments[0]
            # simplify the usage
            symbols = value_seg.stand_symbols
            tokens = value_seg.tokens

            # find the actual value index in symbols/tokens
            val_ind = [ind for ind, sym in enumerate(symbols) if
                       sym is not None and
                       (sym.class_type == Bind.ValueNumber or
                       sym.class_type == Bind.ValueDate)][0]

            if self.stand_symbols[0].class_type == Bind.PreDeter:
                update_value = str(int(symbols[val_ind].origin) + 1)
            else:
                update_value = str(int(symbols[val_ind].origin) - 1)

            # modify origin sentence
            if not intent_is_compare:
                symbols[val_ind].origin = update_value
                tokens[val_ind] = update_value
            # append to origin sentence
            else:
                fetch_source = symbols[val_ind]
                symbols += [StandardSymbol(update_value,
                                           fetch_source.header,
                                           fetch_source.class_type)]
                tokens += ["and", update_value]

            # update self reference
            self.symbols = symbols
            self.tokens = tokens
        else:
            print("Unable Access for Reference Resolution of {0}".format(self.__class__))


class ItemPossessSegment(PronounSegment):
    match_patterns = [
        [Bind.ItemPossess]
    ]
    standard_pattern = [Bind.ItemPossess]

    def __init__(self, symbol_seq, token_seq):
        # assign standard symbols, in the order of standard pattern
        super().__init__(symbol_seq, token_seq)
        self.stand_symbols = self.organize_stand_symbols()

    def coarse_conflict(self, other_segment):
        if isinstance(other_segment, ValSegment):
            return True
        else:
            return True

    def reference_resolution(self, other_segments, conflict_state, intent_is_compare):
        if conflict_state is None:
            # how many players are in front / show their names
            symbols, tokens = self.union_other_segments(other_segments)
        else:
            # show me BMW sales / show its profit
            symbols, tokens = self.union_other_segments(other_segments)
        self.symbols = symbols
        self.tokens = tokens


class PersonPossessSegment(PronounSegment):
    match_patterns = [
        [Bind.PersonPossess]
    ]
    standard_pattern = [Bind.PersonPossess]

    def __init__(self, symbol_seq, token_seq):
        # assign standard symbols, in the order of standard pattern
        super().__init__(symbol_seq, token_seq)
        self.stand_symbols = self.organize_stand_symbols()

    def coarse_conflict(self, other_segment):
        try:
            if isinstance(other_segment, ValSegment) and \
                    other_segment.stand_symbols[0].sub_type in self.stand_symbols[0].sub_type:
                return True
        except:
            pass

        try:
            if isinstance(other_segment, ColSegment) and \
                    other_segment.stand_symbols[0].sub_type in self.stand_symbols[0].sub_type:
                return True
        except:
            pass

        return False

    def reference_resolution(self, other_segments, conflict_state, intent_is_compare):
        if conflict_state is None:
            # which player is in the front / show his names
            symbols, tokens = self.union_other_segments(other_segments)
        else:
            # How much has the player YaoMing earned / show his salary
            symbols, tokens = self.union_other_segments(other_segments)
        self.symbols = symbols
        self.tokens = tokens


class ConflictLinker(object):

    def __init__(self, logic_tokens, follow_tokens,
                 logic_tags, follow_tags,
                 logic_spans, follow_spans, training=True):

        logic_segment_seq = []
        for (prev_start, prev_end) in logic_spans:
            logic_segment_seq.append(PlainSegment.construct_segment(
                logic_tags[prev_start: prev_end],
                logic_tokens[prev_start: prev_end],
                training
            ))

        follow_segment_seq = []
        for (fol_start, fol_end) in follow_spans:
            follow_segment_seq.append(PlainSegment.construct_segment(
                follow_tags[fol_start: fol_end],
                follow_tokens[fol_start: fol_end],
                training
            ))

        # deepcopy
        self.dynamic_logic_segment_seq = deepcopy(logic_segment_seq)

        # static logic segment seq is just used for reference
        self.static_logic_segment_seq = deepcopy(logic_segment_seq)
        self.follow_segment_seq = deepcopy(follow_segment_seq)

        # record the actual conflict segment in logic
        self.logic_seg_used = np.zeros(len(self.dynamic_logic_segment_seq), dtype=np.int)
        self.follow_seg_used = np.zeros(len(self.follow_segment_seq), dtype=np.int)

    def conflict_resolution(self, conflict_map, intent):
        """
        Conflict matrix should be [follow_len, previous_len] matrix, with 1 means conflict, 0 means non-conflict.
        And then we use the coarse_conflict of pronoun segment to set new conflict matrix of True, False and None
        :param conflict_map:
        :param intent:
        :return:
        """
        semantic_complete = True if len([ele for ele in self.follow_segment_seq
                                         if isinstance(ele, ItemPossessSegment)
                                         or isinstance(ele, PersonPossessSegment)]) > 0 else False

        log_inform = ""
        log_inform += "Logic Segmentation:\n{0}\n\n".format(
            "\n".join([repr(seg) for seg in self.dynamic_logic_segment_seq]))
        log_inform += "Follow Segmentation:\n{0}\n\n".format("\n".join([repr(seg) for seg in self.follow_segment_seq]))
        log_inform += "Logic Query:{0}\n".format(" ".join([str(seg) for seg in self.dynamic_logic_segment_seq]))
        log_inform += "Follow Query:{0}\n\n".format(" ".join([str(seg) for seg in self.follow_segment_seq]))

        # collect all coarse conflict state, whether True, False or None
        conflict_matrix = [[TriState(False) for _ in range(len(self.dynamic_logic_segment_seq))]
                           for _ in range(len(self.follow_segment_seq))]
        # traverse the segment sequence
        for follow_ind, follow_segment in enumerate(self.follow_segment_seq):
            for logic_ind, logic_segment in enumerate(self.dynamic_logic_segment_seq):
                if isinstance(follow_segment, PronounSegment) and conflict_map[follow_ind][logic_ind] == 1:
                    conflict_matrix[follow_ind][logic_ind] = TriState(follow_segment.coarse_conflict(logic_segment))
                elif conflict_map[follow_ind][logic_ind] == 1:
                    conflict_matrix[follow_ind][logic_ind] = TriState(True)

        # 1.explicit conflict resolution
        # WARNING: if just using complement(No actual conflict, there should not be any compare
        self.explicit_conflict_resolve(conflict_matrix, intent)

        # 2.implicit conflict resolution
        # WARNING: if has no body pronoun, there is no use of intention comparsion
        has_body_pronoun, has_pronoun = self.implicit_conflict_resolve(conflict_matrix, intent)

        # 3. append extra conflict to logic tail
        self.extra_segments_to_tail()

        # 4. choose one sentence to export
        logic_fusion = self.export_sentence(self.dynamic_logic_segment_seq)
        follow_fusion = self.export_sentence(self.follow_segment_seq)

        logic_symbols = self.export_symbols(self.dynamic_logic_segment_seq)
        follow_symbols = self.export_symbols(self.follow_segment_seq)

        log_inform += "Logic Fusion:{0}\nFollow Fusion:{1}\n------------------".format(logic_fusion, follow_fusion)
        # print(log_inform)

        from_prev_to_fol = True if has_body_pronoun or semantic_complete else False

        return logic_symbols, logic_fusion, follow_symbols, follow_fusion, from_prev_to_fol

    @classmethod
    def export_sentence(cls, segment_seq):
        sentence = ""
        for segment in segment_seq:
            sentence += " ".join(segment.tokens) + " "
        return sentence

    @classmethod
    def export_symbols(cls, segment_seq):
        symbols = []
        for seg in segment_seq:
            # append symbols
            symbols += [sym for sym in seg.symbols if sym]
        return symbols

    def explicit_conflict_resolve(self, conflict_matrix, intent):
        """
        Repair logic information using follow-up segments
        :param conflict_matrix: coarse matrix of conflict mapping
        :param intent: replace or compare, two intents
        :return: used logic segment, for reference
        """
        # TODO: Not Correct
        intent_is_compare = False

        for follow_index, conflict_row in enumerate(conflict_matrix):
            # no conflict, no changes are done
            if True not in conflict_row:
                continue

            # find the most appropriate conflict
            stand_segment = self.follow_segment_seq[follow_index]

            # no consideration on pronoun conflict
            if isinstance(stand_segment, PronounSegment):
                continue

            for logic_index, conflict_state in enumerate(conflict_row):
                if conflict_state:
                    # update the information from logic segments
                    self.logic_seg_used[logic_index] = 1
                    self.dynamic_logic_segment_seq[logic_index].apply_changes(
                        [self.follow_segment_seq[follow_index]], intent_is_compare
                    )

            self.follow_seg_used[follow_index] = 1

    def implicit_conflict_resolve(self, conflict_matrix, intent):

        intent_is_compare = False

        # detect whether there is a body pronoun, which refers to the body of one sentence
        has_body_pronoun = False

        # detect whether there is a plain pronoun, for indicating whether the compare intention useful
        has_plain_pronoun = False

        for follow_index, conflict_row in enumerate(conflict_matrix):
            # find the most appropriate conflict
            stand_segment = self.follow_segment_seq[follow_index]

            # no consideration on pronoun conflict
            if not isinstance(stand_segment, PronounSegment):
                continue

            # None means all in global antecedent
            if True in conflict_row:
                # do operations in actual conflict
                conflict_arr = [logic_index for logic_index, conflict_flag in enumerate(conflict_row) if conflict_flag]
                # get other segments
                other_segments = [self.dynamic_logic_segment_seq[index] for index in conflict_arr]
                # assert not used
                stand_segment.reference_resolution(other_segments, True, intent_is_compare)
                has_plain_pronoun = True

                # TODO: special judgment
                if type(stand_segment) is ItemSegment:
                    has_body_pronoun = True

            elif None in conflict_row:
                # if compare, no modification on original, use static logic segments
                if intent_is_compare:
                    # use all antecedents
                    global_antecedent = [ind for ind, flag in enumerate(self.logic_seg_used)]
                    other_segments = [self.static_logic_segment_seq[index] for index in global_antecedent]
                # in the state of replace, we would avoid reference on used segment
                else:
                    # used filter would not added to global antecedents
                    global_antecedent = [ind for ind, flag in enumerate(self.logic_seg_used)
                                         if flag == 0 or
                                         isinstance(self.static_logic_segment_seq[ind], ColSegment)]
                    other_segments = [self.static_logic_segment_seq[index] for index in global_antecedent]
                # assert not used
                stand_segment.reference_resolution(other_segments, None, intent_is_compare)
                # inherit the select
                has_body_pronoun = True
            else:
                # if return False, means it is actually not a pronoun.
                # random select a error symbol to make model non confused.
                # error_segment = choice([ind for ind, flag in enumerate(self.logic_used) if flag == 1])
                # stand_segment.reference_resolution()

                # False means do nothing
                pass
        return has_body_pronoun, has_body_pronoun | has_plain_pronoun

    def extra_segments_to_tail(self):
        # extra segments with context
        unused_follow = [ind for ind, flag in enumerate(self.follow_seg_used)
                         if flag == 0 and not isinstance(self.follow_segment_seq[ind], PronounSegment)]
        # append it to self tail segment
        other_segments = [self.follow_segment_seq[ind] for ind in unused_follow]

        ind = 0
        # remove empty segments
        while ind < len(other_segments):
            segment = other_segments[ind]
            if not segment.has_tag:
                del other_segments[ind]
            else:
                break

        self.dynamic_logic_segment_seq += other_segments


class TriState(object):
    def __init__(self, value=None):
        if any(value is v for v in (True, False, None)):
            self.value = value
        else:
            raise ValueError("TriState value must be True, False, or None")

    def __eq__(self, other):
        return (self.value is other.value if isinstance(other, TriState)
        else self.value is other)

    def __ne__(self, other):
        return not self == other

    def __bool__(self):
        if self.value is True:
            return True
        else:
            return False

    def __str__(self):
        return str(self.value)

    def __repr__(self):
        return "TriState(%s)" % self.value
