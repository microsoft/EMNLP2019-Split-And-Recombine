from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField, SequenceLabelField, MetadataField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import WordTokenizer
from allennlp.data.tokenizers.word_splitter import JustSpacesWordSplitter
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer, TokenCharactersIndexer
from allennlp.common.checks import ConfigurationError
import sys
import traceback
from typing import Dict, List, Any
import logging
from overrides import overrides
import os
import dill
import re
import json

from dialogue.bind_define import StandardSpan
from dialogue.constant import COL_PREFIX, VAL_PREFIX, COLUMN_BIND_TYPES, VALUE_BIND_TYPES, Bind


logger = logging.getLogger(__name__)


def abstract_utterance(utterance_obj: StandardSpan, col_counter=None, val_counter=None):
    # default setting
    if val_counter is None:
        val_counter = {}
    if col_counter is None:
        col_counter = {}
    tags = utterance_obj.tags
    uttr = [tags[ind].origin if tags[ind] is not None else utterance_obj.utterance[ind]
            for ind, _ in enumerate(utterance_obj.utterance)]
    for ind, tag in enumerate(tags):
        # ind is index of tag in tags
        if tag is None:
            continue
        if tag.class_type in COLUMN_BIND_TYPES or tag.class_type == Bind.Table:
            if uttr[ind] not in col_counter:
                col_counter[uttr[ind]] = len(col_counter)
            col_id = col_counter[uttr[ind]]
            uttr[ind] = COL_PREFIX + str(col_id)
        elif tag.class_type in VALUE_BIND_TYPES:
            if uttr[ind] not in val_counter:
                val_counter[uttr[ind]] = len(val_counter)
            val_id = val_counter[uttr[ind]]
            uttr[ind] = VAL_PREFIX + str(val_id)
        elif " " in uttr[ind]:
            uttr[ind] = uttr[ind].replace(" ", "_")
        else:
            pass
    return uttr, col_counter, val_counter


@DatasetReader.register("followup")
class FollowUpDataReader(DatasetReader):
    def __init__(self, is_pretrain,
                 token_indexer: Dict[str, TokenIndexer] = None,
                 char_indexer: Dict[str, TokenCharactersIndexer] = None,
                 lazy: bool = False,
                 tables_file: str = 'data\\tables.jsonl',
                 test_sym_file: str = 'data\\test.sym',
                 load_cache: bool = True,
                 save_cache: bool = True,
                 cache_dir: str = 'cache',
                 loading_limit: int = -1):
        super().__init__(lazy=lazy)
        self._tokenizer = WordTokenizer(word_splitter=JustSpacesWordSplitter())
        self._token_indexers = token_indexer or {"tokens": SingleIdTokenIndexer()}
        self._char_indexers = char_indexer
        self._is_pretrain = is_pretrain

        self._table_file = tables_file
        self._loading_limit = loading_limit

        self._load_cache = load_cache
        self._save_cache = save_cache
        self._cache_dir = cache_dir

        if self._load_cache or self._save_cache:
            if not os.path.exists(self._cache_dir):
                os.mkdir(self._cache_dir)

        self._test_sym_file = test_sym_file

    @overrides
    def _read(self, file_path: str):
        if not file_path.endswith(".jsonl"):
            raise ConfigurationError(f"The file path is not from FollowUp dataset {file_path}")

        train_mode = os.path.split(file_path)[-1]
        cache_dir = os.path.join(self._cache_dir, train_mode)

        if self._load_cache or self._save_cache:
            if not os.path.exists(cache_dir):
                os.mkdir(cache_dir)

        with open(file_path, "r", encoding="utf8", errors="ignore") as data_file:
            for total_cnt, json_line in enumerate(data_file):
                # loading some examples
                if self._loading_limit == total_cnt:
                    break

                cache_file = os.path.join(cache_dir, f'ins-{total_cnt}.bin')

                if self._load_cache and os.path.exists(cache_file):  # (not A) or B
                    ins = dill.load(open(cache_file, 'rb'))
                    # None passing.
                    if ins is not None:
                        # read pre-training file
                        if not self._is_pretrain and 'train' in train_mode:
                            # remove the labels
                            ins.fields.pop('prev_labels')
                            ins.fields.pop('fol_labels')
                            ins.fields.pop('conflicts')
                            yield ins
                        else:
                            yield ins
                    continue

                # just use 1st and 2nd as independent/dependent query
                # construct based on inter_ex
                try:
                    snip_obj = json.loads(json_line)

                    # recover to StandardSpan
                    snip_obj['prev'] = StandardSpan.from_json_dict(snip_obj['prev'])
                    snip_obj['follow'] = StandardSpan.from_json_dict(snip_obj['follow'])
                    snip_obj['restate'] = StandardSpan.from_json_dict(snip_obj['restate'])

                    ins = self.text_to_instance(
                        origin_obj=snip_obj,
                    )

                    if self._save_cache and ins is not None:
                        # save cache into file
                        cache_file = os.path.join(cache_dir, f'ins-{total_cnt}.bin')
                        dill.dump(ins, open(cache_file, 'wb'))

                    if ins is not None:
                        yield ins
                except Exception as e:
                    print(f'Error in line: {total_cnt}')
                    exec_info = sys.exc_info()
                    traceback.print_exception(*exec_info)

    def text_to_instance(self, origin_obj: Any) -> Instance:

        prev_obj = origin_obj['prev']
        fol_obj = origin_obj['follow']

        abs_prev_tokens, col_counter, val_counter = abstract_utterance(prev_obj)
        abs_fol_tokens, _, _ = abstract_utterance(fol_obj, col_counter, val_counter)

        # token level tokenizing
        prev_tokens = self._tokenizer.tokenize(" ".join(abs_prev_tokens))
        prev_tokens = TextField(prev_tokens, self._token_indexers)

        fol_tokens = self._tokenizer.tokenize(" ".join(abs_fol_tokens))
        fol_tokens = TextField(fol_tokens, self._token_indexers)

        # char level tokenizing
        prev_tag_tokens = []
        prev_anno: StandardSpan = origin_obj["prev"]
        for ind, tag in enumerate(prev_anno.tags):
            if tag is None:
                prev_tag_tokens.append(prev_tokens[ind].text)
            elif tag.class_type in COLUMN_BIND_TYPES:
                prev_tag_tokens.append(tag.header.replace(" ", "_"))
            elif tag.class_type in VALUE_BIND_TYPES:
                if len(tag.header) > 0:
                    prev_tag_tokens.append(tag.header[0].replace(" ", "_"))
                else:
                    prev_tag_tokens.append(tag.origin.replace(" ", "_"))
            else:
                prev_tag_tokens.append(prev_tokens[ind].text)

        fol_char_str = []
        fol_anno: StandardSpan = origin_obj["follow"]
        for ind, tag in enumerate(fol_anno.tags):
            if tag is None:
                fol_char_str.append(fol_tokens[ind].text)
            elif tag.class_type in COLUMN_BIND_TYPES:
                fol_char_str.append(tag.header.replace(" ", "_"))
            elif tag.class_type in VALUE_BIND_TYPES:
                if len(tag.header) > 0:
                    fol_char_str.append(tag.header[0].replace(" ", "_"))
                else:
                    fol_char_str.append(tag.origin.replace(" ", "_"))
            else:
                fol_char_str.append(fol_tokens[ind].text)

        # split into char-based tokens
        prev_tag_str = " ".join(prev_tag_tokens)
        prev_tag_tokens = self._tokenizer.tokenize(prev_tag_str)
        prev_tag_field = TextField(prev_tag_tokens, self._char_indexers)

        fol_tag_str = " ".join(fol_char_str)
        fol_char_str = self._tokenizer.tokenize(fol_tag_str)
        fol_tag_field = TextField(fol_char_str, self._char_indexers)

        fields = {'prev_tokens': prev_tokens,
                  'fol_tokens': fol_tokens,
                  'prev_tags': prev_tag_field,
                  'fol_tags': fol_tag_field}

        metadata = {"origin_obj": origin_obj,
                    "tokens_origin": abs_prev_tokens + abs_fol_tokens}
        metadata_field = MetadataField(metadata)
        fields['metadata'] = metadata_field

        # pre-training object caching
        prev_snippets = origin_obj['prev'].snippet
        fol_snippets = origin_obj['follow'].snippet
        conflict = origin_obj['conflicts']

        origin_obj.pop('conflicts')

        prev_labels = SequenceLabelField(prev_snippets, prev_tokens)
        fields['prev_labels'] = prev_labels
        fol_labels = SequenceLabelField(fol_snippets, fol_tokens)
        fields['fol_labels'] = fol_labels
        conflict_field = MetadataField(conflict)
        fields['conflicts'] = conflict_field

        fields['metadata'].metadata['origin_obj']['prev_labels'] = prev_snippets
        fields['metadata'].metadata['origin_obj']['fol_labels'] = fol_snippets

        return Instance(fields)
