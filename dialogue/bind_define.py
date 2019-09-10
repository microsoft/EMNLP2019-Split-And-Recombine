// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

"""
Normalised format for data binding
"""
from typing import List, Optional, Dict, Any
from dialogue.constant import Bind, StandardSymbol


class StandardSpan(object):
    def __init__(self, utterance, tags, snippet):
        # abstract utterance
        self.utterance: List[str] = utterance
        self.tags: List[StandardSymbol] = tags
        self.snippet: List[str] = snippet

    @staticmethod
    def from_json_dict(obj_dict: Dict):
        tags = []
        for tag in obj_dict['tags']:
            if tag is not None:
                tags.append(StandardSymbol(tag["origin"],
                                           tag["header"],
                                           Bind(tag["class_type"])))
            else:
                tags.append(None)
        utterance = obj_dict['utterance']
        # the pre-trained snippet
        snippet = obj_dict['snippet'] if 'snippet' in obj_dict else None
        return StandardSpan(utterance, tags, snippet)
