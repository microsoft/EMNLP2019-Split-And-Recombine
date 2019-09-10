// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

from typing import List
from overrides import overrides
from allennlp.training.metrics.metric import Metric
import statistics
from nltk.translate.bleu_score import sentence_bleu
from string import punctuation
import nltk
from dialogue.bind_define import Bind, StandardSymbol


def tokenize_sentence(sentence):
    tokens = [token.lower() for token in sentence.split(" ")]
    return tokens


def evaluate_bleu_score(predict_restate: str, ground_restate: str):
    predict = [ele for ele in tokenize_sentence(predict_restate) if ele not in punctuation]
    target = [ele for ele in tokenize_sentence(ground_restate) if ele not in punctuation]

    bleu_score = sentence_bleu([target], predict,
                               smoothing_function=nltk.translate.bleu_score.SmoothingFunction().method2)
    return bleu_score


def evaluate_symbol_score(predict_restate_tags: List[StandardSymbol], ground_restate_tags: List[StandardSymbol]):
    # remove determiner tag
    ground_restate_tags = [ele for ele in ground_restate_tags if
                           ele and
                           ele.class_type != Bind.PreDeter and
                           ele.class_type != Bind.NextDeter and
                           ele.class_type != Bind.AllDeter and
                           ele.class_type != Bind.Table and
                           ele.class_type != Bind.ThatDeter and
                           ele.class_type != Bind.ItemEntity and
                           ele.class_type != Bind.Item and
                           ele.class_type != Bind.ItemPossess and
                           ele.class_type != Bind.PersonPossess]

    predict_restate_tags = [ele for ele in predict_restate_tags if
                            ele and
                            ele.class_type != Bind.PreDeter and
                            ele.class_type != Bind.NextDeter and
                            ele.class_type != Bind.AllDeter and
                            ele.class_type != Bind.Table and
                            ele.class_type != Bind.ThatDeter and
                            ele.class_type != Bind.ItemEntity and
                            ele.class_type != Bind.Item and
                            ele.class_type != Bind.ItemPossess and
                            ele.class_type != Bind.PersonPossess]

    # calculate string exact match, if no return others
    repair_str_list = set([ele.origin for ele in ground_restate_tags])
    predict_str_list = set([ele.origin for ele in predict_restate_tags])

    if len(repair_str_list) != 0 and len(predict_str_list) != 0 and \
            len(ground_restate_tags) == len(predict_restate_tags) and \
            len(repair_str_list) == len(predict_str_list) and \
            len(repair_str_list.intersection(predict_str_list)) == len(repair_str_list):
        return 1

    if len(predict_restate_tags) == 0:
        return 0

    predict_set = set(predict_restate_tags)
    repair_set = set(ground_restate_tags)

    if len(predict_set) == 0:
        return 0
    if predict_set == repair_set and len(predict_restate_tags) == len(ground_restate_tags):
        return 1
    else:
        return 0


@Metric.register("snippet_reward")
class RewardScore(Metric):
    """
    This :class:`Metric` takes the best span string computed by a model, along with the answer
    strings labeled in the data, and computed exact match and F1 score using the official SQuAD
    evaluation script.
    """

    def __init__(self) -> None:
        self._total_bleu = 0.0
        self._count = 0

    @overrides
    def __call__(self, sample_rewards: List[float]):
        """
        Record the reward trends
        :param sample_rewards: rewards from all sampling snippets
        :return:
        """
        for reward in sample_rewards:
            self._total_bleu += reward
            self._count += 1

    @overrides
    def get_metric(self, reset: bool = False) -> float:
        """
        Returns
        -------
        Average exact match and F1 score (in that order) as computed by the official SQuAD script
        over all inputs.
        """
        avg_bleu_score = self._total_bleu / self._count if self._count > 0 else 0.0
        if reset:
            self.reset()
        return avg_bleu_score

    @overrides
    def reset(self):
        self._total_bleu = 0.0
        self._count = 0

    def __str__(self):
        return f"BLEUScore={self._total_bleu}"


@Metric.register("snippet_symbol")
class SymbolScore(Metric):
    def __init__(self) -> None:
        self._total_correct = 0
        self._count = 0

    @overrides
    def __call__(self, predict_restate_tags, ground_restate_tags):
        """
        This method is responsible for calculating symbol score in train/test.
        :return:
        """
        all_sym_acc = []
        for predict_tag_seq, fusion_tag_seq in zip(predict_restate_tags, ground_restate_tags):
            is_correct = evaluate_symbol_score(predict_tag_seq, fusion_tag_seq)
            self._total_correct += is_correct
            all_sym_acc.append(is_correct)
            self._count += 1
        return statistics.mean(all_sym_acc)

    @overrides
    def get_metric(self, reset: bool = False) -> float:
        """
        Returns
        -------
        Average exact match and F1 score (in that order) as computed by the official SQuAD script
        over all inputs.
        """
        avg_ratio = self._total_correct / self._count if self._count > 0 else 0.0
        if reset:
            self.reset()
        return avg_ratio

    @overrides
    def reset(self):
        self._total_correct = 0.0
        self._count = 0

    def __str__(self):
        return f"SymbolScore={self._total_correct}"


@Metric.register("followup_bleu")
class BLEUScore(Metric):
    """
    This :class:`Metric` takes the best span string computed by a model, along with the answer
    strings labeled in the data, and computed exact match and F1 score using the official SQuAD
    evaluation script.
    """

    def __init__(self) -> None:
        self._total_bleu = 0.0
        self._count = 0

    @overrides
    def __call__(self, predictions: List[str], golden_labels: List[str]):
        """
        Parameters
        ----------
        predictions : fusion predict string
        golden_labels : golden labels
        """
        all_bleus = []
        for prediction, ground in zip(predictions, golden_labels):
            bleu_score = evaluate_bleu_score(prediction, ground)
            all_bleus.append(bleu_score)
            self._total_bleu += bleu_score
            self._count += 1
        return statistics.mean(all_bleus)

    @overrides
    def get_metric(self, reset: bool = False) -> float:
        """
        Returns
        -------
        Average exact match and F1 score (in that order) as computed by the official SQuAD script
        over all inputs.
        """
        avg_bleu_score = self._total_bleu / self._count if self._count > 0 else 0.0
        if reset:
            self.reset()
        return avg_bleu_score

    @overrides
    def reset(self):
        self._total_bleu = 0.0
        self._count = 0

    def __str__(self):
        return f"BLEUScore={self._total_bleu}"
