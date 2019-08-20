from typing import Dict, Optional, List, Any
import numpy as np
import torch
import torch.nn.functional as F
from allennlp.data import Vocabulary
from dialogue.constant import DETER_BIND_TYPES
from allennlp.models.model import Model
from allennlp.modules import TextFieldEmbedder
from allennlp.modules.input_variational_dropout import InputVariationalDropout
from allennlp.modules.matrix_attention.cosine_matrix_attention import CosineMatrixAttention
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder
from allennlp.modules.token_embedders import Embedding
from allennlp.nn import RegularizerApplicator, InitializerApplicator, util
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from overrides import overrides
from torch.distributions.categorical import Categorical
from torch.nn.modules.distance import CosineSimilarity
from torch.nn.modules.loss import MarginRankingLoss
from multiprocessing.dummy import Pool as ThreadPool
from dialogue.refer_resolution import ConflictLinker
from model.metric import RewardScore, SymbolScore, BLEUScore
from model.util import RewardCalculator
from random import choice

"""
DEBUG flag, if in debug, the reinforce sampling will be executed within one thread.
"""
DEBUG = False
EPS = np.finfo(np.float32).eps.item()


def predict_span_start_end(prev_labels,
                           fol_labels):
    # previous phrase list & follow-up phrase list
    prev_start_end = []
    fol_start_end = []
    start = 0

    # cut into span start/end representation
    for pos_ind in range(len(prev_labels) + 1):
        # pos of SPLIT
        if pos_ind == len(prev_labels):
            if (start, pos_ind) not in prev_start_end:
                prev_start_end.append((start, pos_ind))
            break
        elif prev_labels[pos_ind] != 0 and pos_ind != start:
            prev_start_end.append((start, pos_ind))
            start = pos_ind

    start = 0
    # cut into span start/end representation
    for pos_ind in range(len(fol_labels) + 1):
        # pos of SPLIT
        if pos_ind == len(fol_labels):
            if (start, pos_ind) not in fol_start_end:
                fol_start_end.append((start, pos_ind))
            break
        elif fol_labels[pos_ind] != 0 and pos_ind != start:
            fol_start_end.append((start, pos_ind))
            start = pos_ind

    return prev_start_end, fol_start_end


class PolicyNet(torch.nn.Module):
    """
    Policy Network for reinforcement learning. Here we do not use dropout to NOT impose more uncertainty.
    """

    def __init__(self, hidden_size, output_size):
        super(PolicyNet, self).__init__()
        self.hidden2tag = torch.nn.Linear(hidden_size, output_size)
        self.saved_log_probs = []
        self.rewards = []
        self.saved_action_probs = []

    def forward(self, state):
        logistic = self.hidden2tag(state)
        return logistic

    def reset(self):
        self.saved_log_probs = []
        self.rewards = []
        self.saved_action_probs = []


@Model.register("snippet_model")
class FollowUpSnippetModel(Model):
    def __init__(self, vocab: Vocabulary,
                 char_embedder: TextFieldEmbedder,
                 word_embedder: TextFieldEmbedder,
                 tokens_encoder: Seq2SeqEncoder,
                 model_args,
                 inp_drop_rate: float = 0.5,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        """
        :param vocab: vocabulary from train and dev dataset
        :param char_embedder: character embedding + cnn encoder
        :param word_embedder: word embedding
        :param tokens_encoder: Bi-LSTM backbone for split
        :param model_args: model arguments
        :param inp_drop_rate: input dropout rate
        """
        super(FollowUpSnippetModel, self).__init__(vocab, regularizer)

        self.tokens_encoder = tokens_encoder

        self.projection_layer = torch.nn.Linear(
            in_features=word_embedder.get_output_dim() + 1 + char_embedder.get_output_dim(),
            out_features=self.tokens_encoder.get_input_dim(),
            bias=False)

        # integer to mark field, 0 or 1
        self.num_classes = 2
        self.num_conflicts = 2

        self._non_linear = torch.nn.PReLU()

        self.hidden_size = int(self.tokens_encoder.get_output_dim() / 2)

        self.policy_net = PolicyNet(self.tokens_encoder.get_output_dim() * 3,
                                    self.num_classes)

        self.token_field_embedding = word_embedder
        self.char_field_embedding = char_embedder

        self._scaled_value = 1.0
        self._self_attention = CosineMatrixAttention()

        self.margin_loss = MarginRankingLoss(margin=model_args.margin)

        # calculate span similarity
        self.cosine_similar = CosineSimilarity(dim=0)

        if inp_drop_rate > 0:
            self._variational_dropout = InputVariationalDropout(p=inp_drop_rate)
        else:
            self._variational_dropout = lambda x: x

        self.metrics = {
            "bleu": BLEUScore(),
            "reward": RewardScore(),
            "symbol": SymbolScore(),
            "reward_var": RewardScore(),
            "overall": RewardScore()
        }

        initializer(self)

    @overrides
    def forward(self, prev_tokens: Dict[str, torch.LongTensor],
                prev_tags: Dict[str, torch.LongTensor],
                fol_tokens: Dict[str, torch.LongTensor],
                fol_tags: Dict[str, torch.LongTensor],
                prev_labels: torch.Tensor = None,
                fol_labels: torch.Tensor = None,
                conflicts: List[Any] = None,
                metadata: List[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:

        prev_mask = get_text_field_mask(prev_tokens)
        # embedding sequence
        prev_embedding_seq = self.token_field_embedding(prev_tokens)
        # embedding tag
        prev_tag_embedding = self.char_field_embedding(prev_tags)

        fol_mask = get_text_field_mask(fol_tokens)
        # embedding sequence
        fol_embedding_seq = self.token_field_embedding(fol_tokens)
        # embedding tag
        fol_tag_embedding = self.char_field_embedding(fol_tags)

        batch_size, _ = prev_mask.size()

        # initialization in specific gpu devices
        gpu_device = prev_embedding_seq.device

        prev_phrase_tensor = torch.tensor([0.0], device=gpu_device)
        fol_phrase_tensor = torch.tensor([1.0], device=gpu_device)

        prev_phrase_embedding_seq = prev_phrase_tensor.repeat(
            prev_embedding_seq.size(0),
            prev_embedding_seq.size(1),
            1
        )

        fol_phrase_embedding_seq = fol_phrase_tensor.repeat(
            fol_embedding_seq.size(0),
            fol_embedding_seq.size(1),
            1
        )

        # concat embedding and phrase
        prev_embedding_seq = torch.cat([prev_embedding_seq, prev_phrase_embedding_seq, prev_tag_embedding],
                                       dim=2)
        fol_embedding_seq = torch.cat([fol_embedding_seq, fol_phrase_embedding_seq, fol_tag_embedding], dim=2)

        prev_embedding_seq = self.projection_layer(prev_embedding_seq)
        fol_embedding_seq = self.projection_layer(fol_embedding_seq)

        # embedding phrase label 0 means prev, 1 means follow-up
        if self.training:
            embedding = torch.cat([prev_embedding_seq, fol_embedding_seq], dim=1)
            embedding_var = self._variational_dropout(embedding)
            prev_mask_len = prev_mask.size(1)
            prev_embedding_seq_var = embedding_var[:, :prev_mask_len]
            fol_embedding_seq_var = embedding_var[:, prev_mask_len:]
        else:
            prev_embedding_seq_var = prev_embedding_seq
            fol_embedding_seq_var = fol_embedding_seq

        # encode sequence
        prev_encoder_out = self.tokens_encoder(prev_embedding_seq_var, prev_mask)
        fol_encoder_out = self.tokens_encoder(fol_embedding_seq_var, fol_mask)

        prev_forward_output = prev_encoder_out[:, :, :self.hidden_size]
        prev_backward_output = prev_encoder_out[:, :, self.hidden_size:]

        fol_forward_output = fol_encoder_out[:, :, :self.hidden_size]
        fol_backward_output = fol_encoder_out[:, :, self.hidden_size:]

        prev_attn_mask = prev_mask.view(batch_size, -1, 1) * fol_mask.view(batch_size, 1, -1)
        prev_forward_attn_matrix = self._self_attention(prev_forward_output, fol_forward_output) / self._scaled_value
        prev_backward_attn_matrix = self._self_attention(prev_backward_output, fol_backward_output) / self._scaled_value
        prev_mean_pooling_attn = util.masked_softmax(prev_forward_attn_matrix + prev_backward_attn_matrix,
                                                     prev_attn_mask)

        # take max pooling rather than average
        prev_attn_vec = torch.matmul(prev_mean_pooling_attn, fol_encoder_out)

        fol_attn_mask = fol_mask.view(batch_size, -1, 1) * prev_mask.view(batch_size, 1, -1)
        fol_forward_attn_matrix = self._self_attention(fol_forward_output, prev_forward_output) / self._scaled_value
        fol_backward_attn_matrix = self._self_attention(fol_backward_output, prev_backward_output) / self._scaled_value
        fol_mean_pooling_attn = util.masked_softmax(fol_forward_attn_matrix + fol_backward_attn_matrix, fol_attn_mask)

        # take max pooling rather than average
        fol_attn_vec = torch.matmul(fol_mean_pooling_attn, prev_encoder_out)

        # non_linear_output = self._non_linear(torch.cat([encoder_out, self_attention_vec], dim=2))
        # prev_linear = torch.cat([prev_encoder_out, prev_attn_vec], dim=2)
        # fol_linear = torch.cat([fol_encoder_out, fol_attn_vec], dim=2)
        prev_attn_multiply = prev_encoder_out * prev_attn_vec
        zero_tensor = torch.zeros((batch_size, 1, prev_attn_multiply.size(2)), device=gpu_device, dtype=torch.float)
        prev_attn_shift = torch.cat((zero_tensor,
                                     prev_attn_multiply[:, :-1, :]), dim=1)
        # shift attn vector to right, and then subtract them
        prev_linear = torch.cat([prev_encoder_out, prev_attn_multiply, prev_attn_shift], dim=2)

        fol_attn_multiply = fol_encoder_out * fol_attn_vec
        fol_attn_shift = torch.cat((zero_tensor,
                                    fol_attn_multiply[:, :-1, :]), dim=1)
        # shift attn vector to right, and then subtract them
        fol_linear = torch.cat([fol_encoder_out, fol_attn_multiply, fol_attn_shift], dim=2)

        prev_tag_logistics = self.policy_net(prev_linear)
        fol_tag_logistics = self.policy_net(fol_linear)

        # project to space
        prev_tag_prob = F.softmax(prev_tag_logistics, dim=2)
        prev_predict_labels = torch.argmax(prev_tag_prob, dim=2)

        fol_tag_prob = F.softmax(fol_tag_logistics, dim=2)
        fol_predict_labels = torch.argmax(fol_tag_prob, dim=2)

        predict_restate_str_list = []
        predict_restate_tag_list = []
        max_bleu_list = []

        # debug information
        _debug_batch_conflict_map = {}

        # using predict labels to cut utterance into span and fetch representations of span
        for batch_ind in range(batch_size):
            _debug_batch_conflict_map[batch_ind] = []

            # batch reference object
            batch_origin_obj = metadata[batch_ind]["origin_obj"]

            prev_start_end, fol_start_end = predict_span_start_end(
                prev_predict_labels[batch_ind, :sum(prev_mask[batch_ind])],
                fol_predict_labels[batch_ind, :sum(fol_mask[batch_ind])])

            # Phase 2: Predict actual fusion str via span start/end and similar gate
            predict_restate_str, predict_restate_tag \
                = self.predict_restate(batch_origin_obj,
                                       fol_start_end,
                                       prev_start_end,
                                       prev_forward_output,
                                       prev_backward_output,
                                       fol_forward_output,
                                       fol_backward_output,
                                       batch_ind,
                                       gpu_device,
                                       _debug_batch_conflict_map)

            # add it to batch
            predict_restate_str_list.append(predict_restate_str)
            predict_restate_tag_list.append(predict_restate_tag)

        batch_golden_restate_str = [" ".join(single_metadata["origin_obj"]["restate"].utterance)
                                    for single_metadata in metadata]

        batch_golden_restate_tag = [single_metadata["origin_obj"]["restate"].tags
                                    for single_metadata in metadata]
        output = {
            "probs": prev_tag_prob,
            "prev_labels": prev_predict_labels,
            "fol_labels": fol_predict_labels,
            "restate": predict_restate_str_list,
            "max_bleu": max_bleu_list
        }

        avg_bleu = self.metrics["bleu"](predict_restate_str_list, batch_golden_restate_str)
        avg_symbol = self.metrics["symbol"](predict_restate_tag_list, batch_golden_restate_tag)

        # overall measure
        self.metrics["overall"]([0.4 * avg_bleu + 0.6 * avg_symbol] * batch_size)

        conflict_confidences = []

        # condition on training to
        if self.training:
            if prev_labels is not None:

                labels = torch.cat([prev_labels, fol_labels], dim=1)
                # Initialization pre-training with longest common string
                logistics = torch.cat([prev_tag_logistics, fol_tag_logistics], dim=1)
                mask = torch.cat([prev_mask, fol_mask], dim=1)
                loss_snippet = sequence_cross_entropy_with_logits(logistics, labels, mask,
                                                                  label_smoothing=0.2)

                # for pre-training, we regard them as optimal ground truth
                conflict_confidences = [1.0] * batch_size
            else:
                if DEBUG:
                    rl_sample_count = 1
                else:
                    rl_sample_count = 20

                batch_loss_snippet = []
                batch_sample_conflicts = []

                # Training Phase 2: train conflict model via margin loss
                for batch_ind in range(batch_size):

                    dynamic_conflicts = []
                    dynamic_confidence = []

                    # batch reference object
                    batch_origin_obj = metadata[batch_ind]["origin_obj"]

                    prev_mask_len = prev_mask[batch_ind].sum().view(1).data.cpu().numpy()[0]
                    fol_mask_len = fol_mask[batch_ind].sum().view(1).data.cpu().numpy()[0]

                    sample_data = []

                    for _ in range(rl_sample_count):
                        prev_multi = Categorical(logits=prev_tag_logistics[batch_ind])
                        fol_multi = Categorical(logits=fol_tag_logistics[batch_ind])

                        prev_label_tensor = prev_multi.sample()
                        prev_label_tensor.data[0].fill_(1)
                        prev_sample_label = prev_label_tensor.data.cpu().numpy().astype(int)[:prev_mask_len]

                        fol_label_tensor = fol_multi.sample()
                        fol_label_tensor.data[0].fill_(1)
                        fol_sample_label = fol_label_tensor.data.cpu().numpy().astype(int)[:fol_mask_len]

                        log_prob = torch.cat(
                            [prev_multi.log_prob(prev_label_tensor), fol_multi.log_prob(fol_label_tensor)],
                            dim=-1)

                        conflict_prob_mat = self.calculate_conflict_prob_matrix(prev_sample_label,
                                                                                fol_sample_label,
                                                                                batch_ind,
                                                                                prev_forward_output,
                                                                                prev_backward_output,
                                                                                fol_forward_output,
                                                                                fol_backward_output,
                                                                                gpu_device)
                        self.policy_net.saved_log_probs.append(log_prob)
                        sample_data.append((prev_sample_label, fol_sample_label, batch_origin_obj, conflict_prob_mat))

                    if DEBUG:
                        ret_data = [sample_action(row) for row in sample_data]
                    else:
                        # Parallel to speed up the sampling process
                        with ThreadPool(4) as p:
                            chunk_size = rl_sample_count // 4
                            ret_data = p.map(sample_action, sample_data, chunksize=chunk_size)

                    for conflict_confidence, reinforce_reward, conflict_pair in ret_data:
                        self.policy_net.rewards.append(reinforce_reward)
                        dynamic_conflicts.append(conflict_pair)
                        dynamic_confidence.append(conflict_confidence)

                    rewards = torch.tensor(self.policy_net.rewards, device=gpu_device).float()
                    self.metrics["reward"](self.policy_net.rewards)
                    rewards -= rewards.mean().detach()
                    self.metrics["reward_var"]([rewards.std().data.cpu().numpy()])

                    loss_snippet = []
                    # reward high, optimize it; reward low, reversal optimization
                    for log_prob, reward in zip(self.policy_net.saved_log_probs,
                                                rewards):
                        loss_snippet.append((- log_prob * reward).unsqueeze(0))

                    loss_snippet = torch.cat(loss_snippet).mean(dim=1).sum().view(1)
                    batch_loss_snippet.append(loss_snippet)

                    # random select one
                    best_conflict_id = choice(range(rl_sample_count))
                    # best_conflict_id = np.argmax(self.policy_net.rewards)
                    batch_sample_conflicts.append(dynamic_conflicts[best_conflict_id])
                    conflict_confidences.append(dynamic_confidence[best_conflict_id])

                    self.policy_net.reset()

                loss_snippet = torch.cat(batch_loss_snippet).mean()

                # according to confidence
                conflicts = []
                for conflict_batch_id in range(batch_size):
                    conflicts.append(batch_sample_conflicts[conflict_batch_id])

            # Training Phase 1: train snippet model
            total_loss = loss_snippet

            border = torch.tensor([0.0], device=gpu_device)
            pos_target = torch.tensor([1.0], device=gpu_device)
            neg_target = torch.tensor([-1.0], device=gpu_device)

            # Training Phase 2: train conflict model via margin loss

            loss_conflict = torch.tensor([0.0], device=gpu_device)[0]
            # random decision on which to use

            for batch_ind in range(0, batch_size):
                batch_conflict_list = conflicts[batch_ind]
                # use prediction results to conflict

                temp_loss_conflict = torch.tensor([0.0], device=gpu_device)[0]

                if batch_conflict_list and len(batch_conflict_list) > 0:
                    for conflict in batch_conflict_list:
                        (prev_start, prev_end), (fol_start, fol_end), conflict_mode = conflict

                        fol_span_repr = get_span_repr(fol_forward_output[batch_ind],
                                                      fol_backward_output[batch_ind],
                                                      fol_start, fol_end)

                        prev_span_repr = get_span_repr(prev_forward_output[batch_ind],
                                                       prev_backward_output[batch_ind],
                                                       prev_start, prev_end)

                        inter_prob = self.cosine_similar(fol_span_repr, prev_span_repr).view(1)
                        # actual conflict
                        if conflict_mode == 1:
                            temp_loss_conflict += self.margin_loss(inter_prob,
                                                                   border,
                                                                   pos_target)
                        else:
                            temp_loss_conflict += self.margin_loss(inter_prob,
                                                                   border,
                                                                   neg_target)

                    temp_confidence = conflict_confidences[batch_ind]
                    loss_conflict += temp_confidence * temp_loss_conflict / len(batch_conflict_list)

            loss_conflict = loss_conflict / batch_size

            # for larger margin
            total_loss += loss_conflict

            output["loss"] = total_loss

        return output

    def evaluate_on_instances(self, instances):
        # logging errors
        # traverse on instances
        self.get_metrics(reset=True)
        for _instance in instances:
            self.forward_on_instance(_instance)
        metrics = self.get_metrics()
        return metrics

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        """
        Get metrics of all
        """
        return {
            "bleu": self.metrics["bleu"].get_metric(reset),
            "reward": self.metrics["reward"].get_metric(reset),
            "symbol": self.metrics["symbol"].get_metric(reset),
            "reward_var": self.metrics["reward_var"].get_metric(reset),
            "overall": self.metrics["overall"].get_metric(reset)
        }

    def calculate_conflict_prob_matrix(self, prev_sample_label, fol_sample_label,
                                       batch_ind, prev_forward, prev_backward,
                                       fol_forward, fol_backward, gpu_device):
        """
        Probability of conflict mapping, details can be viewed in Section 2.3.2 of our paper.
        """
        if isinstance(prev_sample_label[0], tuple) or prev_sample_label[0] is None:
            prev_start_end = prev_sample_label
            fol_start_end = fol_sample_label
        else:
            prev_start_end, fol_start_end = predict_span_start_end(prev_sample_label,
                                                                   fol_sample_label)

        # gather all similar
        conflict_prob_mat = np.zeros((len(prev_start_end), len(fol_start_end)))

        for fol_ind, fol_start_end_tuple in enumerate(fol_start_end):
            # keep ind unchanged
            if fol_start_end_tuple is None:
                continue

            fol_start, fol_end = fol_start_end_tuple
            # start/end is relative position

            fol_span_repr = get_span_repr(fol_forward[batch_ind],
                                          fol_backward[batch_ind],
                                          fol_start, fol_end)

            for prev_ind, prev_start_end_tuple in enumerate(prev_start_end):
                if prev_start_end_tuple is None:
                    continue

                # keep index unchanged
                prev_start, prev_end = prev_start_end_tuple

                prev_span_repr = get_span_repr(prev_forward[batch_ind],
                                               prev_backward[batch_ind],
                                               prev_start, prev_end)

                # header existing in previous spans
                inter_prob = self.cosine_similar(fol_span_repr, prev_span_repr).view(1)
                conflict_prob_mat[prev_ind, fol_ind] = inter_prob.data.cpu() / 2 + 0.5

        return conflict_prob_mat

    def predict_restate(self, batch_origin_obj,
                        fol_start_end,
                        prev_start_end,
                        prev_forward,
                        prev_backward,
                        fol_forward,
                        fol_backward,
                        batch_ind,
                        gpu_device,
                        _debug_batch_conflict_map):
        """
        After predicting the span start/end in precedent and follow-up, generate the restated query and
        restated symbol sequences.
        """
        conflict_map = np.zeros((len(fol_start_end), len(prev_start_end)), dtype=np.int)

        # tag flag, used for judging whether there is any
        pre_tag_flag = [True if tag is not None and tag.class_type not in DETER_BIND_TYPES
                        else False for tag in batch_origin_obj["prev"].tags]
        fol_tag_flag = [True if tag is not None else False for tag in batch_origin_obj["follow"].tags]

        prev_valid_start_end = [(start, end) if True in pre_tag_flag[start: end] else None
                                for start, end in prev_start_end]

        fol_valid_start_end = [(start, end) if True in fol_tag_flag[start: end] else None
                               for start, end in fol_start_end]

        for fol_ind, fol_start_end_tuple in enumerate(fol_valid_start_end):
            # keep ind unchanged
            if fol_start_end_tuple is None:
                continue

            fol_start, fol_end = fol_start_end_tuple
            # start/end is relative position

            fol_span_repr = get_span_repr(fol_forward[batch_ind],
                                          fol_backward[batch_ind],
                                          fol_start, fol_end)
            # span str repr
            fol_span_str = " ".join(batch_origin_obj["follow"].utterance[fol_start:fol_end])

            # gather all similar
            similar_gather = torch.zeros(len(prev_start_end), device=gpu_device, dtype=torch.float)

            for prev_id, prev_start_end_tuple in enumerate(prev_valid_start_end):
                # keep index unchanged
                if prev_start_end_tuple is None:
                    continue

                prev_start, prev_end = prev_start_end_tuple

                prev_span_repr = get_span_repr(prev_forward[batch_ind],
                                               prev_backward[batch_ind],
                                               prev_start, prev_end)

                # prev span str
                prev_span_str = " ".join(batch_origin_obj["prev"].utterance[prev_start:prev_end])

                # header existing in previous spans
                inter_prob = self.cosine_similar(fol_span_repr, prev_span_repr).view(1) / 2 + 0.5

                similar_gather[prev_id] = inter_prob
                _debug_batch_conflict_map[batch_ind].append("\t".join([prev_span_str,
                                                                       fol_span_str,
                                                                       str(inter_prob.data.cpu().numpy())]))

            # take the max similarity, if max less than 0.5, then judge no conflict
            max_similar_ind = torch.argmax(similar_gather)

            if similar_gather[max_similar_ind] > 0.6:
                conflict_map[fol_ind, max_similar_ind] = 1

        # previous fusion & follow-up fusion, 0 means source from prev, 1 means from Follow
        pre_tags = batch_origin_obj["prev"].tags
        fol_tags = batch_origin_obj["follow"].tags

        # previous fusion & follow-up fusion, 0 means source from prev, 1 means from Follow
        linker = ConflictLinker(batch_origin_obj["prev"].utterance,
                                batch_origin_obj["follow"].utterance,
                                pre_tags,
                                fol_tags,
                                prev_start_end,
                                fol_start_end)
        logic_symbol_seq, logic_fusion, fol_symbol_seq, fol_fusion, from_prev_to_fol = \
            linker.conflict_resolution(conflict_map, "")

        ret_symbol_seq, ret_fusion = (fol_symbol_seq, fol_fusion) if from_prev_to_fol \
            else (logic_symbol_seq, logic_fusion)

        return ret_fusion, ret_symbol_seq


def get_span_repr(forward_encoder_out,
                  backward_encoder_out,
                  span_start, span_end):
    """
    Given a span start/end position, fetch the subtraction representation of the span from LSTM.
    """
    # span end is always larger than actual value
    span_end -= 1
    forward_span_repr = get_forward_span_repr(forward_encoder_out, span_start, span_end)
    backward_span_repr = get_backward_span_repr(backward_encoder_out, span_start, span_end)
    # cat two representations
    span_repr = torch.cat((forward_span_repr, backward_span_repr))
    return span_repr


def get_forward_span_repr(forward_encoder_out, span_start, span_end):
    """
    Get forward span representation
    """
    if span_end >= len(forward_encoder_out):
        span_end = len(forward_encoder_out) - 1

    if span_start > span_end:
        forward_span_repr = torch.from_numpy(np.random.normal(0, 0.5, forward_encoder_out.size(-1))).cuda(
            forward_encoder_out.device).float()
    elif span_start == 0:
        forward_span_repr = forward_encoder_out[span_end]
    else:
        forward_span_repr = forward_encoder_out[span_end] - forward_encoder_out[span_start - 1]
    return forward_span_repr


def get_backward_span_repr(backward_encoder_out, span_start, span_end):
    """
    Get backward span representation
    """
    if span_start > span_end:
        backward_span_repr = torch.from_numpy(np.random.normal(0, 0.5, backward_encoder_out.size(-1))).cuda(
            backward_encoder_out.device).float()
    elif span_end >= len(backward_encoder_out) - 1:
        backward_span_repr = backward_encoder_out[span_start]
    else:
        backward_span_repr = backward_encoder_out[span_start] - backward_encoder_out[span_end + 1]
    return backward_span_repr


def sample_action(row):
    """
    Prepared data, sampling an span start/end sequence and gets its corresponding reward.
    """
    prev_sample_label, fol_sample_label, batch_origin_obj, conflict_prob_mat = row

    prev_start_end, fol_start_end = predict_span_start_end(prev_sample_label,
                                                           fol_sample_label)

    pre_tag_flag = [True if tag is not None and tag.class_type not in DETER_BIND_TYPES
                    else False for tag in batch_origin_obj["prev"].tags]
    fol_tag_flag = [True if tag is not None else False for tag in batch_origin_obj["follow"].tags]

    prev_valid_start_end = [(start, end) if True in pre_tag_flag[start: end] else None
                            for start, end in prev_start_end]

    fol_valid_start_end = [(start, end) if True in fol_tag_flag[start: end] else None
                           for start, end in fol_start_end]

    # Extra info of MAX BLEU, which potentially evaluates performance of snippet model
    # Sample actions only in training mode
    calculator = RewardCalculator(batch_origin_obj["prev"].utterance,
                                  batch_origin_obj["follow"].utterance,
                                  batch_origin_obj["restate"].utterance,
                                  batch_origin_obj["prev"].tags,
                                  batch_origin_obj["follow"].tags,
                                  batch_origin_obj["restate"].tags,
                                  prev_start_end, fol_start_end,
                                  prev_valid_start_end, fol_valid_start_end,
                                  True)

    # Extra info of MAX BLEU, which potentially evaluates performance of snippet model
    conflict_confidence, reinforce_reward, best_conflict = calculator.combination_reward_feedback(conflict_prob_mat)

    return conflict_confidence, reinforce_reward, best_conflict
