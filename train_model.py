import os
import random
import torch
import torch.optim as optim
from allennlp.common.params import Params
from allennlp.data.iterators import BucketIterator
from allennlp.data.token_indexers import TokenCharactersIndexer
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.tokenizers import CharacterTokenizer
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding, TokenCharactersEncoder
from allennlp.modules.seq2vec_encoders import CnnEncoder
from allennlp.training.learning_rate_schedulers import LearningRateScheduler
from allennlp.training.trainer import Trainer
from data_reader.dialogue_reader import FollowUpDataReader
from model.follow_up import FollowUpSnippetModel
import json
import argparse

os.environ['CUDA_VISIBLE_DEVICES'] = '3'


def setup_arguments():
    parser = argparse.ArgumentParser(description='Training the FollowUpSnippet Model')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='learning rate for reinforcement learning or pretrain')
    parser.add_argument('--store_folder', choices=['pretrain', 'reinforce'], required=True,
                        help='Specify the checkpoint folder for model.')
    parser.add_argument('--serialization_dir', default='split_and_recombine_model',
                        help='The actual checkpoint folder which stores all training state/model state and metrics.')
    parser.add_argument('--seed', default=10, help='The seed for reproducing the experiments.')
    parser.add_argument('--rl_basic', default='',
                        help='pretrained serialization dir for reinforcement learning training')
    parser.add_argument('--margin', type=float, required=True,
                        help='margin hyper-parameter for margin loss.')
    parser.add_argument('--patience', type=int, default=10,
                        help='patience of validation.')
    parser.add_argument('--validation_metric', choices=['overall', 'symbol', 'bleu'], default='overall',
                        help='metric keeps the best model in validation.')
    parser.add_argument('--epoch', type=int, default=100,
                        help='maximum training epochs')

    parser_args = parser.parse_args()
    return parser_args


def setup_seed(model_args):
    seed = model_args.seed
    torch.manual_seed(seed)
    random.seed(seed)


def construct_reader(is_pretrain):
    character_tokenizer = CharacterTokenizer(byte_encoding="utf-8",
                                             start_tokens=[259],
                                             end_tokens=[260])
    token_character_indexer = TokenCharactersIndexer(character_tokenizer=character_tokenizer,
                                                     min_padding_length=5)
    token_indexer = SingleIdTokenIndexer(lowercase_tokens=True)
    reader = FollowUpDataReader(token_indexer={
        "token_words": token_indexer
    }, char_indexer={
        "token_characters": token_character_indexer,
    }, is_pretrain=is_pretrain)
    return reader


def construct_model(vocab, args):
    # token embedding

    word_embedding = Embedding.from_params(vocab=vocab, params=Params({
        "pretrained_file": "glove\\glove.vocab.100d.txt",
        "embedding_dim": 100,
        "trainable": True,
        "padding_index": 0
    }))

    word_embedding = BasicTextFieldEmbedder({
        "token_words": word_embedding
    })

    char_embedding = BasicTextFieldEmbedder({
        "token_characters": TokenCharactersEncoder(embedding=Embedding(embedding_dim=20,
                                                                       num_embeddings=262),
                                                   encoder=CnnEncoder(embedding_dim=20,
                                                                      ngram_filter_sizes=[5],
                                                                      num_filters=50)),
    })

    lstm = PytorchSeq2SeqWrapper(
        torch.nn.LSTM(input_size=100,
                      num_layers=1,
                      hidden_size=100,
                      bidirectional=True,
                      batch_first=True))

    model = FollowUpSnippetModel(vocab=vocab,
                                 word_embedder=word_embedding,
                                 char_embedder=char_embedding,
                                 tokens_encoder=lstm,
                                 model_args=args)

    return model


def construct_learning_scheduler(optimizer):
    # scheduler = ReduceLROnPlateau(optimizer, 'max', factor=0.5, patience=10, verbose=True)
    return LearningRateScheduler.from_params(optimizer, params=Params(
        {
            "type": "multi_step",
            "milestones": [45, 60, 75],
            "gamma": 0.5
        }
    ))


def train(model_args):
    model_name = model_args.serialization_dir
    checkpoint_dir = model_args.store_folder
    learning_rate = model_args.learning_rate
    rl_basic = model_args.rl_basic
    pretrain_folder = ''

    if checkpoint_dir == 'pretrain':
        is_pretrain = True
    else:
        # check if rl_basic is specified
        pretrain_folder = os.path.join('pretrain', rl_basic)
        if not os.path.exists(pretrain_folder):
            raise FileNotFoundError(f'Can not find the pretrained model {pretrain_folder}!')
        is_pretrain = False

    reader = construct_reader(is_pretrain=is_pretrain)

    train_dataset = reader.read("data_processed\\train.jsonl")
    test_dataset = reader.read("data_processed\\test.jsonl")

    # build vocabulary
    vocab = Vocabulary.from_instances(train_dataset + test_dataset)

    # build model and move it into cuda
    model = construct_model(vocab, model_args)
    model.cuda()

    # allocate
    optimizer = optim.Adam(model.parameters(), weight_decay=1e-5, lr=learning_rate)
    scheduler = construct_learning_scheduler(optimizer)

    iterator = BucketIterator(batch_size=2, sorting_keys=[("prev_tokens", "num_tokens")])
    iterator.index_with(vocab)

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # not recover from previous state, we should load the pretrain model as default.
    if not is_pretrain and not os.path.exists(os.path.join(checkpoint_dir, model_name, "best.th")):
        model_state = torch.load(os.path.join(pretrain_folder, "best.th"))
        model.load_state_dict(model_state)

    trainer = Trainer(model=model,
                      optimizer=optimizer,
                      iterator=iterator,
                      train_dataset=train_dataset,
                      validation_dataset=test_dataset,
                      learning_rate_scheduler=scheduler,
                      patience=model_args.patience,
                      validation_metric="+{}".format(model_args.validation_metric),
                      num_epochs=model_args.epoch,
                      serialization_dir=os.path.join(checkpoint_dir, model_name),
                      cuda_device=0,
                      should_log_learning_rate=True)

    trainer.train()
    return model_name


def test(model_args):
    reader = construct_reader(is_pretrain=False)
    train_dataset = reader.read("data_processed\\train.jsonl")
    test_dataset = reader.read("data_processed\\test.jsonl")

    # load vocabulary
    vocab = Vocabulary.from_instances(train_dataset + test_dataset)

    model = construct_model(vocab, model_args)

    # load state and evaluate
    model_name = model_args.serialization_dir
    checkpoint_dir = model_args.store_folder

    model_state = torch.load(os.path.join(checkpoint_dir, model_name, "best.th"))
    model.load_state_dict(model_state)
    model.eval()
    # move to GPU
    model.cuda()

    metric = model.evaluate_on_instances(test_dataset)

    # print metrics
    print(json.dumps(metric, indent=4))


if __name__ == '__main__':
    args = setup_arguments()
    # set seed
    setup_seed(args)
    # train model
    train(args)
    # test model
    test(args)
