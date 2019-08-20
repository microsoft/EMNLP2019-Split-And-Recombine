from allennlp.data.token_indexers import TokenCharactersIndexer
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.tokenizers import CharacterTokenizer
from allennlp.data.vocabulary import Vocabulary
import numpy as np
import os

embedding_dim = 100


def download_dataset():
    # download data
    try:
        import requests
    except ImportError:
        print("Please install requests module to download dataset. You could achieve it via `pip install requests`.")
        exit(-1)

    print('Start to download FollowUp dataset from https://github.com/SivilTaram/FollowUp')

    url = 'https://github.com/SivilTaram/FollowUp/archive/master.zip'
    r = requests.get(url)

    store_zip_file = 'FollowUp-master.zip'
    store_zip_folder = 'FollowUp-master'

    with open(store_zip_file, 'wb') as f:
        f.write(r.content)

    from zipfile import ZipFile
    import shutil

    with ZipFile(store_zip_file, 'r') as zip_obj:
        zip_obj.extractall(store_zip_folder)

    try:
        shutil.move('{0}\\{0}\\data'.format(store_zip_folder), 'data')
        shutil.move('{0}\\{0}\\data_processed'.format(store_zip_folder), 'data_processed')
    except Exception:
        print("Fail to move data from `FollowUp-master\\FollowUp-master`. Please do it by yourself.")
    else:
        print("Successfully unzip the FollowUp dataset and process it into `data_processed` and `data`.")
    finally:
        os.remove(store_zip_file)
        shutil.rmtree(store_zip_folder)


def download_glove_embedding():
    # download data
    try:
        import requests
    except ImportError:
        print("Please install requests module at first. You could achieve it via `pip install requests`.")
        exit(-1)

    print('Start to download Glove.twitter.27B from http://nlp.stanford.edu/data/glove.twitter.27B.zip')

    url = 'http://nlp.stanford.edu/data/glove.twitter.27B.zip'

    store_zip_file = 'glove.zip'
    store_zip_folder = 'glove'

    try:
        import tqdm
    except ImportError:
        print("No `tqdm` module found, no progress bar showing.")

        r = requests.get(url)
        with open(store_zip_file, 'wb') as f:
            f.write(r.content)
    else:
        r = requests.get(url, stream=True)
        file_size = int(r.headers['Content-Length'])
        chunk_size = 1024
        num_bars = int(file_size / chunk_size)

        with open(store_zip_file, 'wb') as fp:
            for chunk in tqdm.tqdm(
                    r.iter_content(chunk_size=chunk_size)
                    , total=num_bars
                    , unit='KB'
                    , desc=store_zip_file
                    , leave=True  # progressbar stays
            ):
                fp.write(chunk)

    print('Start to unzip glove.zip to get the glove pre-training embedding.')
    # unzip file
    try:
        from zipfile import ZipFile
        with ZipFile(store_zip_file, 'r') as zip_obj:
            zip_obj.extractall(store_zip_folder)
    except Exception:
        print("Fail to unzip! You should unzip `glove.zip` manually.")
    else:
        os.remove(store_zip_file)
        print("Successfully unzip `glove.zip` into `glove` folder.")


def construct_reader():
    from data_reader.dialogue_reader import FollowUpDataReader
    character_tokenizer = CharacterTokenizer(byte_encoding="utf-8",
                                             start_tokens=[259],
                                             end_tokens=[260])
    token_character_indexer = TokenCharactersIndexer(character_tokenizer=character_tokenizer,
                                                     min_padding_length=5)
    token_indexer = SingleIdTokenIndexer(lowercase_tokens=True)
    reader = FollowUpDataReader(token_indexer={
        # "elmo": elmo_indexer,
        "token_words": token_indexer
    }, char_indexer={
        "token_characters": token_character_indexer,
    }, is_pretrain=True)
    return reader


def build_vocab_embedding():
    reader = construct_reader()
    train_dataset = reader.read("data_processed\\train.jsonl")
    validation_dataset = reader.read("data_processed\\test.jsonl")

    # load vocabulary
    vocab = Vocabulary.from_instances(train_dataset + validation_dataset)
    vocab_namespace = vocab.get_token_to_index_vocabulary()

    col_vec = np.random.uniform(-1, 1, embedding_dim)
    val_vec = np.random.uniform(-1, 1, embedding_dim)

    # load pretrained glove embedding
    with open("glove\\glove.twitter.27B.{}d.txt".format(embedding_dim), "r", encoding="utf8") as f:
        glove_embedding = {}
        for line in f:
            word_vec = line.split(" ")
            word = word_vec[0]
            vec = [float(s) for s in word_vec[1:]]
            # if word in vocab
            glove_embedding[word] = vec
        vocab_embedding = []
        # traverse vocab
        for word in vocab_namespace:
            word_lower = word.lower()
            # check if "_" in word
            if "col#" in word_lower:
                word_map_vec = col_vec + np.random.normal(0, 0.5, embedding_dim)
            elif "val#" in word_lower:
                word_map_vec = val_vec + np.random.normal(0, 0.5, embedding_dim)
            elif "_" in word_lower:
                multi_words = word_lower.split("_")
                # add all word vec to one
                word_map_vec = np.asarray(
                    np.mean(
                        np.asmatrix(
                            [glove_embedding[m_word] for m_word in multi_words if m_word in glove_embedding]), axis=0)
                )[0]
            else:
                word_map_vec = glove_embedding[word] if word in glove_embedding else None
            if word_map_vec is not None:
                concat_line = " ".join([word_lower] + [str(num) for num in word_map_vec])
                vocab_embedding.append(concat_line)
        write_f = open("glove\\glove.vocab.{}d.txt".format(embedding_dim), "w", encoding="utf8")
        write_f.write("\n".join(vocab_embedding))


if __name__ == '__main__':
    # identify whether there is data folder, it not, download FollowUp dataset.
    if not os.path.isdir('data_processed'):
        download_dataset()
    if not os.path.isdir('glove'):
        download_glove_embedding()
    build_vocab_embedding()
