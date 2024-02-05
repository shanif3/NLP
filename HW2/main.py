from transformers import RobertaTokenizer, RobertaModel, pipeline
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
import torch
import nltk
import re


# nltk.download('punkt')
# pip install transformers
# pip install gensim
# pip install nltk


def warm_up():
    model_name = "roberta-base"
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    model = RobertaModel.from_pretrained(model_name)
    sentence = "I am so <mask>"
    input_ids = tokenizer.encode(sentence, return_tensors="pt")
    output = model(input_ids)

    hidden_states = output.last_hidden_state
    print(tokenizer.tokenize(sentence))
    am_token = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sentence)[1])  # Ġam
    mask_token = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sentence)[3])  # <mask>
    mask_token = tokenizer.mask_token_id  # or to use this

    print(f"input_ids: {input_ids}")
    list_tensor = input_ids[0].tolist()
    tokens = tokenizer.convert_ids_to_tokens(list_tensor)
    count = 0
    for i in tokens:
        print(f"token {i} : {input_ids[0][count]} ")
        count += 1

    am_index = [list_tensor.index(i) for i in input_ids[0] if i == am_token]
    mask_index = [list_tensor.index(i) for i in input_ids[0] if i == mask_token]
    print(f"index of am {am_index}, index of <mask> {mask_index}")

    am_vector = hidden_states[0, am_index, :]
    mask_vector = hidden_states[0, mask_index, :]

    print(f"vecotor for 'am': {am_vector}")
    print(f"vecotor for '<mask>': {mask_vector}")

    # Generate predictions for "am"
    # the default predictions is 5, top_k=5
    am_fill_pipeline = pipeline("fill-mask", model=model_name)
    am_predictions = am_fill_pipeline("I <mask> so")

    # Generate predictions for "<mask>"
    mask_fill_pipeline = pipeline("fill-mask", model=model_name)
    mask_predictions = mask_fill_pipeline("I am so <mask>")

    # Display top-5 predictions and their probabilities for "am"
    print("Top-5 predictions for 'am':")
    for prediction in am_predictions:
        print(f"{prediction['token_str']}: {prediction['score']}")

    # Display top-5 predictions and their probabilities for "<mask>"
    print("\nTop-5 predictions for '<mask>':")
    for prediction in mask_predictions:
        print(f"{prediction['token_str']}: {prediction['score']}")

    # Find two sentences that share the same word, such that the cosine similarity between the word vectors in the two sentences is very high.

    model_name = "roberta-base"
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    model = RobertaModel.from_pretrained(model_name)

    sentence1 = "The cat gracefully jumps over the wall of water, but sadly fell. he will be okay."
    sentence2 = "The wall of the library is lined with shelves full of a lot of books."

    input_ids1 = tokenizer.encode(sentence1, return_tensors="pt")
    input_ids2 = tokenizer.encode(sentence2, return_tensors="pt")

    output1 = model(input_ids1)
    output2 = model(input_ids2)

    hidden_states1 = output1.last_hidden_state
    hidden_states2 = output2.last_hidden_state

    print(tokenizer.tokenize(sentence1))
    am_token = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sentence1)[5])  # Ġwall
    two_token = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sentence2)[1])  # Ġwall

    print(am_token)
    # inputs_ids.tolist()
    list_tensor1 = input_ids1[0].tolist()
    list_tensor2 = input_ids2[0].tolist()

    am_index = [list_tensor1.index(i) for i in input_ids1[0] if i == am_token]
    two_index = [list_tensor2.index(i) for i in input_ids2[0] if i == two_token]

    print(f"index of am {am_index}, index of <mask> {two_index}")

    am_vector = hidden_states1[0, am_index, :].detach().numpy()
    two_vector = hidden_states2[0, two_index, :].detach().numpy()

    am_vector = am_vector.reshape(1, -1)
    two_vector = two_vector.reshape(1, -1)

    print(cosine_similarity(am_vector, two_vector))

    # Find two sentences that share the same word, such that the cosine similarity between the word vectors in the two sentences is very high.

    # to put here

    # Find two sentences that share the same word, such that the cosine similarity between the word vectors in the two sentences is very low.

    sentence1 = "The cat gracefully jumps over the wall of water, but sadly fell. he will be okay."
    sentence2 = "The wall of the library is lined with shelves full of a lot of books."

    input_ids1 = tokenizer.encode(sentence1, return_tensors="pt")
    input_ids2 = tokenizer.encode(sentence2, return_tensors="pt")

    output1 = model(input_ids1)
    output2 = model(input_ids2)

    hidden_states1 = output1.last_hidden_state
    hidden_states2 = output2.last_hidden_state

    print(tokenizer.tokenize(sentence1))
    am_token = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sentence1)[5])  # Ġwall
    two_token = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sentence2)[1])  # Ġwall

    print(am_token)
    # inputs_ids.tolist()
    list_tensor1 = input_ids1[0].tolist()
    list_tensor2 = input_ids2[0].tolist()

    am_index = [list_tensor1.index(i) for i in input_ids1[0] if i == am_token]
    two_index = [list_tensor2.index(i) for i in input_ids2[0] if i == two_token]

    print(f"index of am {am_index}, index of <mask> {two_index}")

    am_vector = hidden_states1[0, am_index, :].detach().numpy()
    two_vector = hidden_states2[0, two_index, :].detach().numpy()

    am_vector = am_vector.reshape(1, -1)
    two_vector = two_vector.reshape(1, -1)

    print(cosine_similarity(am_vector, two_vector))

    # Find a sentence with n words, that is tokenized into m > n tokens by the tokenizer.
    sentence_n_word = "I'm so happy"
    tokens = tokenizer.tokenize(sentence_n_word)
    print(
        f" for the sentence 'I'm so happy' that has {len(sentence_n_word.split())} words has been tokinze into {len(tokens)} words")

def word_pos_train():
    with open("ass1-tagger-train", 'r') as file:
        lines = file.readlines()
        word_pos_list = [word for line in lines for word in line.split()]

    word_types_freq = {}
    pos_freq = {}
    for word_pos in word_pos_list:
        last_slash = word_pos.rfind("/")
        word = word_pos[:last_slash]
        pos = word_pos[last_slash + 1:]
        word = word.lower()
        pos_freq[pos] = pos_freq.get(pos, 0) + 1

        if word in word_types_freq:
            if pos in word_types_freq[word]:
                word_types_freq[word][pos] += 1
            else:
                word_types_freq[word][pos] = 1
        else:
            word_types_freq[word] = {pos: 1}

    most_frequent_types_in_train = {word: max(types_freq, key=types_freq.get) for word, types_freq in word_types_freq.items()}
    return most_frequent_types_in_train
def task_11():
    most_frequent_types = word_pos_train()

    # NO LABELS
    with open("ass1-tagger-dev-input", 'r') as file:
        lines = file.readlines()
        words = [word for line in lines for word in line.split()]

    tokens = [word.lower() for word in words]
    dict_stickers = {word: most_frequent_types.get(word, rules(word, tokens)) for word in tokens}
    # WITH LABELS
    with open("ass1-tagger-dev", 'r') as file:
        lines = file.readlines()
        word_pos_list = [word for line in lines for word in line.split()]

    dict_labels = {}
    for word_pos in word_pos_list:
        last_slash = word_pos.rfind("/")
        word = word_pos[:last_slash]
        pos = word_pos[last_slash + 1:]
        dict_labels[word.lower()] = pos

    accuracy(dict_stickers, dict_labels)

def accuracy(dict_stickers, dict_labels):
    true_positives = 0
    for word, pos in dict_stickers.items():
        if word.lower() in dict_labels and dict_labels.get(word.lower()) == dict_stickers.get(word.lower()):
            true_positives += 1
    return true_positives / len(dict_stickers)

def rules(word, tokens):
    if word[0].isdigit():
        return 'CD'
    if word.endswith('ly'):
        return 'RB'
    if word.endswith('ing'):
        return 'VBG'
    if word.endswith('ed'):
        return 'VBD'
    if word.endswith('s'):
        return 'NNS'

    if word.endswith('est'):
        return 'JJS'
    if word.endswith('er'):
        return 'JJR'

    if tokens[tokens.index(word) - 1] == 'the':
        return 'NNP'
    if tokens[tokens.index(word) - 1] in ['he', 'she', 'it', 'they']:
        return 'VBD'

    if word.endswith('al') or word.endswith('y') or word.endswith('ous') or word.endswith('ical') or word.endswith(
            'ic') or word.endswith('ous') or word.endswith('ful') or word.endswith('ive') or word.endswith('able'):
        return 'JJ'

    if word.endswith('ion') or word.endswith('ment') or word.endswith('ness') or word.endswith('ity') or word.endswith(
            'ty'):
        return 'NN'

    if word.endswith('es') or word.endswith('ies') or word.endswith('ves'):
        return 'NNS'

    else:
        return 'NNP'




def task12():
    acc = 0
    window_size = 0
    dict_stickers = {}
    best_acc=0
    best_window=0
    dict_labels=word_pos_train()
    trainset_without_labels("ass1-tagger-train", "ass1-tagger-train-no-labels")
    with open("ass1-tagger-dev-input", 'r') as file:
        lines = file.read()
    for window_size in range(1, 8, 2):
        print(f"checking for window size: {window_size}")
        # for each window size, train the model- the model is trained on each line in the file
        most_frequent_types = static_word_vectors(window_size)
        # for each window size, predict the most likely type of each word in the file based on the max frequency of the type
        dict_stickers = {word.lower(): most_frequent_types.get(word.lower(), not_in_train(word.lower(), most_frequent_types,window_size)) for i in
                  lines for word in i.split()}
        if accuracy(dict_stickers, dict_labels)> best_acc:
            best_acc = accuracy(dict_stickers, dict_labels)
            best_window = window_size
        # add loop for min acc over i.

    # Train Word2Vec model on all text - from file

    #check the accuracy of the model
    print(f"best accuracy: {best_acc} with window size of {best_window}")
    # save model to pickle?


# process train file to be without labels.
def extract_words_without_labels(line):
    word_pos_list = line.split()
    words = []
    for word_pos in word_pos_list:
        last_slash = word_pos.rfind("/")
        word = word_pos[:last_slash]
        type_of_word = word_pos[last_slash + 1:]
        word = word.lower()
        words.append(word)
    return words


def word2vec_for_line(word_types_freq, window_size, line):
    # Train Word2Vec model
    # Get the words from the line without the labels
    words_in_line = extract_words_without_labels(line)
    # Train the model on the words in the line
    model_word2vec = Word2Vec([words_in_line], vector_size=300, window=window_size, min_count=0)

    # Extract the word and its type from the line
    word_pos_list = line.split()
    for word_pos in word_pos_list:
        last_slash = word_pos.rfind("/")
        word = word_pos[:last_slash]
        type_of_word = word_pos[last_slash + 1:]
        word = word.lower()

        # Extract the vector for the words
        word_vector = model_word2vec.wv[word]
        word_vector_as_tuple = tuple(word_vector)

        if word_vector_as_tuple in word_types_freq:
            if type_of_word in word_types_freq[word_vector_as_tuple]:

                word_types_freq[word_vector_as_tuple][type_of_word] += 1
            else:
                word_types_freq[word_vector_as_tuple][type_of_word] = 1
        else:
            word_types_freq[word_vector_as_tuple] = {type_of_word: 1}
    return word_types_freq


def not_in_train(word, most_frequent_types, window_size):
    with open("ass1-tagger-train-no-labels", 'r') as train_file:
        train_words = [line.strip() for line in train_file]

    word2vec_model = Word2Vec(train_words, vector_size=300, window=window_size,
                              min_count=0, workers=4)

    if word in word2vec_model.wv.key_to_index:
        similar_words = word2vec_model.wv.most_similar(word, topn=5)
        for similar_tuple in similar_words:
            similar = similar_tuple[0]
            if similar.lower() in most_frequent_types:
                return most_frequent_types.get(similar.lower())

    return "NNP"


# shani - proccess train to file with no labels
def trainset_without_labels(input_filename, output_filename):
    with open(input_filename, 'r') as input_file:
        with open(output_filename, 'w') as output_file:
            for line in input_file:
                words_in_line = extract_words_without_labels(line)
                # Write the processed line to the output file
                for i in words_in_line:
                    output_file.write(i+' ')



def static_word_vectors(window_size):
    with open("ass1-tagger-train", 'r') as file:
        word_types_freq = {}
        for line in file:
            word_types_freq = word2vec_for_line(word_types_freq, window_size, line)
    # Determine the most frequent type for each word in the line
    most_frequent_types = {word_vector: max(types_freq, key=types_freq.get) for word_vector, types_freq in
                           word_types_freq.items()}
    return most_frequent_types


if __name__ == "__main__":
    # warm_up()
    task12()
