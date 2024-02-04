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

model_name = "roberta-base"
tokenizer = RobertaTokenizer.from_pretrained(model_name)
model = RobertaModel.from_pretrained(model_name)

def warm_up():
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

#to put here

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


#Find a sentence with n words, that is tokenized into m > n tokens by the tokenizer.
    sentence_n_word = "I'm so happy"
    tokens = tokenizer.tokenize(sentence_n_word)
    print(
        f" for the sentence 'I'm so happy' that has {len(sentence_n_word.split())} words has been tokinze into {len(tokens)} words")
def task_11():
    with open("ass1-tagger-train", 'r') as file:
        lines = file.readlines()
        word_pos_list = [word for line in lines for word in line.split()]

    word_types_freq = {}
    pos_freq = {}
    # Print the result
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

    most_frequent_types = {word: max(types_freq, key=types_freq.get) for word, types_freq in word_types_freq.items()}

    # NO LABELS
    with open("ass1-tagger-dev-input", 'r') as file:
        lines = file.readlines()
        words = [word for line in lines for word in line.split()]

    tokens = [word.lower() for word in words]
    dict_stickers = {}
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

    true_positives = 0
    for word, pos in dict_stickers.items():
        if word.lower() in dict_labels and dict_labels.get(word.lower()) == dict_stickers.get(word.lower()):
            true_positives += 1

    print(true_positives / len(dict_stickers))



def rules(word, tokens):
    prev_word = tokens.index(word) - 1
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

    if tokens[prev_word] == 'the':
        return 'NNP'
    if tokens[prev_word] in ['he', 'she', 'it', 'they']:
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



if __name__ == "__main__":
    warm_up()
    task_11()
