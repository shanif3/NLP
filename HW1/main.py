import os
import pickle
import gensim.downloader as dl
import matplotlib.pyplot as plt
import numpy as np
from sklearn import decomposition


print("Welcome!\nFirst, we are loading the models, it may take a while.")

if os.path.exists('word2vec.pkl'):
    with open('word2vec.pkl', 'rb') as model_file:
        word2vec_model = pickle.load(model_file)
else:
    model = dl.load("word2vec-google-news-300")
    with open('word2vec.pkl', 'wb') as model_file:
        pickle.dump(model, model_file)
        word2vec_model = model

if os.path.exists('wiki_news_model.pkl'):
    with open('wiki_news_model.pkl', 'rb') as model_file:
        word2vec_model = pickle.load(model_file)
else:
    model = dl.load("glove-wiki-gigaword-200")
    with open('wiki_news_model.pkl', 'wb') as model_file:
        pickle.dump(model, model_file)
        wiki_news_model = model

if os.path.exists('glove-twitter-200.pkl'):
    with open('twit_model.pkl', 'rb') as model_file:
        twit_model = pickle.load(model_file)
else:
    model = dl.load("glove-twitter-200")
    with open('twit_model.pkl', 'wb') as model_file:
        pickle.dump(model, model_file)
        twit_model = model


def generate_most_smilar_words():

    """
    Generating lists of the most similar words
    Choose 5 words in the vocabulary, and for each of them generate the list of 20 most similar
    words according to word2vec
    """
    # choosing 5 words for vocab
    voc_list = ['game', 'back', 'now', 'some', 'million']
    for i in voc_list:
        print(f"'{i}' most similar: {word2vec_model.most_similar(i, topn=20)}")


def polysemous_words():

    """
    Polysemous Words
    Find three polysemous words (words with at least two different meanings) such that
    the top-10 neighbors of each word reflect both word meanings, and three polysemous words
    such that the top-10 neighbors of each word reflect only a single meaning.
    """

    # Group 1 - **two** meaning in top 10 neighbors
    poly_group1 = ['draft', 'bat', 'tie']
    for i in poly_group1:
        print("Here are the top-10 neighbors of the word " + "'" + i + "'" + " that reflects both word meanings \n")
        print(model.most_similar(i))  # default is top 10 neighbors
        print("\n")

    """Group 2 - **one** meaning in top 10 neighbors
    The possible senses for bank is a **financial institution licensed to receive deposits and make loans** or slope... The first sense was reflected in the top-10 neighbors.
    The possible senses for book is a **book store** or to book a fly... The first sense was reflected in the top-10 neighbors.
    The possible senses for fan is to **admire someone** or a cool by waving something to create a current of air... The first sense was reflected in the top-10 neighbors.
    """

    poly_group2 = ['bank', 'book', 'fan']
    for i in poly_group2:
        print("Here is the top-10 neighbors of the word " + "'" + i + "'" + " reflect only a single meaning \n")
        print(model.most_similar(i))  # defulat is top 10 neighbors
        print("\n")



def synonyms_antonyms():

    """
    Synonyms are words that share the same meaning. Antonyms are words that have an opposite meaning (like “cold” and “hot”).
    We gonna find a triplet of words (w1, w2, w3) such that all of the conditions hold.
    """

    w1 = 'happy'
    w2 = 'joyful'
    w3 = 'sad'
    if word2vec_model.similarity(w1, w2) < word2vec_model.similarity(w1, w3):
        print(f"all the conditions are hold! our triplet is ({w1},{w2},{w3})")


def different_corpora():

    """
    In this section, we would like to compare models based on two sources.
    The first model is based on wikipedia and news text, and the second based on twitter data.
    For the wikipedia and news model, use the gensim model glove-wiki-gigaword-200 .
    For the twitter data, use the gensim model glove-twitter-200(“glove” is a different algorithm than word2vec, but its
    essence is similar).
    """

    print(
        "The 5 words that we found whose top 10 neighbors based on the news corpus are very similar to their top 10 neighbors based on the twitter corpus are 'white', 'green', 'blue', 'gray' and 'red'")

    words = {'white', 'green', 'blue', 'gray', 'red'}
    for i in words:
        list1_boy = wiki_news_model.most_similar(i)
        list2_boy = twit_model.most_similar(i)
        words1 = [pair[0] for pair in list1_boy]
        words2 = [pair[0] for pair in list2_boy]
        common_words = set(words1) & set(words2)
        print(f"Similarity score for the word {i}: {len(common_words)} out of 10 ")

    # Find 5 words whose top 10 neighbors based on the news corpus are substantially different from the top 10 neighbors based on the twitter corpus.
    print(
        "The 5 words that we found whose top 10 neighbors based on the news corpus are substantially different from the top 10 neighbors based on the twitter corpus are 'towel','current','bark','rose' and 'bat'")

    words = {'towel', 'current', 'bark', 'rose', 'bat'}
    for i in words:
        list1 = wiki_news_model.most_similar(i)
        list2 = twit_model.most_similar(i)
        print(f"list1: {list1}\n list2: {list2}")
        words1 = [pair[0] for pair in list1]
        words2 = [pair[0] for pair in list2]
        common_words = set(words1) & set(words2)
        print(f"Similarity score for the word {i}: {len(common_words)} out of 10 ")


def plot_words_2d():

    """
    Dimensionality reduction is a technique by which you take n-dimensional data and transform it
    into m-dimensional data (m < n), while attempting to maintain properties (such as distances) of
    the original data. Of course, dimensionality reduction is always a lossy process (because there
    is more information in n-dimension than in m<n dimensions). Yet, it is still useful.
    """

    # Taking the first 5000 words in the vocabulary
    voc_5000 = word2vec_model.index_to_key[1:5001]

    voc_endswith = [word for word in voc_5000 if
                    word.endswith('ed') or word.endswith('ing')]  # we gonna be left with 708 words as expected

    # creating a matrix with 708 rows, where each row is a 300-dim vector for one word
    # We are using word2vec-google-news-300 model that has word vectors with a dimensionality of 300
    matrix = np.array([model[word] for word in voc_endswith])

    # One way of dimensionality reduction is via the PCA algorithm
    pca = decomposition.PCA(n_components=2)
    words_2d_vector = pca.fit_transform(matrix)

    # Plot the resulting 2-d vectors
    ed_words = [word for word in voc_endswith if word.endswith("ed")]
    matrix_ed = np.array([model[word] for word in ed_words])

    ing_words = [word for word in voc_endswith if word.endswith("ing")]
    matrix_ing = np.array([model[word] for word in ing_words])

    plt.figure(figsize=(10, 6))

    # plotting words ending with 'ed' in blue
    plt.scatter(matrix_ed[0], matrix_ed[1],
                color='blue',
                label='Ends with "ed"'
                )

    # plotting words ending with "ing" in green
    plt.scatter(
        matrix_ing[0], matrix_ing[1],
        color='green',
        label='Ends with "ing"'
    )

    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.title('2D Word Vectors Scatter Plot')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    flag = True
    while flag:
        choice = input(
            "We are offering multiple methods to run, what would you like to choose?\n 1.Generating lists of the most similar words\n2.Polysemous Words\n3.Synonyms and Antonyms\n4.The Effect of Different Corpora\n5.Plotting words in 2D\n5.Exit\nMy choice is:")
        if choice == '1':
            generate_most_smilar_words()
        elif choice == '2':
            polysemous_words()
        elif choice == '3':
            synonyms_antonyms()
        elif choice == '4':
            different_corpora()
        elif choice == '5':
            plot_words_2d()
        elif choice == '6':
            flag = False
        else:
            print("Invalid choice. Please enter a number between 1 and 6.")