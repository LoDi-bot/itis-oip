import math
import os
import re

import nltk
from nltk.tokenize import RegexpTokenizer
from pymorphy2 import MorphAnalyzer
from nltk.corpus import stopwords

# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('snowball_data')
# nltk.download('perluniprops')
# nltk.download('universal_tagset')
# nltk.download('nonbreaking_prefixes')
# nltk.download('wordnet')

stop_words = stopwords.words('russian') + stopwords.words('english')
ALL_DOCS_COUNT = 1794
DIRECTORY_LEMMAS_TF_IDF = 'C:/Users/Asadu/Desktop/itis-oip/task 4/tf_idf_lemmas'

def get_tokens(text):
    # токенизатор на регулярных выражениях
    tokenizer = RegexpTokenizer('[А-Яа-яёЁ]+')
    clean_words = tokenizer.tokenize(text)
    clean_words = [w.lower() for w in clean_words if w != '']
    clean_words = [w for w in clean_words if w not in stop_words]
    return set(clean_words)


def get_lemmas(tokens):
    pymorphy2_analyzer = MorphAnalyzer()
    lemmas = []
    for token in tokens:
        if re.match(r'[А-Яа-яёЁ]', token):
            lemma = pymorphy2_analyzer.parse(token)[0].normal_form
            lemmas.append(lemma)
    return lemmas


def vector_norm(vec):
    return sum([el ** 2 for el in vec]) ** 0.5


def get_index():
    with open('C:/Users/Asadu/Desktop/itis-oip/task 1/index.txt') as f:
        return {int(s.split()[0]): s.split()[1] for s in f.readlines()}


def get_tf_terms():
    tf_idf_dicts = []
    idx = 0
    for root, dirs, files in os.walk(DIRECTORY_LEMMAS_TF_IDF):
        for file in files:
            if file.lower().endswith('.txt') and file.lower().startswith('tf_idf_lemmas'):
                path_file = os.path.join(root, file)
                with open(path_file, encoding="utf=8") as f:
                    tf_idf_dicts.append({str(line.split()[0]): float(line.split()[1]) for line in f.readlines()})
                idx += 1
    return tf_idf_dicts


def calculate(term, tokens_list, documents_count, documents_with_term_count):
    tf = tokens_list.count(term) / len(tokens_list)
    if documents_with_term_count == 0:
        idf = 0
    else:
        idf = math.log(documents_count / documents_with_term_count)

    return round(tf, 6), round(idf, 6), round(tf * idf, 6)


def cosine_similarity(vec1, vec2):
    dot = 0
    for x1, x2 in zip(vec1, vec2):
        dot += x1 * x2
    if dot == 0:
        return 0
    return dot / (vector_norm(vec1) * vector_norm(vec2))


def search(query):
    print("SEARCHING: {}".format(query))
    tokens = get_lemmas(get_tokens(query))
    index_dict = get_index()
    tf_idf_dicts_lemmas = get_tf_terms()
    if len(tokens) == 0:
        print("Empty query")
        return

    print("LEMMATIZED: {}\n".format(' '.join(tokens)))
    query_vector = []

    for token in tokens:
        doc_with_terms_count = sum(token in tf_idf_dict for tf_idf_dict in tf_idf_dicts_lemmas)
        # print(doc_with_terms_count)
        _, _, tf_idf = calculate(token,
                                 tokens,
                                 ALL_DOCS_COUNT,
                                 doc_with_terms_count)
        query_vector.append(tf_idf)

    distances = {}

    for index in range(ALL_DOCS_COUNT):
        document_vector = []

        for token in tokens:
            try:
                tf_idf = tf_idf_dicts_lemmas[index][token]
                document_vector.append(tf_idf)
            except KeyError:
                document_vector.append(0.0)

        distances[index] = cosine_similarity(query_vector, document_vector)

    searched_indices = sorted(distances.items(), key=lambda x: x[1], reverse=True)
    result_data = []
    print('query_vector', query_vector)
    print('searched_indices', searched_indices)
    for doc_id, cosine_sim in searched_indices:
        if cosine_sim < 0.05:
            continue

        print("Index: {}\n Link: {}\n Cosine:{}\n".format(doc_id, index_dict[doc_id], cosine_sim))
        result_data.append({'doc_id': doc_id, 'link': index_dict[doc_id], 'cosine_sim': cosine_sim})
    return result_data


if __name__ == '__main__':
    query = input()
    search(query)

