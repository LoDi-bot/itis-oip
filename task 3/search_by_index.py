import re
import sys

from nltk import RegexpTokenizer
from pymorphy2 import MorphAnalyzer
from make_inverted_index import get_inverted_index

ALL_DOCUMENTS = set(range(1794))
inverted_index = get_inverted_index()


def tokenize(word):
    # токенизатор на регулярных выражениях
    tknzr = RegexpTokenizer(r'[А-Яа-яёЁ&(\|)~\)\(]+')
    clean_words = tknzr.tokenize(word)
    # print(clean_words)
    clean_words = [w.lower() for w in clean_words if w != '']
    return list(clean_words)


def lemmatize(tokens):
    pymorphy2_analyzer = MorphAnalyzer()
    lemmas = []
    for token in tokens:
        if re.match(r'[А-Яа-яёЁ]', token):
            lemma = pymorphy2_analyzer.parse(token)[0].normal_form
            lemmas.append(lemma)
        else:
            lemmas.append(token)
    return lemmas


def priority(oper):
    if oper == '&':
        return 3
    elif oper == '|':
        return 2
    elif oper == '(':
        return 1
    return -1


def get_postfix(tokens):
    result = []
    stack = []
    for token in tokens:
        if token not in ['&', '|', '(', ')']:
            result.append(token)
        elif token == '(':
            stack.append(token)
        elif token == ')':
            top_operand = stack.pop()
            while not top_operand == '(':
                result.append(top_operand)
                top_operand = stack.pop()
        else:
            while len(stack) > 0 and priority(peek_stack(stack)) >= priority(token):
                result.append(stack.pop())
            stack.append(token)

    while not len(stack) == 0:
        result.append(stack.pop())
    return result


def peek_stack(stack):
    if stack:
        return stack[-1]
    else:
        return None


def get_index(token):
    if token[0] == '~':
        try:
            indices = set(inverted_index[token[1:]])
            return ALL_DOCUMENTS - indices
        except KeyError:
            return set()
    else:
        try:
            index = inverted_index[token]
            return set(index)
        except KeyError:
            return set()


def evaluate(tokens):
    stack = []
    for token in tokens:
        if token in ['&', '|']:
            arg2, arg1 = stack.pop(), stack.pop()
            if token == '&':
                result = arg1 & arg2
            else:
                result = arg1 | arg2
            stack.append(result)
        else:
            stack.append(get_index(token))
    return stack.pop()


def tokenize_query(query):
    tokenized_query = []

    for (index, word) in enumerate(query.split(' ')):
        if word in ['&', '|', '(', ')']:
            tokenized_query.append(word)
        else:
            if word[0] == '~':
                tokenized_word = lemmatize(tokenize(word[1:]))[0]
                tokenized_query.append('~' + tokenized_word)
            else:
                tokenized_word = lemmatize(tokenize(word))[0]
                tokenized_query.append(tokenized_word)

    return tokenized_query


def search(query):
    tokenized_query = tokenize_query(query)
    print("Tokenized query: %s" % " ".join(tokenized_query))
    postfix_query = get_postfix(tokenized_query)
    print("Сonverted query: %s" % " ".join(postfix_query))
    result = evaluate(postfix_query)
    print(result)


def test():
    queries = {
        "википедия & планета",
        "Египет & пирамида",
        "Египет & ~пирамида & Рим & война",
        "Клеопатра & Цезарь | Антоний & Цицерон | Помпей",
        "Клеопатра & ~Цезарь & музыка | Антоний & Цицерон & книга | Помпей & ~остров",
    }
    for query in queries:
        search(query)


if __name__ == '__main__':
    query = sys.argv[1]
    # test()
    search(query)
