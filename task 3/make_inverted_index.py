from collections import defaultdict
import os

DIRECTORY = 'C:/Users/Asadu/Desktop/itis-oip/task 2/Для каждого файла'


def get_inverted_index():
    term_documents_dict = defaultdict(set)
    idx = 0
    for root, dirs, files in os.walk(DIRECTORY):
        for file in files:
            if file.lower().endswith('.txt') and file.lower().startswith('lemmas'):
                path_file = os.path.join(root, file)
                with open(path_file, encoding="utf=8") as f:
                    lemmas = list(map(lambda line: line.strip().split('\n')[0], f.readlines()))
                    # print(f'{path_file}:\n{lemmas}')
                for lemma in lemmas:
                    term_documents_dict[lemma].add(idx)
                idx += 1
    return term_documents_dict


if __name__ == '__main__':
    td_dict = get_inverted_index()
    with open('inverted_index.txt', 'w', encoding='utf-8') as f:
        for k, v in td_dict.items():
            f.write(k + ' ' + ' '.join(map(str, v)) + '\n')
    count_inverted_word = []
    for k, v in td_dict.items():
        count_inverted_word.append({"count": len(v), "inverted_array": v, "word": k})
    with open('inverted_index_2.txt', 'w', encoding='utf-8') as f:
        for ciw in count_inverted_word:
            f.write(str(ciw) + '\n')
