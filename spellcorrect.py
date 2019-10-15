import codecs
import time
import textdistance
import csv

from collections import Counter
from collections import defaultdict
from functools import lru_cache

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer

from multiprocessing import Pool


class Speller(object):
    """
        Поиск слов, наиболее близких по числу общих n-грамм и
        последующее ранжирование по эвристике-близости
    """

    def __init__(self, n_candidates_search=40):
        """
        :param n_candidates_search: число кандидатов-строк при поиске
        """
        # todo: может, это важный параметр?
        self.n_candidates = n_candidates_search

    def fit(self, words_list):
        """
            Подгонка спеллера
        """

        checkpoint = time.time()
        self.words_list = words_list

        self.vectorizer = CountVectorizer(analyzer="char_wb", ngram_range=(2, 2))

        encoded_words = self.vectorizer.fit_transform(words_list).tocoo()

        self.index = defaultdict(lambda: defaultdict(int))

        # строим словарь, отображающий идентификатор нграммы в множество термов
        for word_id, ngram_id, cnt in zip(encoded_words.row, encoded_words.col, encoded_words.data):
            self.index[ngram_id][word_id] = cnt

        print("Speller fitted in", time.time() - checkpoint)

        return self

    @lru_cache(maxsize=1000000)
    def rectify(self, word, next_word):
        """
            Предсказания спеллера
        """

        # запрос, преобразованный в нграммы
        counter = Counter()

        char_ngrams_list = self.vectorizer.transform([word]).tocoo()

        for _, ngram_id, cnt in zip(char_ngrams_list.row, char_ngrams_list.col, char_ngrams_list.data):
            for word_id in self.index[ngram_id]:
                counter[word_id] += 1

        # ищем терм, ближайший по хитрому расстоянию из числа выбранных
        closest_word = word
        min_distance = 1000000
        # среди топа по совпадениям по нграммам ищем "хорошее" исправление
        if word == 'вв':
            closest_word = 'в'
        elif word == 'ии':
            closest_word = 'и'
        elif word == 'сс':
            closest_word = 'с'
        elif word == 'п' and next_word != '.':
            closest_word = 'по'
        elif word == 'ан':
            closest_word = 'на'
        else:
            for suggest in counter.most_common(n=self.n_candidates):

                suggest_word = self.words_list[suggest[0]]

                distance = textdistance.damerau_levenshtein.distance(suggest_word, word)

                if distance < min_distance:
                    min_distance = distance
                    closest_word = suggest_word

        return closest_word


def process_string(data):
    id, mispelled_text = data
    mispelled_tokens = mispelled_text.split()
    was_rectified = False

    for j in range(len(mispelled_tokens)):
        if mispelled_tokens[j] not in words_set:
            if j < len(mispelled_tokens) - 1:
                rectified_token = speller.rectify(mispelled_tokens[j], mispelled_tokens[j + 1])
            else:
                rectified_token = speller.rectify(mispelled_tokens[j], '')
            mispelled_tokens[j] = rectified_token
            was_rectified = True

    if was_rectified:
        mispelled_text = " ".join(mispelled_tokens)

    return [id, mispelled_text]


if __name__ == "__main__":

    np.random.seed(0)

    # зачитываем словарь "правильных слов"
    words_set = set(line.strip() for line in codecs.open("words.txt", "r", encoding="utf-8"))
    words_list = sorted(list(words_set))

    # создаём спеллер
    speller = Speller()
    speller.fit(words_list)

    # читаем выборку
    df = pd.read_csv("broken_texts.csv")

    checkpoint1 = time.time()
    total_rectification_time = 0.0
    total_sentences_rectifications = 0.0

    y_submission = []
    counts = 0

    p = Pool()
    for i in range(20):
        print("started i = ", i)
        curDf = df[i * 1000:(i + 1) * 1000]
        ans = p.map(process_string, zip(curDf["id"], curDf["text"]))
        checkpoint2 = time.time()
        # исправляем, попутно собирая счётчики и засекая время
        print(i)
        print("elapsed", checkpoint2 - checkpoint1)
        print('', flush=True)
        with open('ans.csv', 'a') as csv_file:
            csvwriter = csv.writer(csv_file)
            csvwriter.writerows(ans)

