from wordle import make_guess
from pickle import dump


MIN_LETTERS = 4
MAX_LETTERS = 11

k_iter = lambda: range(MIN_LETTERS, MAX_LETTERS+1)


words = {k: [] for k in k_iter()}
with open('data/wordle_answers.txt') as words_file:
    all_words = map(lambda line: line.strip().upper(), words_file.readlines())
    for word in all_words:
        if MIN_LETTERS <= len(word) <= MAX_LETTERS:
            words[len(word)].append(word)


start_words = {}
for k in k_iter():
    start_words[k] = make_guess(words[k])
    print(k, start_words[k])


# with open('start_words.pickle', 'wb') as start_words_file:
#     dump(start_words, start_words_file)