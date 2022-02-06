from enum import Enum
from math import log2
import pickle
import argparse

class Outcome(Enum):
    NONE = 0,
    PARTIAL = 1,
    FULL = 2


k = 0
words = []
start_words = {}
abc = lambda: map(chr, range(ord('A'), ord('Z')+1))

DEFAULT_WORDLIST = 'data/words_alpha.txt'
START_WORDS = 'data/start_words.pickle'


def outcome(w, cis):
    """Get the outcome of playing a list of guesses against a word w.
    
    Each guess is of the form (c, i): a character and an index.
    """
    def evaluate_guess(c, i):
        if w[i] == c:
            return Outcome.FULL
        k = len([1 for (c_, _) in cis[:i+1] if c == c_])
        if c in w and k <= w.count(c):
            return Outcome.PARTIAL
        return Outcome.NONE

    return tuple([evaluate_guess(c, i) for (c, i) in cis])


def entropy(W, cis):
    """Calcuate the entropy of a list of guesses played on words W.

    Each guess is of the form (c, i): a character and an index.
    """
    n = len(W)
    X = {}

    for w in W:
        r = outcome(w, cis)
        if r not in X:
            X[r] = 0
        X[r] += 1/n

    h = -sum(X[p] * log2(X[p]) for p in X)
    return h


def mask_to_cis(mask):
    return [(c, i) for (i, c) in enumerate(mask) if c != '_']


def make_guess(W, a=1):
    """Make a guess based on the available words W.

    Set hard_mode to True to only allow guesses from the given word set, otherwise
    guesses from all words.

    This code is probably incomprehensible unless you read the README.
    """
    if len(W) == 0:
        return None
    if len(W) == 1:
        return W[0]

    # H(Y_ci)
    HY = {}
    for c in abc():
        for i in range(len(W[0])):
            HY[(c, i)] = entropy(W, [(c, i)])

    # Estimated H(X_w)
    EHX = []
    for w in W:
        cis = mask_to_cis(w)
        h = sum(HY[(c, i)] for (c, i) in cis)
        EHX.append((h, w))

    EHX.sort(reverse=True)
    n = len(EHX)
    if n == 0:
        return None
    # Check a% of the words, and at least 10
    HX = EHX[:min(n, int(10 + a*n/100))]
    HX = [(entropy(W, mask_to_cis(word)), word) for (_, word) in HX]

    return max(HX)[1]


def play(target):
    """Play out a game with a given target word, and return the list of turns
    made.
    """
    start = True
    W = words
    turns = []
    while len(W) >= 1:
        if start and k in start_words:
            w = start_words[k]
        else:
            w = make_guess(W)
        start = False
        if w is None:
            print('could not guess word')

        cis = mask_to_cis(w)
        r = outcome(target, cis)
        W = [t for t in W if r == outcome(t, cis)]

        turns.append(w)
        if w == target:
            break

    return turns


def play_interactive():
    """Play an interactive session against an unknown target word.

    Each turn, a guess will be given. The user needs to type in the outcome of
    the guess, using g/y/. to represent full/partial/none (green/yellow/gray)
    results. If the guess is not a valid word, just hit enter and a new guess
    will be fetched.
    """
    def mask_to_outcomes(mask):
        if len(mask) != k:
            return (f'Mask must be {k} letters long', None)
        r = []
        for c in mask:
            if c == 'g' or c == 'G':
                r.append(Outcome.FULL)
            elif c == 'y' or c == 'Y':
                r.append(Outcome.PARTIAL)
            elif c == '.':
                r.append(Outcome.NONE)
            else:
                return ('Mask must only contain g/y/. (case-insensitive)', None)
        return (None, tuple(r))

    start = True
    W = words
    while True:
        if start and k in start_words:
            w = start_words[k]
        else:
            w = make_guess(W)
        start = False
        if w is None:
            print('! Could not guess word, it\'s probably not in the wordlist')
            break
        print(f'> {w} ({len(W)} words)')

        r = None
        while True:
            mask = input('? ')
            if mask == '':
                W = [t for t in W if t != w]
                break

            (err, r) = mask_to_outcomes(mask)
            if err is not None:
                print('!', err)
                continue
            else:
                W = [t for t in W if outcome(t, mask_to_cis(w)) == r]
                break

        if r is None:
            continue
        if all([x == Outcome.FULL for x in r]):
            break


def main():
    global k, words, start_words

    parser = argparse.ArgumentParser()
    parser.add_argument(
        'word',
        nargs='?',
        help='The word to be played against - if left empty, will launch an interactive session'
    )
    parser.add_argument(
        '-k',
        type=int,
        default=5,
        help='The number of letters in the word'
    )
    parser.add_argument(
        '-n',
        action='store_true',
        default=False,
        help='Set if you do not want to use the precomputed start words'
    )
    parser.add_argument(
        '--wordlist',
        default=DEFAULT_WORDLIST,
        help='Path to the list of valid words to use'
    )
    args = parser.parse_args()
    k = args.k

    with open(args.wordlist) as words_file:
        words = map(lambda line: line.strip().upper(), words_file.readlines())
        words = filter(lambda w: len(w) == k, words)
        words = list(words)

    if not args.n:
        with open(START_WORDS, 'rb') as start_words_file:
            start_words = pickle.load(start_words_file)

    if args.word is not None:
        turns = play(args.word)
        print(turns)
    else:
        play_interactive()


if __name__ == '__main__':
    main()