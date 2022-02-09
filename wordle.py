from enum import Enum
from math import log2
from random import shuffle
from functools import lru_cache
import pickle
import argparse

class Outcome(Enum):
    NULL = 0,
    NONE = 1,
    PARTIAL = 2,
    FULL = 3,


start_words = {}
abc = lambda: map(chr, range(ord('A'), ord('Z')+1))

DEFAULT_WORDLIST = 'data/words_alpha.txt'
START_WORDS = 'data/start_words.pickle'


@lru_cache(maxsize=2**15)
def outcome(w, t):
    """Get the outcome of playing a word w against target t. If a letter in w
    is ., it will be disregarded.
    """

    n = len(w)
    ft = {c: 0 for c in abc()}
    for c in t:
        ft[c] += 1

    r = [Outcome.NULL]*n

    for i in range(n):
        if w[i] == '.':
            continue

        if w[i] == t[i]:
            r[i] = Outcome.FULL
            ft[w[i]] -= 1
        else:
            r[i] = Outcome.NONE

    for i in range(n):
        if r[i] == Outcome.NONE and ft[w[i]] > 0:
            r[i] = Outcome.PARTIAL
            ft[w[i]] -= 1

    return tuple(r)
    

def entropy(w, T):
    """Calcuate the entropy of playing a word against a word set T."""
    n = len(T)
    X = {}

    for t in T:
        r = outcome(w, t)
        if r not in X:
            X[r] = 0
        X[r] += 1/n

    h = -sum(X[p] * log2(X[p]) for p in X)
    return h


def make_guess(W, T, k, a=3):
    """Make a guess based on the available target wordset T, given available
    choices W. T is assumed to be a subset of W.

    This code is probably incomprehensible unless you read the README.
    """
    if len(T) == 0:
        return None
    if len(T) == 1:
        return T[0]

    def single_mask(c, i):
        m = ['.'] * k
        m[i] = c
        return ''.join(m)

    # H(Y_ci)
    HY = {}
    for c in abc():
        for i in range(k):
            HY[(c, i)] = entropy(single_mask(c, i), T)

    # Estimated H(X_w)
    EHX = []
    for w in W:
        h = sum(HY[(c, i)] for (i, c) in enumerate(w))
        EHX.append((h, w))

    EHX.sort(reverse=True)
    n = len(EHX)
    if n == 0:
        return None
    # Check a% of the words, and at least 10
    HX = EHX[:min(n, int(10 + a*n/10))]
    HX = [(entropy(w, T), w) for (_, w) in HX]

    (hw, w) = max(HX)
    if w in T:
        return w
    # Entropy remaining
    hr = log2(len(T))
    (ht, t) = max([(entropy(t, T), t) for t in T])
    # Expected turns remaining if we play a non-target word    = 2^(hr - hw)/2
    # Expected turns remaining if we play the best target word = (|T|-1)/|T| * 2^(hr - ht)/2
    if 2**(hr - hw)/2 > (len(T)-1)/len(T) * 2**(hr - ht)/2:
        return t
    return w


def play(t, W, T, k):
    """Play out a game with a given target word, and return the list of turns
    made.
    """
    start = True
    turns = []
    while len(T) >= 1:
        if start and k in start_words:
            w = start_words[k]
        else:
            w = make_guess(W, T, k)
        start = False
        if w is None:
            print('could not guess word')
            return None

        r = outcome(w, t)
        T = [t_ for t_ in T if r == outcome(w, t_)]

        turns.append(w)
        if w == t:
            break

    return turns


def play_interactive(W, T, k):
    """Play an interactive session against an unknown target word of length k.

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
    while True:
        if start and k in start_words:
            w = start_words[k]
        else:
            w = make_guess(W, T, k)
        start = False
        if w is None:
            print('! Could not guess word, it\'s probably not in the wordlist')
            break
        print(f'> {w} ({len(T)} words)')

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
                T = [t for t in T if outcome(w, t) == r]
                break

        if r is None:
            continue
        if all([x == Outcome.FULL for x in r]):
            break


def run_test(W, T, k, n=100):
    shuffle(T)

    data = []
    for t in T[:n]:
        x = len(play(t, W, T, k))
        data.append(x)

    print(f'k={k}')
    print(f'mean turns to win: {sum(data)/n:.2f} (n={n})')
    # Janky histogram incoming
    hist = {x: 0 for x in range(0, 7)}
    for x in data:
        hist[x if x <= 6 else 0] += 1
    hist_len = 40
    top = max(hist.values())
    bar = lambda x: '='*int(x*hist_len/top) + ' '*(hist_len - int(x*hist_len/top)) + f' | {x}'
    for x in range(1, 7):
        print(f'{x} | {bar(hist[x])}')
    print(f'+ | {bar(hist[0])}')


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
    parser.add_argument(
        '--targetlist',
        default=DEFAULT_WORDLIST,
        help='Path to the list of valid target words to use'
    )
    parser.add_argument(
        '--run-test',
        type=int,
        default=False,
        help='Run a test on the algorithm with n iterations'
    )
    args = parser.parse_args()
    k = args.k

    with open(args.wordlist) as words_file:
        words = map(lambda line: line.strip().upper(), words_file.readlines())
        words = filter(lambda w: len(w) == k, words)
        words = list(words)

    with open(args.wordlist) as targets_file:
        targets = map(lambda line: line.strip().upper(), targets_file.readlines())
        targets = filter(lambda w: len(w) == k, targets)
        targets = list(targets)

    if not args.n:
        with open(START_WORDS, 'rb') as start_words_file:
            start_words = pickle.load(start_words_file)

    if args.run_test:
        run_test(words, targets, k, args.run_test)
        return

    if args.word is not None:
        turns = play(args.word, words, targets, k)
        print(turns)
    else:
        play_interactive(words, targets, k)


if __name__ == '__main__':
    main()