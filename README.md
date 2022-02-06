# Wordle Player

Just for fun, my attempt at making an optimal player for Wordle. Based on a rough understanding of information theory, and the idea that the best guess for a given turn is the one that gives you the most information.

## Usage

`wordle.py` will feed you the guesses for a game of Wordle. If you run `python wordle.py`, you'll be launched into an interactive session, where the program outputs guesses and you just need to give it the outputs of each guess. Here's an example session where the word was PIANO:

```
$ python wordle.py
> TARES (15918 words)
? .y...
> ALOIN (999 words)
? y.yyy
> OMINA (5 words)
? y.ygy
> PIANO (1 words)
? ggggg
```

User input occurs on the lines starting with `?`. Each `.` represents a letter that isn't in the word at all, a `y` represents a letter that is in the word but in the wrong place, and a `g` represents a letter in the right place. Along with the guess, the program outputs the number of words remaining that it can choose from.

The default wordlist has some pretty obscure words in it, which might not be valid for your game of Wordle. If so, just enter a blank line to fetch the next-best guess:

```
$ python wordle.py
> TARES (15918 words)
? .y...
> ALOIN (999 words)
? 
> ANOIL (998 words)
? 
> ANOLI (997 words)
...
```

You can also play with other word sizes using the `-k` flag:

```
$ python wordle.py -k 11
> PERCOLATING (37539 words)
? .yy.y.gggy.
> ENUMERATION (29 words)
...
```

## Background: Entropy

From information theory, the _entropy_ `H(X)` of a random variable `X` with `n` possible events is defined as:
> `H(X) = -sum{ P(x_i) * log2(P(x_i)) } for 1 <= i <= n`

`H(X)` roughly means the amount of information* it takes to describe the outcome of `X`. For example, the entropy of a fair (50/50) coin flip is
> `H(fair coin) = -[1/2*log2(1/2) + 1/2*log2(1/2)] = 1`.

But the entropy of flipping a horribly weighted coin is
> `H(unfair coin) = -[1/10*log2(1/10) + 9/10*log2(9/10)] = 0.47`.

The fair coin takes exactly 1 bit to describe one of two outcomes, which makes sense. A flip of the unfair coin takes less information to describe because the same thing happens most of the time, and every now and then you'll need some extra bits to describe the more unlikely event.

*Measured in bits, since we use log2 - you can use any base you want, though.

## Modeling a game of Wordle

Let's assume that every "target" word in Wordle is equally likely (drawn from a dictionary of `k`-letter words). When we play some word `w`, we get some outcome (in the form of each letter being green, yellow, or gray) - this will be our random variable `X_w`. By carefully choosing the word we play, we get different expected outcomes for `X_w`, and thus get different entropies. It's kind of like choosing the weights of our coin toss, but for a much more complex event. Our goal is to find the `w` that _maximizes_ `H(X_w)`, in order to make sure that when we observe the outcome, we gain as much information as possible.

To calculate `H(X_w)`, we can check each possible target word in the dictionary and see what the outcome of playing `w` gets. There are 3^5 = 243 possible outcomes (different combinations of green, yellow, and gray), and each of these will get an associated probability depending on how many target words they map to. Calculating `H(X_w)` is then straightforward, we can use the formula from earlier:
> `H(X_w) = -sum{ P(r) * log2(P(r)) } for each possible outcome r`.

Now we've chosen a word to play. When we play it, we'll observe the outcome of `X_w`. This will eliminate many words from our dictionary - we can throw out any word that wouldn't have lead to the observed outcome. Doing this iteratively should eventually lead to us either guessing the target word or leaving exactly 1 word in the dictionary.

### Speed

This will work to find the optimal word, but it's an expensive calculation: given a dictionary with `n` words with `k` letters each, this will take `O(k * n^2)` time to run. I'm lazy so I wanted to find a fast approximation.

Instead of playing an entire word at a time, we can play a single character `c` in position `i` and observe its outcome `Y_ci`. Then, we can estimate `H(X_w)` by adding `H(Y_ci)` for each character of `w`. Now, this would give us exactly `H(X_w)` if each of `Y_ci` were independent random events, but this obviously isn't the case - for instance, there are way more words that start with `SH___` than words that start with `HS___`. (Following this strategy to calculate `H(X_w)`, the optimal starting word I got was `SAREE`, which isn't a great starter because it has two of the same letter. Playing E anywhere is a high-entropy play, because it's a very common letter.)

So instead, we can use this estimate as a filter, and just calculate the actual value of `H(X_w)` for the top `a`% of words. Using this strategy with `a = 3`, the optimal starting word I got for 5-letter words was `TARES`. I precomputed all of these for between 4 and 11 letters and saved them in `starting_words.pickle`. The `wordle.py` program uses these by default.

## Pseudocode

```
k := number of letters per word
D := dictionary of k-letter words
a := the retry percentage we choose
H := the entropy function, which takes a distribution and returns its entropy

function guess(W) {
    n = size of W

    -- H(Y_ci)
    HY := empty map of (char, number) -> number
    for each c in the alphabet:
        for each i in [0, k):
            Y := map of possible outcomes (green, yellow, or gray) -> 0
            for each t in W:
                r := the outcome of playing c in position i on target word t
                Y[r] += 1/n
            HY[(c, i)] = H(Y)

    -- Estimated H(X_w)
    EHX := empty map of string -> number
    for each w in D:
        EHX[w] = sum of HY[(w[i], i)] for i in [0, k)
    
    -- Eligible words
    D' := a% of words in D with the highest EHX[w]

    -- Actual H(X_w)
    HX := empty map of string -> number
    for each w in D':
        X := map of possible outcomes -> 0
        for each t in W:
            r := the outcome of playing w on target word t
            X[r] += 1/n
        HX[w] = H(X)
    
    return the word in D' with the maximum HX[w]
}
```

## References

- [dwyl/english-words](https://github.com/dwyl/english-words) - source for words_alpha.txt
- [boompig/wordle-py](https://github.com/boompig/wordle-py) - source for wordle_words.txt