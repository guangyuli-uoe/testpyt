from nlp import preprocess as pre

if __name__ == '__main__':
    txtpath = './timemachine.txt'

    '''
        corpus: [28, 40, 13,]
        bigram_tokens: (1012, 4), (4, 65)]
    '''
    corpus, vocab = pre.load_corpus_tm(txtpath, mode='word')
    # print(corpus)
    # print(vocab.to_tokens(corpus))
    print(pre.Vocab(vocab.to_tokens(corpus)))
    print([pair for pair in zip(vocab.to_tokens(corpus)[:-1], vocab.to_tokens(corpus)[1:])][:10])
    bigram_tokens_id = [pair for pair in zip(corpus[:-1], corpus[1:])]
    # print(bigram_tokens)
    bigram_vocab_id = pre.Vocab(bigram_tokens_id)
    print(bigram_vocab_id.token_freqs[:10])
    # print(tokens)

    tokens = vocab.to_tokens(corpus)
    '''
        i love you
        bigram: i love
                love you
        i love
        love you
    '''
    bigram_tokens = [pair for pair in zip(tokens[:-1], tokens[1:])]
    # print(bigram_tokens)
    bigram_vocab = pre.Vocab(bigram_tokens)
    print(bigram_vocab.token_freqs[:10])

