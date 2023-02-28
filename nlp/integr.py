from nlp import preprocess as pre

if __name__ == '__main__':
    txtpath = './timemachine.txt'

    # list of list
    tokens = pre.tokenizer(pre.read_text(txtpath), 'word')
    # print(tokens)
    corpus = [token for line in tokens for token in line]
    # print(corpus) # list
    vocab1 = pre.Vocab(corpus)
    vocab2 = pre.Vocab(tokens)

    print(vocab1.token_freqs)
    print(vocab2.token_freqs)

    corpus11, vocab11 = pre.load_corpus_tm(txtpath, mode='word')
    print(vocab11.token_freqs)
    # corpus = [vocab[token] for line in lines for token in line]
    corpus_test = [vocab11[word] for line in tokens for word in line]
    print(corpus_test[:10])
