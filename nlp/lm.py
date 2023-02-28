from nlp import preprocess as pre



if __name__ == '__main__':
    txtpath = './timemachine.txt'

    corpus, vocab = pre.load_corpus_tm(txtpath, mode='word')
    print(corpus[:10])
    # print(vocab.token_freqs)
    print(corpus[:-1][:10])
    print(vocab.to_tokens(corpus[:10]))

    print(len(corpus))
    print(len(corpus[:-1]))
    print(len(corpus[1:]))
    print(list(vocab.token_to_idx.items())[:10])

    lines = pre.read_text(txtpath)
    tokens = pre.tokenizer(lines, 'word')
    corpus = [vocab[token] for line in lines for token in line]
    print(corpus[:10])
    # corpus1 = [vocab[token] for ]
    vocb = pre.Vocab(tokens)
    # print(vocb[])
    print(tokens)
    print(lines)

    count = 0
    for line in lines:
        print(f'line: {line}')
        count += 1
        for token in line:
            print(f'token: {token}')
            break

        if count >= 10:
            break
