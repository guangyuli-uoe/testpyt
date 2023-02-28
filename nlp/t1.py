from nlp import preprocess as pre
import matplotlib as mtl
import matplotlib.pyplot as plt

if __name__ == '__main__':
    txtpath = './timemachine.txt'

    corpus, vocab = pre.load_corpus_tm(txtpath, mode='char')
    # print(corpus)
    print(vocab.token_to_idx)
    print(len(corpus))
    print(len(vocab))


    tokens = pre.tokenizer(pre.read_text(txtpath), 'word')
    # print(tokens)
    w_vocab = pre.Vocab(tokens)
    # print(w_vocab)
    tokens_freq = w_vocab.token_freqs
    print(f'tokens_freq: {tokens_freq}')
    freqs = [freq for token,freq in tokens_freq]
    # print(freqs)

    # probability of appearance of term
    prob = [(freq/sum(freqs)) for freq in freqs]
    # print(prob)
    # print(enumerate(freqs))
    rank = [i for i,freq in enumerate(freqs)]
    # print(rank)
    plt.style.use('ggplot')
    fig, axes = plt.subplots(2, 2
                             # facecolor='gray', # 设置背景颜色
                             # sharex=True,
                             # sharey=True
                             )
    axes[0, 0].set_xscale('log', base=10)
    axes[0, 0].set_xscale('log', base=10)
    axes[0, 0].plot(rank, freqs)
    axes[0, 0].set_title('log-log')
    axes[0, 0].set_ylabel('log(freq)')
    axes[0, 0].set_xlabel('log(rank)')

    axes[0, 1].plot(rank, freqs)

    axes[1, 0].plot(rank, prob)


    plt.grid()
    # plt.legend()
    # plt.savefig('result.pdf')
    plt.show()

