from nlp import preprocess as pre

import matplotlib.pyplot as plt


if __name__ == '__main__':
    txtpath = './timemachine.txt'

    '''
        corpus: [28, 40, 13,]
        bigram_tokens: (1012, 4), (4, 65)]
        
    '''
    corpus, vocab = pre.load_corpus_tm(txtpath, mode='word')
    # print(corpus)
    '''
        Vocab class
            可以接收 
                original：
                        list of list
                    [['the', 'time', 'machine', 'by', 'h', 'g', 'wells'], [],
                        list of tuple
                    [('the', 'time'), ('time', 'machine')]
            或
                list
                    tokens
                        ['the', 'time', 'machine', 
                
    '''
    tokens = vocab.to_tokens(corpus)
    # print(tokens[:10]) # ['the', 'time', 'machine', 'by', 'h', 'g', 'wells', 'i', 'the', 'time']
    '''
        i love you
            i love, love you
        i love
        love you
    '''
    bigram = [pair for pair in zip(tokens[:-1], tokens[1:])]
    print(f'bigram: {bigram[:10]}')
    bigram_vocab = pre.Vocab(bigram)
    print(bigram_vocab.token_freqs[:10])


    # vocab_t = pre.Vocab(tokens)
    # print(f'vocab_t, list : {vocab_t.token_freqs[:10]}')
    # print(f'vocab, list of list: {vocab.token_freqs[:10]}')

    '''
        i love you baby
            i love you, love you baby
        i love [:-2]
        love you [1:-1]
        you baby [2:]

        i love you my baby
            i love you, love you my, you my baby
        i love you
        love you my
        you my baby
    '''
    trigram = [pair for pair in zip(tokens[:-2], tokens[1:-1], tokens[2:])]
    print(f'trigram: {trigram[:10]}')
    trigram_vocab = pre.Vocab(trigram)
    print(trigram_vocab.token_freqs[:10])
    print(f'unigram: {vocab.token_freqs[:10]}')


    '''
        plot
    '''

    # plt.style.use('ggplot')
    fig, axes = plt.subplots(1, 1
                             # facecolor='gray', # 设置背景颜色
                             # sharex=True,
                             # sharey=True
                             )
    unigram_freq = [freq for token,freq in vocab.token_freqs]
    bigram_freq = [freq for token,freq in bigram_vocab.token_freqs]
    trigram_freq = [freq for token,freq in trigram_vocab.token_freqs]
    x_uni = [i for i, freq in enumerate(unigram_freq)]
    x_bi = [i for i, freq in enumerate(bigram_freq)]
    x_tri = [i for i, freq in enumerate(trigram_freq)]

    axes.set_xscale('log', base=10)
    axes.set_yscale('log', base=10)
    axes.plot(x_uni, unigram_freq, label='unigram')
    axes.plot(x_bi, bigram_freq, label='bigram')
    axes.plot(x_tri, trigram_freq, label='trigram')
    axes.set_ylabel('log(freq)')
    axes.set_xlabel('log(rank)')

    plt.grid()
    plt.legend()
    # plt.savefig('ngram.pdf')
    plt.show()







