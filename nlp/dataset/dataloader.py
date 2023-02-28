from nlp import preprocess as pre
import random
import torch
import numpy as np

def seq_iter_random(corpus, batch_size, num_steps):
    '''

    :param corpus:
    :param batch_size:
    :param num_steps:  每个样本对长度
    :return:
    '''

    random_point = random.randint(0, num_steps-1)
    # print(corpus[-10:])
    # print(f'corpus[num_steps]: {corpus[num_steps]}')
    print(f'random: {random_point}')
    corpus = corpus[random_point:]
    # print(corpus)
    '''
        num_steps: 32770
        len: 32775
        random: 32768
        
        remain: 32775-(32768-1) = 8
        
        32775 - (32769 -1) = 7
        
        少算了第0个元素
        
        1 2 3 4 5 6
        
    '''

    num_subseqs = (len(corpus)-1) // num_steps
    print(f'num_subseqs: {num_subseqs}')

    # print(5 // 2) # 2

    initial_indices = list(range(0, num_subseqs * num_steps, num_steps))
    print(initial_indices)
    print(f'initial_indices[-1]: {initial_indices[-1]}')
    random.shuffle(initial_indices)
    print(initial_indices)
    print(f'len(initial): {len(initial_indices)}')

    def data(pos):
        return corpus[pos: pos+num_steps]

    num_batches = num_subseqs // batch_size
    print(f'num_batches * batch_size: {num_batches * batch_size}')

    for i in range(0, batch_size * num_batches, batch_size):
        initial_indices_per_batch = initial_indices[i: i+batch_size]
        '''
            initial_indices_per_batch 里， 已经是很靠后的 initial_indices的元素
        '''
        X = [data(j) for j in initial_indices_per_batch]
        Y = [data(k+1) for k in initial_indices_per_batch]
        yield torch.tensor(X), torch.tensor(Y)





def my_iter_random(corpus, batch_size, num_steps, mode):
    # num_subseqs = (len(corpus) - 1) // num_steps
    num_subseqs = len(corpus) + 1 - num_steps
    last_subseq_initial = len(corpus) - num_steps
    # initial_indices = list(range(0, num_subseqs * num_steps, num_steps))

    # new_split_corpus1 = []
    # for i in range(0, last_subseq_initial+1): # +1是因为 last_subseq_initial 在range里取不到
    #     subseq = corpus[i:i+num_steps]
    #     new_split_corpus1.append(subseq)

    new_split_corpus2 = [corpus[i:i+num_steps] for i in range(0, last_subseq_initial+1)]

    # print(f'new_split_corpus1: {len(new_split_corpus1)}, {new_split_corpus1}')
    # print(f'new_split_corpus2: {len(new_split_corpus2)}, {new_split_corpus2}')
    print(f'batch_size: {batch_size}')
    print(f'num_subseqs: {num_subseqs}')
    print(f'len(new_split_corpus2): {len(new_split_corpus2)}')



    def data(pos):
        return new_split_corpus2[pos]

    if mode == 'random':
        num_batches = len(new_split_corpus2) // batch_size
        print(f'num_batches: {num_batches}')
        for i in range(0, num_batches):
            # np.random.randint,  # [ , )
            random_seed = list(np.random.randint(0, len(new_split_corpus2) - 1, size=batch_size))
            # print(f'random_seed: {i}, {random_seed}')

            # X = []
            # Y = []
            # for seed in random_seed:
            #     X.append(data(seed))
            #     Y.append(data(seed+1))
            X = [data(j) for j in random_seed]
            Y = [data(j + 1) for j in random_seed]
            yield torch.tensor(X), torch.tensor(Y)
    elif mode == 'sequential':
        # random_seed = list(np.random.randint(0, len(new_split_corpus2) - 1, size=batch_size))
        num_batches = num_subseqs - batch_size
        print(f'num_batches: {num_batches}')
        initial_seed = [i for i in range(batch_size)]
        for i in range(0, num_batches):
            X = [data(j) for j in initial_seed if j+1 < len(new_split_corpus2)]
            Y = [data(j+1) for j in initial_seed if j+1 < len(new_split_corpus2)]
            initial_seed = [seed+1 for seed in initial_seed]
            if len(X) == batch_size:
                '''
                    len(num_subseqs) = 2
                    if batch_size = 2,
                    num_batches = 1
                '''
                yield torch.tensor(X), torch.tensor(Y)
            else:
                print('batch_size > num_batches, cannot yield')

        # for i in range(0, num_batches):
        #     # np.random.randint,  # [ , )
        #     X = []
        #     Y = []
        #     for seed in random_seed:
        #         if seed + 1 < len(new_split_corpus2):
        #             X.append(data(seed))
        #             Y.append(data(seed+1))


            # X = [data(j) for j in random_seed]
            # Y = [data(j + 1) for j in random_seed]
            # random_seed = [idx + 1 for idx in random_seed]
            # if len(X) == batch_size:
            #     yield torch.tensor(X), torch.tensor(Y)








if __name__ == '__main__':
    txtpath = '../timemachine.txt'
    corpus, vocab = pre.load_corpus_tm(txtpath, mode='word')
    # print(corpus)
    print(f'len(corpus): {len(corpus)}')
    print(len(vocab))

    # seq_iter_random(corpus,3, 5)

    my_seq = list(range(10))
    # print(my_seq)
    '''
        可以生成
            [len(corpus)-1] / batch_size 个 subsequence对（3组X，Y）
    '''
    # for X,Y in seq_iter_random(my_seq, batch_size=2, num_steps=5):
    #     print(f'X: {X}')
    #     print(f'Y: {Y}')

    '''
    
        先生成ordered的subsequence
        再shuffle
    '''

    # my_iter_random(my_seq, batch_size=3, num_steps=5)

    # count = 0
    #
    for i in range(3):
        for X, Y in my_iter_random(my_seq, batch_size=3, num_steps=5, mode='sequential'):
            # count += 1

            print(f'i: {i}')
            print(f'X: {X}')
            # print(f'X.shape: {X.shape}')  # batch_size, num_step
            print(f'Y: {Y}')
            # print(f'y.shape: {Y.shape}')
            # print(f'y.t: {Y.T}')
            # print(f'Y.T.reshape(-1): {Y.T.reshape(-1)}')
            # break