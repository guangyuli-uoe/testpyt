import random
import numpy as np
a = np.arange(0, 10, 1)
print(a)
b = a[random.randint(0, 3):]
print(b)

'''
    test range (0, 10, 2)
    0-9, step==2
'''
for i in range(0, 10):
    print(i)

'''
    test
        yield
'''

def test_yield(n):
    for i in range(n):
        yield i

generator1 = test_yield(3)
print(generator1)

# for i in generator1:
#     print(i)

'''
0
1
2
Traceback (most recent call last):
  File "/Users/liguangyu/testpyt/nlp/dataset/t1.py", line 30, in <module>
    print(next(generator1))
StopIteration
'''
print(next(generator1)) # 0
print(next(generator1)) # 1


def my_iter_random(corpus, batch_size, num_steps):
    # num_subseqs = (len(corpus) - 1) // num_steps
    num_subseqs = len(corpus) + 1 - num_steps
    last_subseq_initial = len(corpus) - num_steps
    # initial_indices = list(range(0, num_subseqs * num_steps, num_steps))
    # initial_indices = list(range(0, ))

    new_split_corpus1 = []
    for i in range(0, last_subseq_initial + 1):  # +1是因为 last_subseq_initial 在range里取不到
        subseq = corpus[i:i + num_steps]
        new_split_corpus1.append(subseq)

    new_split_corpus2 = [corpus[i:i + num_steps] for i in range(0, last_subseq_initial + 1)]

    print(f'new_split_corpus1: {len(new_split_corpus1)}, {new_split_corpus1}')
    print(f'new_split_corpus2: {len(new_split_corpus2)}, {new_split_corpus2}')

my_seq = list(range(6))
print(my_seq)
print('======')
for i in range(0, len(my_seq)):
    print(i)

print(my_seq[-1])
my_iter_random(my_seq, 0, num_steps=4)

aaa = np.random.randint(10, 100, size=10)
print(aaa)

initial_seed = [i for i in range(3)]
print(initial_seed)