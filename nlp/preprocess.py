import collections
import re

def read_text(txtpath):
    newlines = []
    pattern = r'[^A-Za-z]'
    with open(txtpath, 'r', encoding='utf-8') as f:
        count = 0
        for line in f:
            count += 1
            # print(line)
            newline = re.sub(pattern, ' ', line)
            newline = newline.lower().strip()
            # print(newline)
            # print()
            newlines.append(newline)

            # if count >= 30:
            #     break
        # lines = f.readlines()
        # # print(lines)
        # print(type(lines))

    return newlines


def tokenizer(lines, token):
    '''

    :param lines:
    :param token:
    :return: 将文本行拆分为单词或字符
    '''
    if token == 'word':
        tokens = [line.split() for line in lines]
        return tokens
    elif token == 'char':
        tokens = [list(line) for line in lines]
        return tokens
    else:
        print('sha sun')

def count_corpus(tokens):
    '''
    :param tokens:
    :return:
        如果tokens是list，直接调用Counter
        如果是list of list，先转化
    '''
    if len(tokens) == 0 or isinstance(tokens[0], list):
        print('11111111')
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)


class Vocab:
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        counter = count_corpus(tokens)

        # [('the', 2261), ('i', 1267),
        self.token_freqs = sorted(counter.items(), key=lambda x: x[1], reverse=True)

        self.unk, self.uniq_tokens = 0, ['<unk>'] + reserved_tokens

        self.uniq_tokens += [
            token for token,freq in self.token_freqs
            if freq >= min_freq and token not in self.uniq_tokens
        ]

        self.idx_to_token, self.token_to_idx = [], dict()

        for token in self.uniq_tokens:
            self.idx_to_token.append(token)
            self.token_to_idx[token] = len(self.idx_to_token) - 1


    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            # print('not isinstance (list, tuple)')
            # print(f'self.token_to_idx.get(tokens, self.unk): {self.token_to_idx.get(tokens, self.unk)}')
            return self.token_to_idx.get(tokens, self.unk)

        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]


def load_corpus_tm(txtpath, mode, max_token = -1, ):
    lines = read_text(txtpath)
    tokens = tokenizer(lines, mode)
    # print(f'test load: {tokens[:10]}')
    vocab = Vocab(tokens)
    if mode == 'char':
        corpus = [vocab[token] for line in lines for token in line]
    elif mode == 'word':
        # corpus = [vocab[token] for line in lines for token in line]
        corpus = [vocab[token] for line in tokens for token in line]

    if max_token > 0:
        corpus = corpus[:max_token]
    return corpus, vocab

if __name__ == '__main__':
    textpath = './timemachine.txt'

    newlines = read_text(textpath)
    # print(newlines)
    # print(type(newlines))
    print(len(newlines))

    tokens1 = tokenizer(newlines, 'word')
    tokens2 = tokenizer(newlines, 'char')

    # for i in range(11):
    #     # print(tokens1[i])
    #     print(tokens2[i])
    '''
        [['the', 'time', 'machine', 'by', 'h', 'g', 'wells'], [], [],
    '''
    # print(tokens1)
    # print(tokens1)

    tokenss = [token for line in tokens1 for token in line]

    print(count_corpus(tokens1))
    print(collections.Counter(tokenss))
    print(sorted(collections.Counter(tokenss).items()))
    # sorted(counter.items(), key=lambda x: x[1], reverse=True)
    print(sorted(collections.Counter(tokenss).items(), key=lambda  x: x[1], reverse=True))


    '''
        test vocav
    '''
    print('======')
    voc = Vocab(tokens1[0])
    print(voc.token_freqs)
    print(voc.idx_to_token)
    print(voc.uniq_tokens)
    print(voc.token_to_idx)

    # print(tokens2)


