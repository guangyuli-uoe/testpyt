import os

valpath = './val.csv'
newtest = './newtest.csv'

with open(newtest, 'w', encoding='utf-8') as fw:
    with open(valpath, 'r', encoding='utf-8') as fr:
        count = 0
        for line in fr:
            # count += 1
            # if count == 1:
            #     fw.write(line)
            #
            # else:
            #     line_list = line.split(',')
            #
            #     fw.write(line_list[0] + '\n')
            line_list = line.split(',')
            fw.write(line_list[0] + '\n')


