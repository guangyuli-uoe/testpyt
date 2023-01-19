import pandas as pd


datapath = './classify-leaves/train.csv'

dataframe = pd.read_csv(datapath)
print(dataframe.head(5))

print(dataframe.describe())

# print(dataframe.dtypes)

print('==========')

with open(datapath, 'r', encoding='utf-8') as fr:
    count = 0
    imglab = 0
    labels = {}
    for line in fr:
        count += 1
        if count >= 2:
            imglab += 1
            # print(line)
            line_list = line.strip('\n').split(',')
            # print(line_list)
            if line_list[-1] not in labels:
                labels[line_list[-1]] = 1
            else:
                labels[line_list[-1]] += 1


        # if count >= 5:
        #     # print(imglab)
        #     break
    print(imglab)
    # print(labels)
    print(len(labels))
