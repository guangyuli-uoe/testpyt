import os
import pandas as pd

rootpath = './images/'

traincsv = './train.csv'
# newtraincsv = './newtrain.csv'
newtraincsv = './temp.csv'

class_num = 10
# img_num = 100
img_num = 118


def create_train(traincsv, newtrain, classnum, img_num):
    dataframe = pd.read_csv(traincsv)
    # print(dataframe.head(5))
    # print(dataframe.describe())
    count = 0

    # with open(traincsv, 'r', encoding='utf-8') as fr:
    #     with open(newtrain, 'w', encoding='utf-8') as fw:
    #         for line_fr in fr:
    #             if count = 0:

    classlist = set()
    classdict = {}

    with open(newtrain, 'w', encoding='utf-8') as fw:
        with open(traincsv, 'r', encoding='utf-8') as fr:
            # lines = fr.readlines()
            # fw.write('')
            for line in fr:
                count += 1

                # fw.write(line)
                if count >= 2:
                    # print(line)
                    line_list = line.strip('\n').split(',')
                    # print(line_list)

                    img = line_list[0]
                    label = line_list[1]

                    if len(classlist) >= classnum:
                        if sum(classdict.values()) >= classnum*img_num:
                            break

                    if label not in classlist:
                        if len(classlist) < classnum:
                            classlist.add(label)

                    if label not in classdict:
                        if len(classdict) < classnum:
                            classdict[label] = 1
                            fw.write(line)
                    else:

                        if classdict[label] <= img_num-1:
                            # next(fr)
                            fw.write(line)
                            classdict[label] += 1

                        # classdict[label] += 1
                        # if classdict[label] > 2:

                else:
                    fw.write(line)

                # if count >= 10:
                #     break

            print('===')
            print(classdict)
            print(classlist)
            print(len(classdict))
            print(len(classlist))
            # print(sum(classdict.values()))


    # df = pd.read_csv(newtrain, header=0, names=['image', 'label'])
    # # df.columns = ['image', 'label']
    # print(df.head())

create_train(traincsv, newtrain=newtraincsv, classnum=class_num, img_num=img_num)
