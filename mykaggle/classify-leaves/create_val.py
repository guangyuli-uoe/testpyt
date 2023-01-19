import os
import pandas as pd

val_path = './val.csv'

train_path = './newtrain.csv'
temp_path = './temp.csv'

train_df = pd.read_csv(train_path)
tmp_df = pd.read_csv(temp_path)

# print(train_df.head())
# # print(train_df['image'])
# print(type(train_df['image']))
# print(list(train_df['image']))

train_list = list(train_df['image'])
tmp_list = list(tmp_df['image'])

val_list = [v for v in tmp_list if v not in train_list]
print(val_list)
print(len(val_list))
# with open(val_path, 'w', encoding='utf-8') as fw:
#     with open(train_path, 'r', encoding='utf-8') as ftrain:
#         with open(temp_path, 'r', encoding='utf-8') as ftmp:
#
#             for line_tmp in ftmp.readlines():
#                 print(line_tmp)
#                 for line_train in ftrain.readlines():
#                     print(line_train)
#                     print(line_tmp == line_train)
#                     if line_tmp.strip() != line_train.strip():
#                         print('1')
#                         # print(line_tmp)
#                         fw.write(line_tmp)
#                         break
#                 break
#
#                 # ftrain.seek(0)

with open(val_path, 'w', encoding='utf-8') as fw:
    with open(temp_path, 'r', encoding='utf-8') as ftmp:
        count = 0
        for line in ftmp:
            if count == 0:
                fw.write(line)
                count += 1
            else:
                img = line.split(',')[0]
                # print(img)
                if img in val_list:
                    fw.write(line)