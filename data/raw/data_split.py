import json
with open('trainset.txt', 'r', encoding='utf_8') as f:
    lines = f.readlines()
train_sets = []
for line in lines:
    train_sets.append(json.loads(line.replace('\n', '')))
train0,train1, train2, train3, train4,train5,train6 = [], [], [], [],[],[],[]
for train_data in train_sets:
    if train_data['response_label_emotion'][0] == '0':
        train0.append(train_data)
    elif train_data['response_label_emotion'][0]== '1':
        train1.append(train_data)
    elif train_data['response_label_emotion'][0]== '2':
        train2.append(train_data)
    elif train_data['response_label_emotion'][0]== '3':
        train3.append(train_data)
    elif train_data['response_label_emotion'][0]== '4':
        train4.append(train_data)
    elif train_data['response_label_emotion'][0]== '5':
        train5.append(train_data)
    elif train_data['response_label_emotion'][0]== '6':
        train6.append(train_data)
print(len(train_sets))
print('******')
print(len(train0))
print(len(train1))
print(len(train2))
print(len(train3))
print(len(train4))
print(len(train5))
print(len(train6))

min_train_label_num = 100000
if min_train_label_num > len(train1):
    min_train_label_num = len(train1)
if min_train_label_num > len(train2):
    min_train_label_num = len(train2)
if min_train_label_num > len(train3):
    min_train_label_num = len(train3)
if min_train_label_num > len(train4):
    min_train_label_num = len(train4)
if min_train_label_num > len(train5):
    min_train_label_num = len(train5)
if min_train_label_num > len(train6):
    min_train_label_num = len(train6)
if min_train_label_num > len(train0):
    min_train_label_num = len(train0)
print(min_train_label_num)
# train_new_sets = train0[:min_train_label_num] + train1[:min_train_label_num] + train2[:min_train_label_num] \
#                  + train3[:min_train_label_num] + train4[:min_train_label_num] + train5[:min_train_label_num]\
#                  + train6[:min_train_label_num]
train_new_sets = train0[:800] + train1[:] + train2[:] \
                 + train3[:] + train4[:800] + train5[:]\
                 + train6[:800]
print(len(train_new_sets))
with open('train_newset_emotion.txt','w',encoding='utf-8') as f:
    for data in train_new_sets:
        f.write(json.dumps(data, ensure_ascii=False))
        f.write('\n')
