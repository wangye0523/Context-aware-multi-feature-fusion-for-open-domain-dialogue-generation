import json
with open('datas.txt','r',encoding='utf-8') as f:
    lines = f.readlines()
inform = []
question = []
directive = []
commissive = []
for line in lines:
    line = json.loads(line)
    if line['post_label_act'][0] == '1':
        line['post_label_act'][0] = '0'
        inform.append(line)
    elif line['post_label_act'][0] == '2':
        line['post_label_act'][0] = '1'
        question.append(line)
    elif line['post_label_act'][0] == '3':
        line['post_label_act'][0] = '2'
        directive.append(line)
    else:
        line['post_label_act'][0] = '3'
        commissive.append(line)
act_trainset_classify = inform[:26754]+question[:20313]+directive[:11108] + commissive[:4720]
with open('act/act_trainset_classify.txt', 'w', encoding='utf-8') as f:
    for data in act_trainset_classify:
        f.write(json.dumps(data, ensure_ascii=False))
        f.write('\n')
act_validset_classify = inform[26754:32487] + question[20313:24666] + directive[11108:13488] + commissive[4720:5731]
with open('act/act_validset_classify.txt', 'w', encoding='utf-8') as f:
    for data in act_validset_classify:
        f.write(json.dumps(data, ensure_ascii=False))
        f.write('\n')
act_testset_classify = inform[32487:] + question[24666:] + directive[13488:] + commissive[5731:]
with open('act/act_testset_classify.txt', 'w', encoding='utf-8') as f:
    for data in act_testset_classify:
        f.write(json.dumps(data, ensure_ascii=False))
        f.write('\n')

with open('act/inform_train.txt', 'w', encoding='utf-8') as f:
    for data in inform[:26754]:
        f.write(json.dumps(data, ensure_ascii=False))
        f.write('\n')
with open('act/inform_valid.txt', 'w', encoding='utf-8') as f:
    for data in inform[26754:32487]:
        f.write(json.dumps(data, ensure_ascii=False))
        f.write('\n')
with open('act/inform_test.txt', 'w', encoding='utf-8') as f:
    for data in inform[32487:]:
        f.write(json.dumps(data, ensure_ascii=False))
        f.write('\n')

with open('act/question_train.txt', 'w', encoding='utf-8') as f:
    for data in question[:20313]:
        f.write(json.dumps(data, ensure_ascii=False))
        f.write('\n')
with open('act/question_valid.txt', 'w', encoding='utf-8') as f:
    for data in question[20313:24666]:
        f.write(json.dumps(data, ensure_ascii=False))
        f.write('\n')
with open('act/question_test.txt', 'w', encoding='utf-8') as f:
    for data in question[24666:]:
        f.write(json.dumps(data, ensure_ascii=False))
        f.write('\n')

with open('act/directive_train.txt', 'w', encoding='utf-8') as f:
    for data in directive[:11108]:
        f.write(json.dumps(data, ensure_ascii=False))
        f.write('\n')
with open('act/directive_valid.txt', 'w', encoding='utf-8') as f:
    for data in directive[11108:13488]:
        f.write(json.dumps(data, ensure_ascii=False))
        f.write('\n')
with open('act/directive_test.txt', 'w', encoding='utf-8') as f:
    for data in directive[13488:]:
        f.write(json.dumps(data, ensure_ascii=False))
        f.write('\n')

with open('act/commissive_train.txt', 'w', encoding='utf-8') as f:
    for data in commissive[:4720]:
        f.write(json.dumps(data, ensure_ascii=False))
        f.write('\n')
with open('act/commissive_valid.txt', 'w', encoding='utf-8') as f:
    for data in commissive[4720:5731]:
        f.write(json.dumps(data, ensure_ascii=False))
        f.write('\n')
with open('act/commissive_test.txt', 'w', encoding='utf-8') as f:
    for data in commissive[5731:]:
        f.write(json.dumps(data, ensure_ascii=False))
        f.write('\n')


