import json
with open('datas.txt','r',encoding='utf-8') as f:
    lines = f.readlines()
no_emotion = []
positive = []
negative = []
for line in lines:
    line = json.loads(line)
    if line['post_label_emotion'][0] == '0':
        no_emotion.append(line)
    elif line['post_label_emotion'][0] in ['4', '6']:
        line['post_label_emotion'][0] = '1'
        positive.append(line)
    else:
        line['post_label_emotion'][0] = '2'
        negative.append(line)
emotion_trainset_classify = no_emotion[:10000]+positive[:10000]+negative[:1800]
with open('emotion/emotion_trainset_classify.txt', 'w', encoding='utf-8') as f:
    for data in emotion_trainset_classify:
        f.write(json.dumps(data, ensure_ascii=False))
        f.write('\n')
emotion_validset_classify = no_emotion[20000:22004] + positive[10000:10496] + negative[1800:2000]
with open('emotion/emotion_validset_classify.txt', 'w', encoding='utf-8') as f:
    for data in emotion_validset_classify:
        f.write(json.dumps(data, ensure_ascii=False))
        f.write('\n')
emotion_testset_classify = no_emotion[22004:24000] + positive[10496:] + negative[2000:]
with open('emotion/emotion_testset_classify.txt', 'w', encoding='utf-8') as f:
    for data in emotion_testset_classify:
        f.write(json.dumps(data, ensure_ascii=False))
        f.write('\n')
with open('emotion/no_emotion_train.txt', 'w', encoding='utf-8') as f:
    for data in no_emotion[:10000]:
        f.write(json.dumps(data, ensure_ascii=False))
        f.write('\n')
with open('emotion/no_emotion_valid.txt', 'w', encoding='utf-8') as f:
    for data in no_emotion[20000:22004]:
        f.write(json.dumps(data, ensure_ascii=False))
        f.write('\n')
with open('emotion/no_emotion_test.txt', 'w', encoding='utf-8') as f:
    for data in no_emotion[22004:24000]:
        f.write(json.dumps(data, ensure_ascii=False))
        f.write('\n')

with open('emotion/positive_train.txt', 'w', encoding='utf-8') as f:
    for data in positive[:10000]:
        f.write(json.dumps(data, ensure_ascii=False))
        f.write('\n')
with open('emotion/positive_valid.txt', 'w', encoding='utf-8') as f:
    for data in positive[10000:10496]:
        f.write(json.dumps(data, ensure_ascii=False))
        f.write('\n')
with open('emotion/positive_test.txt', 'w', encoding='utf-8') as f:
    for data in positive[10496:]:
        f.write(json.dumps(data, ensure_ascii=False))
        f.write('\n')

with open('emotion/negative_train.txt', 'w', encoding='utf-8') as f:
    for data in negative[:1800]:
        f.write(json.dumps(data, ensure_ascii=False))
        f.write('\n')
with open('emotion/negative_valid.txt', 'w', encoding='utf-8') as f:
    for data in negative[1800:2000]:
        f.write(json.dumps(data, ensure_ascii=False))
        f.write('\n')
with open('emotion/negative_test.txt', 'w', encoding='utf-8') as f:
    for data in negative[2000:]:
        f.write(json.dumps(data, ensure_ascii=False))
        f.write('\n')
