# import json
# with open("dialogues_text.txt", 'r', encoding='utf-8') as dialogue_text:
#     dialogues_text = dialogue_text.readlines()
# with open("dialogues_act.txt", 'r', encoding='utf-8') as dialogue_act:
#     dialogues_act = dialogue_act.readlines()
# with open("dialogues_emotion.txt", 'r', encoding='utf-8') as dialogue_emotion:
#     dialogues_emotion = dialogue_emotion.readlines()
# f_data = open('datas.txt','w',encoding='utf-8')
# for i in range(0, len(dialogues_act)):
#     texts = dialogues_text[i].split(' __eou__')[:-1]
#     acts = dialogues_act[i].split(' ')[:-1]
#     emotions = dialogues_emotion[i].split(' ')[:-1]
#     if len(texts) == len(acts) and len(texts) == len(emotions):
#         for j in range(1, len(texts)):
#             dataset = {'post': texts[j - 1].split(' '), 'response': texts[j].split(' '),
#                        'post_label_act': [acts[j - 1]],
#                        'post_label_emotion': [emotions[j - 1]], 'response_label_act': [acts[j]],
#                        'response_label_emotion': [emotions[j]]}
#             f_data.write(json.dumps(dataset, ensure_ascii=False))
#             f_data.write('\n')
#
# f_data.close()

# with open('datas.txt', 'r', encoding='utf-8') as f:
#     lines = f.readlines()
# i=0
# with open('trainset.txt', 'w', encoding='utf-8') as f:
#     while i <= len(lines)*0.9:
#         print(i,len(lines))
#         f.write(lines[i])
#         i+=1
# with open('testset.txt', 'w', encoding='utf-8') as f:
#     while i > len(lines)*0.9 and i <= len(lines)*0.95:
#         f.write(lines[i])
#         i+=1
# with open('validset.txt', 'w', encoding='utf-8') as f:
#     while i > len(lines)*0.95 and i < len(lines):
#         f.write(lines[i])
#         i+=1

with open('dialogues_text.txt','r',encoding='utf-8') as f:
    lines = f.readlines()
vocab={}
for line in lines:
    texts = line.split(' __eou__')[:-1]
    for text in texts:
        for i in text.split(' '):
            if i not in vocab:
                vocab[i] = 0
            else:
                vocab[i] = vocab[i] + 1
with open('vocab.txt', 'w',encoding='utf-8') as f:
    for i in vocab:
        if vocab[i]>5:
            f.write(i+'\n')