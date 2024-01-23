import json
f_w_data = open('daily_dialog/validation/dialogues.txt', 'w', encoding='utf-8')
f_w_action = open('daily_dialog/validation/dialogues_act.txt', 'w', encoding='utf-8')
f_w_emotion = open('daily_dialog/validation/dialogues_emotion.txt', 'w', encoding='utf-8')
with open('negative_valid.txt','r',encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
        line = json.loads(line)
        f_w_data.write(" ".join(line["post"]) + " __eou__ " + " ".join(line["response"])+" __eou__\n")
        f_w_action.write(line["post_label_emotion"][0] + " " + line["response_label_emotion"][0] + "\n")
        f_w_emotion.write(line["post_label_emotion"][0] + " " + line["response_label_emotion"][0] + "\n")
with open('positive_valid.txt','r',encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
        line = json.loads(line)
        f_w_data.write(" ".join(line["post"]) + " __eou__ " + " ".join(line["response"])+" __eou__\n")
        f_w_action.write(line["post_label_emotion"][0] + " " + line["response_label_emotion"][0] + "\n")
        f_w_emotion.write(line["post_label_emotion"][0] + " " + line["response_label_emotion"][0] + "\n")