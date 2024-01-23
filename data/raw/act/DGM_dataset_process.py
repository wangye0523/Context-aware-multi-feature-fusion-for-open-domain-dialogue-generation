import json
f_w_data = open('daily_dialog/test/dialogues.txt', 'w', encoding='utf-8')
f_w_action = open('daily_dialog/test/dialogues_act.txt', 'w', encoding='utf-8')
f_w_emotion = open('daily_dialog/test/dialogues_emotion.txt', 'w', encoding='utf-8')
with open('commissive_test.txt','r',encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
        line = json.loads(line)
        f_w_data.write(" ".join(line["post"]) + " __eou__ " + " ".join(line["response"])+" __eou__\n")
        f_w_action.write(line["post_label_act"][0] + " " + line["response_label_act"][0] + "\n")
        f_w_emotion.write(line["post_label_emotion"][0] + " " + line["response_label_emotion"][0] + "\n")
with open('directive_test.txt','r',encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
        line = json.loads(line)
        f_w_data.write(" ".join(line["post"]) + " __eou__ " + " ".join(line["response"])+" __eou__\n")
        f_w_action.write(line["post_label_act"][0] + " " + line["response_label_act"][0] + "\n")
        f_w_emotion.write(line["post_label_emotion"][0] + " " + line["response_label_emotion"][0] + "\n")
with open('inform_test.txt','r',encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
        line = json.loads(line)
        f_w_data.write(" ".join(line["post"]) + " __eou__ " + " ".join(line["response"])+" __eou__\n")
        f_w_action.write(line["post_label_act"][0] + " " + line["response_label_act"][0]+ "\n")
        f_w_emotion.write(line["post_label_emotion"][0] + " " + line["response_label_emotion"][0]+ "\n")
with open('question_test.txt','r',encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
        line = json.loads(line)
        f_w_data.write(" ".join(line["post"]) + " __eou__ " + " ".join(line["response"])+" __eou__\n")
        f_w_action.write(line["post_label_act"][0] + " " + line["response_label_act"][0]+ "\n")
        f_w_emotion.write(line["post_label_emotion"][0] + " " + line["response_label_emotion"][0]+ "\n")

