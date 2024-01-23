import json
import random
positive_emotion = ["surprised", "excited", "proud", "grateful", "impressed","hopeful","confident","anticipating","joyful","nostalgic","prepared","content","caring","trusting"]
negative_emotion = ["angry", "sad", "annoyed", "lonely", "afraid", "terrified","guilty","disgusted","furious","anxious","disappointed","jealous","devastated","embarrassed","sentimental","ashamed","apprehensive","faithful"]
lines = []
positive, negative = [], []
positive_train, negative_tain = [], []
positive_valid, negative_valid = [], []
positive_test, negative_test = [], []
with open("trainset.txt", 'r', encoding="utf-8") as f:
    lines += f.readlines()
with open("validset.txt", 'r', encoding="utf-8") as f:
    lines += f.readlines()
with open("testset.txt", 'r', encoding="utf-8") as f:
    lines += f.readlines()
print(len(lines))
for line in lines:
    line = json.loads(line)
    if line["response_label_act"][0] in positive_emotion:
        line["response_label_emotion"] = ["0"]
        line["post_label_emotion"] = ["0"]
        positive.append(line)
    elif line["response_label_act"][0] in negative_emotion:
        line["post_label_emotion"] = ["1"]
        line["response_label_emotion"] = ["1"]
        negative.append(line)
print(len(positive))
print(len(negative))
random.shuffle(positive)
random.shuffle(negative)
positive_train = positive[:int(36072*0.8)]
positive_valid = positive[int(36072*0.8):int(36072*0.9)]
positive_test = positive[int(36072*0.9):]
negative_train = negative[:int(42955*0.8)]
negative_valid = negative[int(42955*0.8):int(42955*0.9)]
negative_test = negative[int(42955*0.9):]
with open("positive_train.txt", 'w', encoding="utf-8") as fw:
    print(len(positive_train))
    for data in positive_train:
        fw.write(json.dumps(data))
        fw.write("\n")
with open("positive_valid.txt", 'w', encoding="utf-8") as fw:
    print(len(positive_valid))
    for data in positive_valid:
        fw.write(json.dumps(data))
        fw.write("\n")
with open("positive_test.txt", 'w', encoding="utf-8") as fw:
    print(len(positive_test))
    for data in positive_test:
        fw.write(json.dumps(data))
        fw.write("\n")
with open("negative_train.txt", 'w', encoding="utf-8") as fw:
    print(len(negative_train))
    for data in negative_train:
        fw.write(json.dumps(data))
        fw.write("\n")
with open("negative_valid.txt", 'w', encoding="utf-8") as fw:
    print(len(negative_valid))
    for data in negative_valid:
        fw.write(json.dumps(data))
        fw.write("\n")
with open("negative_test.txt", 'w', encoding="utf-8") as fw:
    print(len(negative_test))
    for data in negative_test:
        fw.write(json.dumps(data))
        fw.write("\n")
with open("all_test.txt", 'w', encoding="utf-8") as fw:
    print(len(negative_test))
    for data in positive_test:
        fw.write(json.dumps(data))
        fw.write("\n")
    for data in negative_test:
        fw.write(json.dumps(data))
        fw.write("\n")
with open("all_valid.txt", 'w', encoding="utf-8") as fw:
    print(len(negative_test))
    for data in positive_valid:
        fw.write(json.dumps(data))
        fw.write("\n")
    for data in negative_valid:
        fw.write(json.dumps(data))
        fw.write("\n")


