# -*- coding: utf-8 -*-
import json
import datetime
import os
from pattern.text.en import positive, sentiment


# Task 1: Test Python Environment
def task1():
    print("hello world")


# Task 2: Define Object
def task2():
    return [1, 2, 3, 4, 5]


# Task 3: File Reading
def task3():
    f = open("task3.data")
    line = f.readline()
    words = line.split(' ')
    nums = [int(s) for s in words]
    items1 = nums[0: 5]
    items2 = nums[5: 10]
    return [items1, items2]


# Task 4: Data Structure
def task4_show_dict_functions(dic):
    print("Show three dictionary functions:")
    print("items(): " + str(type(dic.items())))
    for (k, v) in dic.items():
        print("{0} -> {1}".format(k, v))
    print("keys(): " + str(type(dic.keys())))
    for k in dic.keys():
        print(k)
    print("values(): " + str(type(dic.values())))
    for v in dic.values():
        print(v)


def task4_print_results(dic):
    for (k, v) in dic.items():
        print("{0}: {1}".format(k, v))


# Task 5: Data Serialization
def task5_save(dir, dic):
    json_obj = json.dumps(dic, indent=4)
    print(json_obj)
    with open(dir, 'w') as f:
        f.write(json_obj)


def task5_load(dir):
    with open(dir, 'r') as f:
        json_obj = f.read()
    dic = json.loads(json_obj)
    return dic


# Task 6: Data Serialization
def task6_save(dir, objs):
    if os.path.exists(dir):
        os.remove(dir)
    with open(dir, 'w') as f:# create an empty file
        pass
    for i in range(len(names)):# save obj to dir one by each time
        with open(dir, 'a') as f:
            f.write(json.dumps(objs[i], indent=4))
            f.write("\n")


def task6_load(dir):
    inp = []
    with open(dir, 'r') as f:
        line = f.readline()
        obj = ""
        while line:
            obj += line
            if line == "]\n" or line == "}\n":
                inp.append(json.loads(obj))
                obj = ""
            line = f.readline()
    if obj != "":
        inp.append(json.loads(obj))
    return inp


# Task 7: Data Preprocessing
def task7_radar(dir):
    with open(dir, 'r') as f:
        for i in range(3):
            line = f.readline()
            tweet = json.loads(line)
            print(tweet.keys())


def task7_printid(dir):
    ids = []
    with open(dir, 'r') as f:
        while True:
            line = f.readline()
            if not line:
                break
            tweet = json.loads(line)
            ids.append(tweet['id'])
    return ids


# Task 8: Data Preprocessing: tweets filtering
def task8(inp, outp, filter):
    with open(inp, 'r') as f:
        for line in f.readlines():
            tweet = json.loads(line)
            tweets.append(tweet)
    sorted_tweets = sorted(tweets, key=lambda
        item: datetime.datetime.strptime(item['created_at'], '%a %b %d %H:%M:%S +0000 %Y'))
    with open(outp, 'w')  as f:
        for i in range(filter):
            tweet = sorted_tweets[-i]
            f.write(json.dumps(tweet) + '\n')


# Task 9: File operations
def datetime2label(t):
    return "{0:02d}-{1:02d}-{2:02d}-{3:02d}".format(t.month, t.day, t.year, t.hour)


def task9_group(inp):
    tweet_dic = {}
    with open(inp, 'r') as f:
        for line in f.readlines():
            tweet = json.loads(line)
            time = datetime.datetime.strptime(tweet['created_at'], '%a %b %d %H:%M:%S +0000 %Y')
            label = datetime2label(time)
            if label in tweet_dic.keys():
                tweet_dic[label].append(tweet)
            else:
                tweet_dic[label] = [tweet]
    return tweet_dic


def task9_save(dic, folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
    for (label, tweets) in dic.items():
        file_name = folder + '/' + label
        with open(file_name, 'w') as f:
            for tweet in tweets:
                f.write(json.dumps(tweet) + '\n')


# Task 10: NLP and Sentiment Analysis
def task10(inp, pos, neg):
    pos_list, neg_list = [], []
    with open(inp, 'r') as f:
        for line in f.readlines():
            tweet = json.loads(line)
            print(sentiment(tweet['text']))
            if positive(tweet['text']):
                pos_list.append(tweet)
            else:
                neg_list.append(tweet)

    with open(pos, 'w') as f2:
        for tweet in pos_list:
            f2.write(json.dumps(tweet) + '\n')
    with open(neg, 'w') as f3:
        for tweet in neg_list:
            f3.write(json.dumps(tweet) + '\n')


if __name__ == "__main__":
    print("\nTask 1: Test Python Environment")
    task1()

    print("\nTask 2: Define Object")
    items = task2()
    print("items: " + str(items))

    print("\nTask 3: File Reading")
    print("items1: " + str(task3()[0]))
    print("items2: " + str(task3()[1]))

    print("\nTask 4: Data Structure")
    data = {
        "school": "UAlbany",
        "address": "1400 Washington Ave, Albany, NY 12222",
        "phone": "(518)442-3300"
    }
    task4_show_dict_functions(data)
    # For printing the results as shown, I choose to traverse items().
    # Luckily, the required results does not change the sequence.
    print("\nPrint the required results:")
    task4_print_results(data)

    print("\nTask 5: Data Serialization")
    # dict to json
    json_dir = "task5.json"
    task5_save(json_dir, data)
    print("Save dict to " + json_dir + ".")
    new_dic = task5_load(json_dir)
    print(str(type(new_dic)) + ", " + str(new_dic))
    print("Read dict from " + json_dir + ".")

    print("\nTask 6: Data Serialization")
    outp_dir = "task6.data"
    print(items)
    print(data)
    names = ["items", "data"]
    outp = [items, data]
    task6_save(outp_dir, outp)
    print("Save data structures to " + outp_dir + ".")
    inp = task6_load(outp_dir)
    for i in range(len(names)):
        print(names[i] + ": " + str(type(inp[i])) + ", " + str(inp[i]))
    print("Read data structures from " + outp_dir + ".")

    print("\nTask 7: Data Preprocessing")
    tweets_dir = "CrimeReport.txt"
    task7_radar(tweets_dir)
    # We can see the key for id should be 'id'
    id_list = task7_printid(tweets_dir)
    print("\n" + str(id_list))
    if len(id_list) >= 20:
        print("\nPresent with first 20 tweet ids:")
        for i in range(4):
            for j in range(5):
                index = 5 * i + j
                print(str(id_list[index]) + ", ", end='')
            print("")
        print("...")

    print("\nTask 8: Data Preprocessing: tweets filtering")
    tweets = []
    outp_dir = "task8.data"
    filter_width = 10
    task8(tweets_dir, outp_dir, filter_width)
    print("Save " + str(filter_width) + " most recently tweets in " + outp_dir + ".")

    print("\nTask 9: File operations")
    # For here, assume we do not need to sort tweets in each group
    outp_folder = "task9_output"
    tweet_dic = task9_group(tweets_dir)
    for (k, v) in tweet_dic.items():
        print(str(k) + ", " + str(v))
    task9_save(tweet_dic, outp_folder)

    print("\nTask 10: NLP and Sentiment Analysis")
    pos_dir = "positive-sentiment-tweets.txt"
    neg_dir = "negative-sentiment-tweets.txt"
    task10(tweets_dir, pos_dir, neg_dir)
    print("Classified into positive and negative by sentiment.")


