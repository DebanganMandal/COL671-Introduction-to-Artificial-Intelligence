import json

with open('./data/phoneme_table.json') as f:
    phoneme = json.load(f)

vm = list(phoneme.values())
km = list(phoneme.keys())


with open('./data/vocabulary.json') as f:
    vocabulary = json.load(f)

sentence = "SHE FOLBED AMONG TE FLICKEWING SHADOWS A GRACEFUL AND HARMONIOUS IMAGE"
new_sentence = ""

def get_list(word, vm, km):
    list_new_words = [word]
    for ic,c in enumerate(word):
        for i, l in enumerate(vm):
            if c in l:
                # print(km[i], c)
                new_word = word[:ic] + km[i] + word[ic+1:]
                list_new_words.append(new_word)
    return list_new_words



def all_permute_word(w):


    list1 = get_list(w, vm, km)
    list2 = []

    for w in list1:
        list2 += get_list(w, vm, km)

    return list2

def get_sentences(sentence):
    sent_list = []
    words = sentence.split(' ')
    temp_words = sentence.split(' ')

    for i,w in enumerate(words):
        words_list = all_permute_word(w)

        for x in words_list:
            temp_words[i] = x
            new_sentence = ' '.join(temp_words)
            sent_list.append(new_sentence)

        temp_words[i] = words_list[0]

    return sent_list


print(get_sentences('FOLBED'))



# word = sentence.split(' ')[1]

# list_new_words_1 = []

# # for word in sentence.split(' '):
# list_new_words_1 += get_list(word, vm, km)

# list_new_words_2 = []

# for w in list_new_words_1:
#     list_new_words_2 += get_list(w, vm, km)

# print(all_permute_word(word))
# list_new_words_2 += list_new_words_1 

# print('FLICKERING' in list_new_words_2)


# for word in sentence.split(' '):
# changes = 0
# while changes<=2:
# for ic,c in enumerate(word):
#     for i, l in enumerate(vm):
#         if c in l:
#             # print(km[i], c)
#             new_word = word[:ic] + km[i] + word[ic+1:]
#             list_new_words.append(new_word)

