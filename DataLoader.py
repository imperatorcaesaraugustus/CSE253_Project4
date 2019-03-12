################################################################################
# CSE 253: Programming Assignment 4
# Feb 2019
################################################################################
    

def create_datasets():

    train_data = "data/train.txt"
    val_data = "data/val.txt"
    test_data = "data/test.txt"

    dict1, char_dictionary, reverse_char_dictionary = set(), {}, {}
    train_loader, val_loader, test_loader = [], [], []

    with open(train_data, 'r') as file:
        dict1 = set()
        while True:
            char = file.read(1)
            if not char: break
            dict1.add(char)

        dictionary_list = list(dict1)
        dictionary_list.sort()
        for i in range(len(dict1)):
            char_dictionary[dictionary_list[i]] = i
            reverse_char_dictionary[i] = dictionary_list[i]
        #print(char_dictionary, reverse_char_dictionary)

    with open(train_data, 'r') as file:   # all the chars in val.txt and test.txt are included in train.txt
        while True:
            char = file.read(1)
            if not char: break
            train_loader.append((char, char_dictionary[char]))
        #print(train_loader[:10])

    with open(val_data, 'r') as file:
        while True:
            char = file.read(1)
            if not char: break
            val_loader.append((char, char_dictionary[char]))

    with open(test_data, 'r') as file:
        while True:
            char = file.read(1)
            if not char: break
            test_loader.append((char, char_dictionary[char]))

    #print(len(train_loader),len(val_loader),len(test_loader))

    return train_loader, val_loader, test_loader, char_dictionary, reverse_char_dictionary   # return 3 lists of tuples and 2 char dictionaries

#create_datasets()



