import csv

with open("datasets/1.csv", "r") as csv_file:
    csv_reader = csv.reader(csv_file)
    dataset = [row for row in csv_reader]
    print("The dataset is:")
    for row in dataset:
        print(row)
    print()
    general = ["?"]*(len(dataset[0]) - 1)
    specific = ["phi"]*(len(dataset[0]) - 1)
    for row in dataset:
        if row[-1] == "TRUE":
            for i in range(len(row) - 1):
                if specific[i] == row[i]:
                    continue
                elif specific[i] == "phi":
                    specific[i] = row[i]
                else:
                    for j in range(len(general)):
                        if isinstance(general[j], list):
                            if specific[i] == general[j][i] and specific[i] != "?":
                                del general[j]
                                break
                    specific[i] = "?"
        if row[-1] == "FALSE":
            temp_1 = []
            for i in range(len(row) - 1):
                if specific[i] != row[i] and specific[i] != "?":
                    temp_2 = ["?"] * (len(row) - 1)
                    temp_2[i] = specific[i]
                    temp_1.append(temp_2)
            general = temp_1
        print(specific)
        print(general)
    version_space= []
    for i in range(len(general)):
        for j in range(len(specific)):
            if general[i][j] == specific[j]:
                specific[j] = "?"
            else:
                temp_2 = list(general[i])
                temp_2[j] = specific[j]
                version_space.append(temp_2)
    print(version_space)


            
