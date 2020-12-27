import csv

with open("datasets/1.csv", "r") as csv_file:
    csv_reader = csv.reader(csv_file)
    dataset = [row for row in csv_reader]
    print("The dataset is")
    for row in dataset:
        print(row)
    hypothesis = ["phi"]*(len(dataset[0]) - 1)
    print()
    print("The initial hypothesis is")
    print(hypothesis)
    print()
    count = 0
    for row in dataset:
        count = count + 1
        if row[-1] == "FALSE":
            print("Training instance: " + str(count))
            print("Rejected")
            continue
        for i in range(len(row) - 1):
            if hypothesis[i] == row[i]:
                continue
            elif hypothesis[i] == "phi":
                hypothesis[i] = row[i]
            else:
                hypothesis[i] = "?"
        print("Training instance: " + str(count))
        print(hypothesis)
