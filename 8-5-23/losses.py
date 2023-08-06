import csv

csvfile = open("saved_I.csv", "r")
csvreader = csv.reader(csvfile)

for row in csvreader:
    print(row[1:10])
csvfile.close()
