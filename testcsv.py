import csv

csvfile = open("saved.csv", 'r')
csvreader = csv.reader(csvfile)

for row in csvreader:
	print(row[2700])
