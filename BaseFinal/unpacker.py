import csv

csvfile = open('saved.csv', 'r')
csvreader = csv.reader(csvfile)

c = 0
n = 0
l =  100
m = 133421

for row in csvreader: 
    if n > (m - 2000):
        if float(row[1]) < float(l):
            l = float(row[1])
            c = n
            print(l)
            print(c)
    n += 1
print("first")
v = 0
for row in csvreader:
    if v == c:
        print(row)
    v += 1

csvfile.close()
