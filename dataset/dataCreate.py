import csv

from itertools import product

#Create headers
header =[]
for i in range(1, 11):
  for j in range((ord('A')), (ord('E'))):
    header.append('Q'+str(i)+chr(j))
header.append("Classification")

#Return Classification
def classify_score(score):
    if score >= 45:
        return "Aggressive Investor"
    elif score >= 39:
        return "Moderately Aggressive Investor"
    elif score >= 33:
        return "Moderate Investor"
    elif score >= 27:
        return "Moderately Conservative Investor"
    elif score >= 20:
        return "Conservative Investor"
    else:
        return "Ultra Conservative Investor"

#Write to CSV File
filename = "Questionaire.csv"
with open(filename, 'w') as csvfile:
   csvwriter = csv.writer(csvfile)
   csvwriter.writerow(header)

   #Create Product of options
   optionSelectedProduct = list(product(range(0,4), repeat=10))

   #Weightage of each
   scoreCal = [5, 4, 3, 2]

   #Create subsequent rows for CSV
   rowOfOption = []
   for optionSelect in optionSelectedProduct :
       score = 0
       optionRow = [0 for col in range(0, 41)]
       for i in range(0, len(optionSelect)):
           optionRow[4*i + optionSelect[i]] = 1
           score = score + scoreCal[optionSelect[i]]
       optionRow[40] = classify_score(score)
       rowOfOption.append(optionRow)

   csvwriter.writerows(rowOfOption)










