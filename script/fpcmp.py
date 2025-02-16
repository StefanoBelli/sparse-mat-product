import csv
import sys

def cmp_yvec_csv(file1, file2, tolerance=0.01):
    with open(file1, 'r') as f1, open(file2, 'r') as f2:
        reader1 = csv.reader(f1)
        reader2 = csv.reader(f2)

        next(reader1)
        next(reader2)

        neqs = []
        for line1, line2 in zip(reader1, reader2):
            for val1, val2 in zip(line1, line2):
                try:
                    num1 = float(val1)
                    num2 = float(val2)
                    if abs(num1 - num2) > tolerance:
                        neqs.append((line1, line2))
                        #break
                except ValueError:
                    continue

        for line1, line2 in neqs:
            print(f"neq {file1}: {line1} -- {file2}: {line2}")

cmp_yvec_csv(sys.argv[1], sys.argv[2])