#!/usr/bin/env python3


from csv import reader


def load_csv(filename):
    with open(filename) as f:
        csvreader = reader(f, delimiter=',')
        return [(int(row[0]), int(row[1])) for row in csvreader if csvreader.line_num > 1]     # line_num is used to skip the initial column headers
