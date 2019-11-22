#!/usr/bin/env python3

from csv import reader, writer


def load_csv(filename):
    with open(filename) as f:
        csvreader = reader(f, delimiter=',')
        return [row for row in csvreader if csvreader.line_num > 1]     # line_num is used to skip the initial column headers


def __space_separated_list(l):
    l = [str(x) for x in l]
    return ' '.join(l)


def export_csv(header, data):
    for row in data:
        assert len(header) == len(row)
    with open('export.csv', 'w') as f:
        csvwriter = writer(f, delimiter=',')
        csvwriter.writerow(header)
        for row in data:
            csvwriter.writerow((row[0], __space_separated_list(row[1])))
