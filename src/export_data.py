#!/usr/bin/env python3

from csv import writer


def __space_separated_list(l):
    l = [str(x) for x in l]
    return ' '.join(l)


def export_csv(data):
    with open('export.csv', 'w') as f:
        csvwriter = writer(f, delimiter=',')
        csvwriter.writerow(('playlist_id', 'track_ids'))
        for row in data:
            csvwriter.writerow((row[0], __space_separated_list(row[1])))