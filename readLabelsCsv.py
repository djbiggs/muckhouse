"""
Read in the training with labels CSV and return a dictionary with all coordinates.
"""
import csv


PATH_CSV = './competition_files/train_with_labels_v3.csv'


def parseCsvImgRow(csvRow):
    result = dict()
    numCol = 0

    # Filename
    result['fileName'] = csvRow[numCol]
    numCol += 1

    # Parse the drill core rows
    drillCores = []
    for numCoreRow in range(11):

        # Parse the corners for each core
        corners = []
        for numCorner in range(4):

            # Parse a single corner
            corner = dict()
            corner['x'] = int(csvRow[numCol]) if (int(csvRow[numCol]) > 0) else None
            numCol += 1
            corner['y'] = int(csvRow[numCol]) if (int(csvRow[numCol]) > 0) else None
            numCol += 1

            corners.append(corner)

        drillCores.append(dict(corners = corners))

    result['drillCores'] = drillCores

    # Parse the types
    imgType = None
    for i in range(6):
        imgType = i if (int(csvRow[numCol]) > 0) else imgType
        numCol += 1

    result['imgType'] = imgType

    # Image url
    result['imgLocation'] = csvRow[numCol]
    numCol += 1

    # Image mask url objects
    result['imgMask'] = csvRow[numCol]

    return result


def readCsv():
    result = dict()

    with open(PATH_CSV, newline='') as f:
        reader = csv.reader(f)

        for numRow, row in enumerate(reader):
            # Skip the header row
            if numRow > 0:
                result[str(row[0])] = parseCsvImgRow(row)

    return result


if __name__ == '__main__':
    print(readCsv())
