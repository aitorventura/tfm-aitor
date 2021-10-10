import json
import numpy as np
from scipy import io


def create_mat_file(folder):
    f = open('{0}.json'.format(folder), )
    data = json.loads(f.read())
    images_to_evaluate = "1009#1026#1092#1093#1094#1097#1144#1159#1164#1278#1387#1408#1409#1414#1442#1471#1527#1572" \
                         "#1585#1617"

    images_to_evaluate = images_to_evaluate.split("#")

    diccionario = dict()
    for img in images_to_evaluate:
        diccionario[img] = []

    values = data["values"]
    for val in range(0, len(values)):
        images = values[val]["images"]
        for img in images_to_evaluate:
            for d in range(len(images)):
                if (images[d]["id"]) == img:
                    array_x = []
                    array_y = []
                    for coordinate in images[d]["coordinates"]:
                        array_x.append(coordinate[0])
                        array_y.append(coordinate[1])

                    a = np.array([array_x, array_y])
                    a = a.astype(np.double)

                    L = diccionario[img]
                    L.append(a)
                    diccionario[img] = L

    for img in images_to_evaluate:
        L = diccionario[img]
        FrameStack = np.empty((len(L),), dtype=object)
        for i in range(len(L)):
            FrameStack[i] = L[i]

        io.savemat("coordinatesToMatlab/{1}/{0}.mat".format(img, folder), {"eyedat": FrameStack})


if __name__ == "__main__":
    create_mat_file('high-quality')
    create_mat_file('low-quality')
