import json
import numpy as np
from scipy import io

f = open('data.json', )
data = json.loads(f.read())

diccionario = dict()
for img in range(1001, 1003):
    diccionario[img] = []

values = data["values"]
for val in range(0, len(values)):
    images = values[val]["images"]
    for img in range(1001, 1003):
        if (images[img - 1001]["id"]) == img:
            array_x = []
            array_y = []
            for coordinate in images[img - 1001]["coordinates"]:
                array_x.append(coordinate[0])
                array_y.append(coordinate[1])

            a = np.array([array_x, array_y])
            a = a.astype(np.double)

            L = diccionario[img]
            L.append(a)
            diccionario[img] = L


for img in range(1001, 1003):
    L = diccionario[img]
    FrameStack = np.empty((len(L),), dtype=object)
    for i in range(len(L)):
        FrameStack[i] = L[i]

    io.savemat("coordinatesToMatlab/{0}.mat".format(img), {"eyedat": FrameStack})