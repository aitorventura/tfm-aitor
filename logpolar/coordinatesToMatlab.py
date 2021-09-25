import json
import numpy as np
from scipy import io

f = open('data.json', )
data = json.loads(f.read())
images_to_evaluate = "1001#1012#1018#1026#1036#1057#1067#1098#1102#1104#1131#1163#1274#1278#1299#1375#1385#1409" \
                     "#1499#1501#1663"

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

    io.savemat("coordinatesToMatlab/{0}.mat".format(img), {"eyedat": FrameStack})
