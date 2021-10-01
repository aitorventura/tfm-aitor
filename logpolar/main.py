import json
from json import JSONDecodeError

from PIL import Image
import cv2
import numpy as np
import time

from logpolar.logPolarFunction import GenLogPolar

global pos_x
global pos_y
global coordinates


def move(event, x, y, flags, param):
    # grab references to the global variables
    global pos_x
    global pos_y
    global coordinates
    # mouse is moving
    if event == cv2.EVENT_MOUSEMOVE:
        # change variable mouse to true in order to start applying the mask
        pos_x = x
        pos_y = y
        coordinates.append([pos_x, pos_y])


if __name__ == "__main__":
    # test_generalLP()
    #   23/3/21
    bColor = True
    if bColor:
        imfile = 'images800x600/1001Mouse.jpg'
    else:
        imfile = 'lena.pgm'

    print("¿Cómo te llamas?")
    name = input()

    f = open('test.json', )

    images_to_evaluate = "1001#1012#1018#1026#1036#1057#1067#1098#1102#1104#1131#1163#1274#1278#1299#1375#1385#1409" \
                         "#1499#1501#1663"

    images_to_evaluate = images_to_evaluate.split("#")

    try:
        data = json.loads(f.read())
        values = data['values']
    except JSONDecodeError:
        data = dict()
        values = []

    persona = dict()
    id_persona = len(values) + 1

    images = dict()
    array_images = []

    for i in images_to_evaluate:
        print(array_images)
        img = dict()
        coordinates = []

        imfile = 'images800x600/{0}.jpg'.format(i)
        print(imfile)
        im_pil = Image.open(imfile)  # .convert('L')
        im = np.array(im_pil)

        if bColor:
            M, N, C = im.shape
        else:
            M, N = im.shape
            C = 1

        rho0 = 5.0
        R, S = 80, 60

        #rho0 = 4.0
        #R, S = 140, 90
        rhoMax = min(M, N) / 2

        pos_x = 0
        pos_y = 0

        im_inicial = Image.open('imgInicial.jpg')  # .convert('L')
        im_inicial = np.array(im_inicial)

        cv2.imshow('image', im_inicial / 255.0)

        cv2.setMouseCallback('image', move)

        while True:
            if cv2.waitKey(15) & 0xFF == 27:
                break
            if 297 <= pos_y <= 303 and 397 <= pos_x <= 403:
                break

        initial_time = time.time()
        print(initial_time)

        while time.time() - initial_time < 10:
            GLP = GenLogPolar(M, N, pos_x, pos_y, rho0, R, S, rhoMax)

            if bColor:
                lp_im = np.zeros((R, S, C))
                for c in range(C):
                    lp_im[:, :, c] = GLP.lp(im[:, :, c])
            else:
                lp_im = GLP.lp(im)

            if bColor:
                c_im = np.zeros_like(im)
                for c in range(C):
                    c_im[:, :, c] = GLP.ilp(lp_im[:, :, c])

                c_im = cv2.cvtColor(c_im, cv2.COLOR_BGR2RGB)

            else:
                c_im = GLP.ilp(lp_im)

            cv2.imshow('image', c_im / 255.0)

            if cv2.waitKey(15) & 0xFF == 27:
                break
            key = cv2.waitKey(20)  # pauses for 3 seconds before fetching next image
            if key == 27:  # if ESC is pressed, exit loop
                cv2.destroyAllWindows()
                break

        img["id"] = i
        img['coordinates'] = coordinates

        array_images.append(img)

    persona['id'] = id_persona
    persona['name'] = name
    persona['images'] = array_images

    values.append(persona)

    data['values'] = values

    with open('test.json', 'w') as my_file:
        json.dump(data, my_file)

    f = open('test.json', )
    dict = json.loads(f.read())
    print(dict)
