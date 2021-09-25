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

    f = open('data.json', )

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

    for i in range(1001, 1002):
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
        R, S = 90, 60
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
            if 237 <= pos_y <= 242 and 317 <= pos_x <= 322:
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

    with open('data.json', 'w') as my_file:
        json.dump(data, my_file)

    f = open('data.json', )
    dict = json.loads(f.read())
    print(dict)