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
global test

def pulsar():
    print("introduce una tecla")
    input()

def show_image(image, array_images, R, S):
    global pos_x
    global pos_y
    global coordinates
    global test

    img = dict()
    coordinates = []

    imfile = 'images800x600/{0}.jpg'.format(image)
    im_pil = Image.open(imfile)
    im = np.array(im_pil)

    M, N, C = im.shape

    rho0 = 5.0
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

    while time.time() - initial_time < 10:
        GLP = GenLogPolar(M, N, pos_x, pos_y, rho0, R, S, rhoMax)

        lp_im = np.zeros((R, S, C))
        for c in range(C):
            lp_im[:, :, c] = GLP.lp(im[:, :, c])

        c_im = np.zeros_like(im)
        for c in range(C):
            c_im[:, :, c] = GLP.ilp(lp_im[:, :, c])

        c_im = cv2.cvtColor(c_im, cv2.COLOR_BGR2RGB)

        cv2.imshow('image', c_im / 255.0)

        if cv2.waitKey(15) & 0xFF == 27:
            break
        key = cv2.waitKey(20)  # pauses for 3 seconds before fetching next image
        if key == 27:  # if ESC is pressed, exit loop
            cv2.destroyAllWindows()
            break

    if not test:
        img["id"] = i
        img['coordinates'] = coordinates

        array_images.append(img)


def move(event, x, y, flags, param):
    # grab references to the global variables
    global pos_x
    global pos_y
    global coordinates
    global test

    # mouse is moving
    if event == cv2.EVENT_MOUSEMOVE:
        # change variable mouse to true in order to start applying the mask
        pos_x = x
        pos_y = y
        if not test:
            coordinates.append([pos_x, pos_y])


if __name__ == "__main__":
    # test_generalLP()
    #   23/3/21

    images_to_test = "1001#1012"

    images_to_test = images_to_test.split("#")

    images_to_evaluate = "1018#1026"

    images_to_evaluate = images_to_evaluate.split("#")

    # BAJA CALIDAD
    test = True
    for j in images_to_test:
        show_image(j, None, 40, 50)
    cv2.destroyAllWindows()
    pulsar()

    f = open('low-quality.json', )
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

    test = False
    for i in images_to_evaluate:
        show_image(i, array_images, 40, 50)
    cv2.destroyAllWindows()

    persona['id'] = id_persona
    persona['images'] = array_images

    values.append(persona)

    data['values'] = values
    pulsar()

    with open('low-quality.json', 'w') as my_file:
        json.dump(data, my_file)

    # ALTA CALIDAD
    test = True
    for j in images_to_test:
        show_image(j, None, 140, 175)
    cv2.destroyAllWindows()
    pulsar()

    f = open('high-quality.json', )
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

    test=False
    for i in images_to_evaluate:
        show_image(i, array_images, 140, 175)
    cv2.destroyAllWindows()

    persona['id'] = id_persona
    persona['images'] = array_images

    values.append(persona)

    data['values'] = values

    with open('high-quality.json', 'w') as my_file:
        json.dump(data, my_file)