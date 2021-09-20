import json
from json import JSONDecodeError

f = open('data.json',)

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
#Para cada imagen
for image in range (1, 5):
    #añado sus coordenadas
    img = dict()
    coordinates = []
    for i in range(0, 2):
        for j in range(0, 2):
            coordinates.append([i, j])
    img["id"] = image
    img['coordinates'] = coordinates

    array_images.append(img)

persona['id'] = id_persona
persona['imagenes'] = array_images


values.append(persona)

data['values'] = values

with open('data.json', 'w') as my_file:
    json.dump(data, my_file)

f = open('data.json',)
dict = json.loads(f.read())
print( dict)

