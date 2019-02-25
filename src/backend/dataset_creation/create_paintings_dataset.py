import requests
import json


response = requests.get('https://collectionapi.metmuseum.org/public/collection/v1/search?q=portraits')

search_result_json = response.json()
result_object_IDs = search_result_json["objectIDs"]

image_id = 0

for object_id in result_object_IDs:
    response = requests.get('https://collectionapi.metmuseum.org/public/collection/v1/objects/' + str(object_id))
    object_json = response.json()
    artwork_type = object_json['medium']

    if 'canvas' in artwork_type or 'paper' in artwork_type or\
            'oil' in artwork_type or 'paint' in artwork_type:
        image_url = object_json['primaryImageSmall']
        response = requests.get(image_url)
        print(image_url)
        print(response.status_code)
        image_data = response.content
        print(image_data)

        with open('image' + str(image_id) + '.jpg', 'wb') as handler:
            handler.write(image_data)

        image_id += 1

    break