import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import imageio

import io
import os

# Imports the Google Cloud client library
from google.cloud import vision
from google.cloud.vision import types

image_list = ['gs://cloud-samples-data/vision/landmark/eiffel_tower.jpg',
              'gs://cloud-samples-data/vision/landmark/pofa.jpg',
              'gs://cloud-samples-data/vision/landmark/st_basils.jpeg'  ]

for i in range(1,4):
    image_uri='gs://cloud-samples-data/vision/web/city.jpg'

    client = vision.ImageAnnotatorClient()
    image=vision.types.Image()
    image.source.image_uri=image_uri

    response=client.label_detection(image=image)

    f=open("out.rtf","a")

    print('Labels (and confidence score):')
    print('='*79)
    for label in response.label_annotations:
        print(f'{label.description} ({label.score*100.:.2f}%)')
        f.write(label.description+" ")
    f.close()

# set the shape of the word cloud
shape = imageio.imread("shape.jpg")

# create STOOPWORDS list
stopwords = set(STOPWORDS)
stopwords.update(["in","a","the","in","of","by","within","was","is","are","it","and","as","Studies"])

# create a world cloud object
wc = WordCloud(
    width=200,
    height=100,
    background_color='white',
    font_path='Hiragino Sans GB.ttc',
    font_step=1,
    min_font_size=1,
    max_font_size=None,
    max_words=50,
    stopwords=stopwords,
    scale=20,
    prefer_horizontal=0.4,
    relative_scaling=0.7,
    mask=shape
)

# read the .txt file
f = open('out.rtf','r')
string = f.read()
wc.generate(string)

# extract color from an image
image_colors = ImageColorGenerator(shape)

# # display
# fig, axes = plt.subplots(1, 3)
# # raw word cloud picture on the left handside
# axes[0].imshow(wc)
# # new word cloud picture in the middle
# axes[1].imshow(wc.recolor(color_func=image_colors), interpolation="bilinear")
# # original image
# axes[2].imshow(shape, cmap=plt.cm.gray)
# for ax in axes:
#  ax.set_axis_off()

# plot the word cloud
plt.imshow(wc.recolor(color_func=image_colors), interpolation="bilinear")
plt.tight_layout(pad = 0)
plt.axis("off")
plt.show()
wc.to_file('output.png')
