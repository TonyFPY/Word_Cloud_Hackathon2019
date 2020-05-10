from google.cloud import vision
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
    





import jieba
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import imageio
#对来自外部文件文本进行分词，得到string
f = open('out.rtf')
txt = f.read()
txtlist = jieba.lcut(txt)
string = " ".join(txtlist)

# 将string变量传入w的generate()方法，给词云输入文字

w = wordcloud.WordCloud()
w.generate(string)
w.to_file('output1.png')
