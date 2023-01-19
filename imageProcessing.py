

########## Source Code UAS Pengolaha ##########

# Source Code - STT NF
import numpy as np
import cv2 as cv

img = cv.imread('img/fruits.jpeg',1)
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
cv.imshow('grayscale',gray)
img = cv.imread('img/fruits.jpeg',1)
cv.imshow('image',img)
hsv =cv.cvtColor(img,cv.COLOR_BGR2GRAY)
cv.imshow('HSV',hsv)
lab = cv.cvtColor(img,cv.COLOR_BGR2LAB)
cv.imshow('LAB',lab)
cv.waitKey(0)

# Source Code Inversi - STT NF
import numpy as np
import cv2 as cv

citra = cv.imread('img/fruits.jpeg',0)
cv.imshow('Hasil Inversi',citra)
hasil = citra + 50
cv.imshow('Hasil Inversi',hasil)
cv.waitKey(0)
cv.destroyAllWindows()

# Source Code Histogram 1 - STT NF
import numpy as np
import cv2 as cv
import matplotlib
import matplotlib.pyplot as plt

citra = cv.imread('img/fruits.jpeg')
histo = cv.calcHist([citra], [0], None,[256], [0, 256])
plt.plot(histo)
plt.show()

# Source Code Histogram 2 - STT NF
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('img/fruits.jpeg',0)
hist,bins = np.histogram(img.flatten(),256,[0,256])
cdf = hist.cumsum()
cdf_normalized = cdf * hist.max()/cdf.max()
plt.plot(cdf_normalized, color ='b')
plt.hist(img.flatten(),256,[0,256],
color = 'r')
plt.xlim([0,256])
plt.legend(('cdf','histogram'), loc= 'upper left')
plt.show()

# Source Code Histogram 3 - STT NF
from matplotlib import pyplot
from os.path import basename
from os.path import splitext
from PIL import Image

def tampilkan_histogram(r, g, b, gambar):
    intensitas = list(range(256))
    lebar_bar = 0.3

    intensitas = [i-lebar_bar for i in intensitas]
    pyplot.bar(intensitas, r, width=lebar_bar, color='r')

    intensitas = [i+lebar_bar for i in intensitas]
    pyplot.bar(intensitas, g, width=lebar_bar, color='g')

    intensitas = [i+lebar_bar for i in intensitas]
    pyplot.bar(intensitas, b, width=lebar_bar, color='b')

    pyplot.title('Histogram ' + gambar)
    pyplot.xlabel('Intensitas')
    pyplot.ylabel('Kemunculan')
    pyplot.legend(['R', 'G', 'B'])
    pyplot.show()

def histogram(gambar):
    GAMBAR = Image.open(gambar)
    PIXEL = GAMBAR.load()

    ukuran_horizontal = GAMBAR.size[0]
    ukuran_vertikal = GAMBAR.size[1]

    gambar_r = Image.new('RGB', (ukuran_horizontal, ukuran_vertikal))
    pixel_r = gambar_r.load()

    gambar_g = Image.new('RGB', (ukuran_horizontal, ukuran_vertikal))
    pixel_g = gambar_g.load()

    gambar_b = Image.new('RGB', (ukuran_horizontal, ukuran_vertikal))
    pixel_b = gambar_b.load()

    r = [0] * 256
    g = [0] * 256
    b = [0] * 256

    for x in range(ukuran_horizontal):
        for y in range(ukuran_vertikal):
            intensitas_r = PIXEL[x, y][0]
            intensitas_g = PIXEL[x, y][1]
            intensitas_b = PIXEL[x, y][2]
            r[intensitas_r] += 1
            g[intensitas_g] += 1
            b[intensitas_b] += 1
            pixel_r[x, y] = (intensitas_r, 0, 0)
            pixel_g[x, y] = (0, intensitas_g, 0)
            pixel_b[x, y] = (0, 0, intensitas_b)

    tampilkan_histogram(r, g, b, gambar)

histogram('img/fruits.jpeg')

# Source Code Equalisasi - STT NF
import cv2
import numpy as np

citra = cv2.imread('img/fruits.jpeg',0)
ekual = cv2.equalizeHist(citra)
hasil = np.hstack((citra, ekual))
cv2.imshow('Contoh Hasil Equalisasi',hasil)
cv2.waitKey(0)                 
cv2.destroyAllWindows()

# Source Code Citra Kuantisasi - STT NF
import numpy as np
import cv2
from matplotlib import pyplot as plt


def show_img_with_matplotlib(color_img, title, pos):

    img_RGB = color_img[:, :, ::-1]

    ax = plt.subplot(2, 3, pos)
    plt.imshow(img_RGB)
    plt.title(title)
    plt.axis('off')


def color_quantization(image, k):

    data = np.float32(image).reshape((-1, 3))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    ret, label, center = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    center = np.uint8(center)
    result = center[label.flatten()]
    result = result.reshape(img.shape)
    return result

fig = plt.figure(figsize=(16, 8))
plt.suptitle("Kuantisasi warna menggunakan algoritma pengelompokan K-means", fontsize=14, fontweight='bold')
fig.patch.set_facecolor('silver')

img = cv2.imread("img/fruits.jpeg")

color_3 = color_quantization(img, 3)
color_5 = color_quantization(img, 5)
color_10 = color_quantization(img, 10)
color_20 = color_quantization(img, 20)
color_40 = color_quantization(img, 40)

show_img_with_matplotlib(img, "citra original", 1)
show_img_with_matplotlib(color_3, "citra kuantisasi (k = 3)", 2)
show_img_with_matplotlib(color_5, "citra kuantisasi (k = 5)", 3)
show_img_with_matplotlib(color_10, "citra kuantisasi (k = 10)", 4)
show_img_with_matplotlib(color_20, "citra kuantisasi (k = 20)", 5)
show_img_with_matplotlib(color_40, "citra kuantisasi (k = 40)", 6)

plt.show()