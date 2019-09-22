#mapto3d.py
#an algorithm for importing and mapping 2D images to 3D plots
import numpy as np
import matplotlib.pyplot as plt
import pickle
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.image as mpimg
from PIL import Image
from scipy import ndimage

# converting image to code; in command line:

imgname = input("Enter image name: ")

# pixel dimensions (px x py)
px = int(input("Enter x size: "))
py = int(input("Enter y size: "))
imgsize = (px, py)


# Resize and invert image by 180 degrees and save
img = Image.open(imgname).convert('LA')
img = img.resize(imgsize, Image.ANTIALIAS)
img = img.rotate(180)
img.save("Transformed_" + imgname[:imgname.rfind('.')] + '.png')

img1 = mpimg.imread("Transformed_" + imgname[:imgname.rfind('.')] + '.png')
f1 = plt.figure(1)
rotated_img = ndimage.rotate(img, 180)[:, :, 0]
plt.imshow(rotated_img, 'gray')

pickle.dump(img1, open("pickledpic.p", "wb"))

# loading encrypted image
img1 = pickle.load(open("pickledpic.p", "rb"), encoding = 'latin1')
# converting image to binary
lum_img1 = img1[:, :, 0]

'''mapping algorithm: maps 2D binary image to 3D form by transforming relative
pixel color to depth'''

layers_maxval = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
layer_colors = ['magenta', 'yellow', 'black', 'cyan', 'pink', 'magenta', 'blue', 'navy', 'black']
n_layers = len(layers_maxval)

X = []
Y = []
Z = []

fig = plt.figure(2)
ax = plt.axes(projection = '3d') 

for layer_index in range(n_layers):
    x = []
    y = []
    z = []
    for yi in range(py):
        for xi in range(px):
            if lum_img1[yi, xi] < layers_maxval[layer_index]:
                x.append(xi)
                y.append(yi)
    z = (layer_index+1) * np.ones(np.size(x))
    
    X.append(x)
    Y.append(y)
    Z.append(z)

    ax.plot(z, x, y, 'o', color = layer_colors[layer_index])

plt.xlabel('')
plt.ylabel('')
plt.suptitle('CV Project')
plt.show()



'''
x=[]
y=[]

for i in range(py):
    for j in range(px):
        if lum_img1[i,j]<0.90:
            x.append(j)
            y.append(i)
z=1*np.ones(np.size(x))

x2=[]
y2=[]
          
for i in range(py):
    for j in range(px):
        if lum_img1[i,j]<0.80:
            x2.append(j)
            y2.append(i)
z2=2*np.ones(np.size(x2))

x3=[]
y3=[]

for i in range(py):
    for j in range(px):
        if lum_img1[i,j]<0.70:
            x3.append(j)
            y3.append(i)
z3=3*np.ones(np.size(x3))
            
x4=[]
y4=[]
            
for i in range(py):
    for j in range(px):
        if lum_img1[i,j]<0.60:
            x4.append(j)
            y4.append(i)
z4=4*np.ones(np.size(x4))

x5=[]
y5=[]
     
for i in range(py):
    for j in range(px):
        if lum_img1[i,j]<0.50:
            x5.append(j)
            y5.append(i)
z5=5*np.ones(np.size(x5))

x6=[]
y6=[]       
            
for i in range(py):
    for j in range(px):
        if lum_img1[i,j]<0.40:
            x6.append(j)
            y6.append(i)
z6=6*np.ones(np.size(x6))

x7=[]
y7=[]       
            
for i in range(py):
    for j in range(px):
        if lum_img1[i,j]<0.30:
            x7.append(j)
            y7.append(i)
z7=7*np.ones(np.size(x7))

x8=[]
y8=[]       
            
for i in range(py):
    for j in range(px):
        if lum_img1[i,j]<0.20:
            x8.append(j)
            y8.append(i)
z8=8*np.ones(np.size(x8))

x9=[]
y9=[]       
            
for i in range(py):
    for j in range(px):
        if lum_img1[i,j]<0.10:
            x9.append(j)
            y9.append(i)
z9=9*np.ones(np.size(x9))
     

fig = plt.figure(2)
ax = plt.axes(projection='3d') 
ax.plot(z,x,y,'o',color='magenta')
ax.plot(z2, x2, y2,'o',color='yellow')
ax.plot(z3, x3, y3,'o',color='black')
ax.plot(z4, x4, y4,'o',color='cyan')
ax.plot(z5,x5, y5,'o',color='pink')
ax.plot(z6,x6, y6,'o',color='magenta')
ax.plot(z7,x7, y7,'o',color='blue')
ax.plot(z8,x8, y8,'o',color='navy')
ax.plot(z9,x9, y9,'o',color='black')
plt.xlabel('')
plt.ylabel('')
plt.suptitle('CV Project')
plt.show()
'''