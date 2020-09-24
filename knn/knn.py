from numpy import asarray,array,mat
from numpy import linalg as LA
from PIL import Image
import cv2
import collections

d = dict()

#for image 1 to binary
image = cv2.imread('D:/ML-Practicles/knn/1(1).jpg', cv2.IMREAD_GRAYSCALE)

thresh, img_b = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
data = asarray(img_b)

for i in range(32):
	for j in range(32):
		if data[i][j] == 255:
			data[i][j] = 1

a = list()

for i in range(32):
	for j in range(32):
		a.append(data[i][j])

#for image 2 to binary
image = cv2.imread('D:/ML-Practicles/knn/1(2).jpg', cv2.IMREAD_GRAYSCALE)

thresh, img_b = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
data = asarray(img_b)

for i in range(32):
	for j in range(32):
		if data[i][j] == 255:
			data[i][j] = 1

for i in range(32):
	for j in range(32):
		a.append(data[i][j])

#for image 3 to binary
image = cv2.imread('D:/ML-Practicles/knn/1(3).jpg', cv2.IMREAD_GRAYSCALE)

thresh, img_b = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
data = asarray(img_b)

for i in range(32):
	for j in range(32):
		if data[i][j] == 255:
			data[i][j] = 1

for i in range(32):
	for j in range(32):
		a.append(data[i][j])


#for image 4 to binary
image = cv2.imread('D:/ML-Practicles/knn/1(4).jpg', cv2.IMREAD_GRAYSCALE)

thresh, img_b = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
data = asarray(img_b)

for i in range(32):
	for j in range(32):
		if data[i][j] == 255:
			data[i][j] = 1

for i in range(32):
	for j in range(32):
		a.append(data[i][j])


#for image 5 to binary
image = cv2.imread('D:/ML-Practicles/knn/1(5).jpg', cv2.IMREAD_GRAYSCALE)

thresh, img_b = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
data = asarray(img_b)

for i in range(32):
	for j in range(32):
		if data[i][j] == 255:
			data[i][j] = 1

for i in range(32):
	for j in range(32):
		a.append(data[i][j])

#for image 6 to binary
image = cv2.imread('D:/ML-Practicles/knn/1(6).jpg', cv2.IMREAD_GRAYSCALE)

thresh, img_b = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
data = asarray(img_b)

for i in range(32):
	for j in range(32):
		if data[i][j] == 255:
			data[i][j] = 1

for i in range(32):
	for j in range(32):
		a.append(data[i][j])

#for image 7 to binary
image = cv2.imread('D:/ML-Practicles/knn/1(7).jpg', cv2.IMREAD_GRAYSCALE)

thresh, img_b = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
data = asarray(img_b)

for i in range(32):
	for j in range(32):
		if data[i][j] == 255:
			data[i][j] = 1

for i in range(32):
	for j in range(32):
		a.append(data[i][j])

#for image 8 to binary
image = cv2.imread('D:/ML-Practicles/knn/1(8).jpg', cv2.IMREAD_GRAYSCALE)

thresh, img_b = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
data = asarray(img_b)

for i in range(32):
	for j in range(32):
		if data[i][j] == 255:
			data[i][j] = 1

for i in range(32):
	for j in range(32):
		a.append(data[i][j])

#for image 9 to binary
image = cv2.imread('D:/ML-Practicles/knn/1(9).jpg', cv2.IMREAD_GRAYSCALE)

thresh, img_b = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
data = asarray(img_b)

for i in range(32):
	for j in range(32):
		if data[i][j] == 255:
			data[i][j] = 1

for i in range(32):
	for j in range(32):
		a.append(data[i][j])

#for image 10 to binary
image = cv2.imread('D:/ML-Practicles/knn/1(10).jpg', cv2.IMREAD_GRAYSCALE)

thresh, img_b = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
data = asarray(img_b)

for i in range(32):
	for j in range(32):
		if data[i][j] == 255:
			data[i][j] = 1

for i in range(32):
	for j in range(32):
		a.append(data[i][j])

#for image 11 to binary
image = cv2.imread('D:/ML-Practicles/knn/0(0).jpg', cv2.IMREAD_GRAYSCALE)

thresh, img_b = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
data = asarray(img_b)

for i in range(32):
	for j in range(32):
		if data[i][j] == 255:
			data[i][j] = 1

for i in range(32):
	for j in range(32):
		a.append(data[i][j])

#for image 12 to binary
image = cv2.imread('D:/ML-Practicles/knn/0(1).jpg', cv2.IMREAD_GRAYSCALE)

thresh, img_b = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
data = asarray(img_b)

for i in range(32):
	for j in range(32):
		if data[i][j] == 255:
			data[i][j] = 1

for i in range(32):
	for j in range(32):
		a.append(data[i][j])

#for image 13 to binary
image = cv2.imread('D:/ML-Practicles/knn/0(2).jpg', cv2.IMREAD_GRAYSCALE)

thresh, img_b = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
data = asarray(img_b)

for i in range(32):
	for j in range(32):
		if data[i][j] == 255:
			data[i][j] = 1

for i in range(32):
	for j in range(32):
		a.append(data[i][j])

#for image 14 to binary
image = cv2.imread('D:/ML-Practicles/knn/0(3).jpg', cv2.IMREAD_GRAYSCALE)

thresh, img_b = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
data = asarray(img_b)

for i in range(32):
	for j in range(32):
		if data[i][j] == 255:
			data[i][j] = 1

for i in range(32):
	for j in range(32):
		a.append(data[i][j])

#for image 15 to binary
image = cv2.imread('D:/ML-Practicles/knn/0(4).jpg', cv2.IMREAD_GRAYSCALE)

thresh, img_b = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
data = asarray(img_b)

for i in range(32):
	for j in range(32):
		if data[i][j] == 255:
			data[i][j] = 1

for i in range(32):
	for j in range(32):
		a.append(data[i][j])

#for image 16 to binary
image = cv2.imread('D:/ML-Practicles/knn/0(5).jpg', cv2.IMREAD_GRAYSCALE)

thresh, img_b = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
data = asarray(img_b)

for i in range(32):
	for j in range(32):
		if data[i][j] == 255:
			data[i][j] = 1

for i in range(32):
	for j in range(32):
		a.append(data[i][j])

#for image 17 to binary
image = cv2.imread('D:/ML-Practicles/knn/0(6).jpg', cv2.IMREAD_GRAYSCALE)

thresh, img_b = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
data = asarray(img_b)

for i in range(32):
	for j in range(32):
		if data[i][j] == 255:
			data[i][j] = 1

for i in range(32):
	for j in range(32):
		a.append(data[i][j])

#for image 18 to binary
image = cv2.imread('D:/ML-Practicles/knn/0(7).jpg', cv2.IMREAD_GRAYSCALE)

thresh, img_b = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
data = asarray(img_b)

for i in range(32):
	for j in range(32):
		if data[i][j] == 255:
			data[i][j] = 1

for i in range(32):
	for j in range(32):
		a.append(data[i][j])

#for image 19 to binary
image = cv2.imread('D:/ML-Practicles/knn/0(8).jpg', cv2.IMREAD_GRAYSCALE)

thresh, img_b = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
data = asarray(img_b)

for i in range(32):
	for j in range(32):
		if data[i][j] == 255:
			data[i][j] = 1

for i in range(32):
	for j in range(32):
		a.append(data[i][j])

#for image 20 to binary
image = cv2.imread('D:/ML-Practicles/knn/0(9).jpg', cv2.IMREAD_GRAYSCALE)

thresh, img_b = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
data = asarray(img_b)

for i in range(32):
	for j in range(32):
		if data[i][j] == 255:
			data[i][j] = 1

for i in range(32):
	for j in range(32):
		a.append(data[i][j])


#image of digit for testing
image = cv2.imread('D:/ML-Practicles/knn/1(11).jpg', cv2.IMREAD_GRAYSCALE)

thresh, img_b = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
data = asarray(img_b)

for i in range(32):
	for j in range(32):
		if data[i][j] == 255:
			data[i][j] = 1

b = list()

for i in range(32):
	for j in range(32):
		b.append(data[i][j])

i=0
j=1024

length = len(a)
tr = array(a)
test = array(b)
c = list()

while j <= length:
	c.append(tr[i:j] - test[0:1024])
	#print(j)
	i += 1024
	j += 1024

result = mat(c)

target = [1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0]

i=0
while i < 20:
	d[LA.norm(c[i][:1024])] = target[i]
	i += 1

od = collections.OrderedDict(sorted(d.items()))

#select 4 nearest neighbour
k=4

#final list of nearest neighbour
final = list()

#append values which is nearest 
for key,v in od.items():
	if  k!= 0:
		final.append(v)
		k -= 1

#final that target which is occuring most of the time
final_dict = {}

for i in range(len(final)):
	label = final[i]
	if label not in final_dict.keys():
		final_dict[label] = 1
	else:
		final_dict[label] += 1

#sort dictionary by values
f = sorted(final_dict.items(), key = lambda x:x[1] ,reverse = True)

#print key which have more occurenece
for i in f:
	print("Your image represnt number: ", i[0])
	break

#open an image of number which you uploaded for testing
img = Image.open('D:/ML-Practicles/knn/1(11).jpg')
img.show()


#d[1873.8817999009436] = 1
#d[2746.4522570035692] = 1
#d[765.0934583434889] = 1
#d[2024.0271737306296] = 1
#d[3651.0498490160335] = 1
#d[2102.804080269962] = 1
#d[2193.6127734857855] = 1
#d[1196.0719041930547] = 1
#d[2102.8012269351566] = 1
#d[2445.8943149694755] = 1
#d[2962.8594296726264] = 0
#d[2472.350298804763] = 0
#d[2962.8589234048927] = 0
#d[3917.395308109714] = 0
#d[1632.8505749149247] = 0
#d[3006.4281132267242] = 0
#d[2896.2660444095945] = 0
#d[2951.863309843462] = 0
#d[3017.2253810413304] = 0
#d[2962.8514981348626] = 0






