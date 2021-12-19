!apt-get -qq install -y libsm6 libxext6
!pip install -q -U opencv-python

import numpy as np
import cv2 
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import ndimage
from scipy.spatial import distance
from sklearn.cluster import KMeans
from scipy.spatial import distance_matrix
import pickle
import math
import matplotlib.pyplot as plt
import imutils

#Prendiamo dunque il nostro Drive e montiamolo come sottodirectory

from google.colab import drive
drive.mount('/gdrive')
%cd /gdrive

# Definiamo il percorso della Directory dalla quale vogliamo prendere le immagini per poi eseguire una SIFT

base_dir = '/gdrive/MyDrive/Colab Notebooks/img/'

# Carichiamo l'immagine d'esempio 
img = plt.imread(os.path.join(base_dir, 'casa1.jpg'))

# Mostriamo l'immagine. Da notare che non è necessario per il funzionamento della SIFT, ma serve per capire con quale immagine stiamo lavorando.
plt.figure(figsize=(6,6))
plt.imshow(img)
plt.show()

# Trasformiamo l'immagine in una scala di grigi (bianco e nero) che servirà dopo per applicare la sift all'oggetto.

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

plt.figure(figsize=(6,6))
plt.imshow(gray, cmap='gray', vmin=0, vmax=255)
plt.show()

#Questi sono i parametri di default (è come scrivere: sift = cv2.SIFT_create() ):
sift = cv2.SIFT_create(nfeatures = 0, nOctaveLayers = 3, contrastThreshold = 0.04, edgeThreshold = 10, sigma = 1.6)	

# Ecco una funzione che permette di stampare tutti i valori del keypoint selezionato

def print_KP_info(kp, i):
  print('Keypoint coordinates: '+ str(kp[i].pt)) 
  print('Keypoint size: '+ str(kp[i].size)) 
  print('Keypoint orientation: '+ str(kp[i].angle)) 
  print('Keypoint response: '+ str(kp[i].response)) 
  print('Keypoint octave: '+ str(kp[i].octave)) 
  print('Keypoint class_id: '+ str(kp[i].class_id)) 

kp = sift.detect(gray,None)
print('Detected KP: '+ str(len(kp)));

KP_i= 0
print_KP_info(kp, KP_i)

# Calcolo il keypoint descriptor
kp, des = sift.compute(gray, kp)

#Oppure così per trovarli e calcolarli direttamente
kp, des = sift.detectAndCompute(gray,None)

#Segna i keypoint sull'immagine
imgKP=cv2.drawKeypoints(gray,kp,img,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

plt.figure(figsize=(30,10))
plt.imshow(imgKP, cmap='gray', vmin=0, vmax=255)
plt.show()

# Per prima cosa definiamo quale feature extractor e quale feature matching utilizzare
feature_extractor = 'sift' # posso scegliere tra 'sift', 'surf', 'brisk', 'orb', etc...etc...
feature_matching = 'bf' # ad esempio bf (brute force), knn (k-nearest-neighbor), etc...etc...

#Carichiamo quindi 2 immagini da utilizzare per l'image stitching
trainImg = plt.imread(os.path.join(base_dir, 'casa1.jpg')) 
trainImg_gray = cv2.cvtColor(trainImg, cv2.COLOR_RGB2GRAY) #N.B: ho bisogno dell'immagine in scala di grigi

queryImg = plt.imread(os.path.join(base_dir, 'casa2.jpg'))
queryImg_gray = cv2.cvtColor(queryImg, cv2.COLOR_RGB2GRAY)

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, constrained_layout=False, figsize=(16,9))
ax1.imshow(queryImg, cmap="gray")
ax1.set_xlabel("Immagine di Query", fontsize=14)

ax2.imshow(trainImg, cmap="gray")
ax2.set_xlabel("Immagine da trasformare", fontsize=14)

plt.show()

def detectAndDescribe(image, method=None):
    
    assert method is not None, "Devi definire un metodo. Un metodo può essere: 'sift', 'surf', etc...etc..."
    
    # Trova ed estrae features dall'immagine
    if method == 'sift':
        descriptor = cv2.SIFT_create()
    elif method == 'surf':
        descriptor = cv2.xfeatures2d.SURF_create()
    elif method == 'brisk':
        descriptor = cv2.BRISK_create()
    elif method == 'orb':
        descriptor = cv2.ORB_create()
        
    # otteniamo i keypoints e i descriptors
    (kps, features) = descriptor.detectAndCompute(image, None)
    
    return (kps, features)

# Applichiamo la funzione "detectAndDescribe" alle nostre immagini in scala di grigio
kpsA, featuresA = detectAndDescribe(trainImg_gray, method=feature_extractor)
kpsB, featuresB = detectAndDescribe(queryImg_gray, method=feature_extractor)

# Mostriamo i keypoint e le features trovate su entrambe le immagini
fig, (ax1,ax2) = plt.subplots(nrows=1, ncols=2, figsize=(20,8), constrained_layout=False)
ax1.imshow(cv2.drawKeypoints(trainImg_gray,kpsA,None,color=(0,255,0)))
ax1.set_xlabel("(a)", fontsize=14)
ax2.imshow(cv2.drawKeypoints(queryImg_gray,kpsB,None,color=(0,255,0)))
ax2.set_xlabel("(b)", fontsize=14)

def createMatcher(method,crossCheck):
    #Restituisce un oggetto "matcher"
    
    if method == 'sift' or method == 'surf':
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=crossCheck)
    elif method == 'orb' or method == 'brisk':
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=crossCheck)
    return bf

def matchKeyPointsBF(featuresA, featuresB, method):
    bf = createMatcher(method, crossCheck=True)
        
    # Trova i matches migliori.
    best_matches = bf.match(featuresA,featuresB)
    
    # Ordina le feature in base alla distanza: I punti con distanza minore (ovvero con più similarità) vengono ordinati per primi nel vettore.
    rawMatches = sorted(best_matches, key = lambda x:x.distance)
    print("Raw matches (Brute force):", len(rawMatches))
    return rawMatches

# NELL'ESEMPIO RIPORTATO UTILIZZIAMO BRUTE FORCE, MA È BENE DEFINIRE ANCHE K-NEAREST-NEIGHBOR ALMENO PER CAPIRE IL SUO FUNZIONAMENTO

def matchKeyPointsKNN(featuresA, featuresB, ratio, method):
    bf = createMatcher(method, crossCheck=False)
    # Calcola i matches "grezzi" e inizializza una lista di matches attuali
    rawMatches = bf.knnMatch(featuresA, featuresB, 2)
    print("Raw matches (knn):", len(rawMatches))
    matches = []

    # Cicla all'interno dei matches grezzi
    for m,n in rawMatches:
        # Si assicura che la distanza per ognuno sia all'interno di un certo range
        if m.distance < n.distance * ratio:
            matches.append(m)
    return matches

print("Using: {} feature matcher".format(feature_matching))

fig = plt.figure(figsize=(20,8))

if feature_matching == 'bf':
    matches = matchKeyPointsBF(featuresA, featuresB, method=feature_extractor)
    img3 = cv2.drawMatches(trainImg,kpsA,queryImg,kpsB,matches[:100],
                           None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
elif feature_matching == 'knn':
    matches = matchKeyPointsKNN(featuresA, featuresB, ratio=0.75, method=feature_extractor)
    img3 = cv2.drawMatches(trainImg,kpsA,queryImg,kpsB,np.random.choice(matches,100),
                           None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
plt.imshow(img3)
plt.show()

def getHomography(kpsA, kpsB, featuresA, featuresB, matches, reprojThresh):
    # convertiamo i keypoint in un numpy array
    kpsA = np.float32([kp.pt for kp in kpsA])
    kpsB = np.float32([kp.pt for kp in kpsB])
    
    if len(matches) > 4:

        # costruisco 2 set di punti
        ptsA = np.float32([kpsA[m.queryIdx] for m in matches])
        ptsB = np.float32([kpsB[m.trainIdx] for m in matches])
        
        # Stimo l'omografia tra i due set di punti
        (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC,
            reprojThresh)

        return (matches, H, status)
    else:
        return None

M = getHomography(kpsA, kpsB, featuresA, featuresB, matches, reprojThresh=4)
if M is None:
    print("Error!")
(matches, H, status) = M
print(H)

# Applico una panorama correction, ovvero corregge il panorama per adattarlo all'immagine "stitchata"
width = trainImg.shape[1] + queryImg.shape[1]
height = trainImg.shape[0] + queryImg.shape[0]

result = cv2.warpPerspective(trainImg, H, (width, height))
result[0:queryImg.shape[0], 0:queryImg.shape[1]] = queryImg

# trasformol'immagine in una scaa di grigi e pongo una soglia 
gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]

# Trovo i contorni dall'immagine binaria
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

# Ottengo il massimo contorno dell'mmagine
c = max(cnts, key=cv2.contourArea)

# Inscatolo l'immagine in un bordo interno
(x, y, w, h) = cv2.boundingRect(c)

# Taglio l'immagine fino al bordo interno
result = result[y:y + h, x:x + w]

# mostro l'immagine ritagliata e corretta
plt.figure(figsize=(20,10))
plt.imshow(result)
