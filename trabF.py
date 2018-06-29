import numpy as np
import imageio
from math import *
import math
import matplotlib
import matplotlib.pyplot as plt
import random
import time
import base64
import hashlib
from Crypto import Random
from Crypto.Cipher import AES
import sys

# Encriptador
class AESCipher(object):
    def __init__(self, key): 
        self.bs = 32
        self.key = hashlib.sha256(key.encode()).digest() # Padrao de 256

	# Metodo para encriptar
    def encrypt(self, raw):
        raw = self._pad(raw)
        iv = Random.new().read(AES.block_size)
        cipher = AES.new(self.key, AES.MODE_CBC, iv)
        return base64.b64encode(iv + cipher.encrypt(raw.encode('utf-8')))

	# Metodo para desencriptar
    def decrypt(self, enc):
        enc = base64.b64decode(enc)
        iv = enc[:AES.block_size]
        cipher = AES.new(self.key, AES.MODE_CBC, iv)
        return self._unpad(cipher.decrypt(enc[AES.block_size:])).decode('utf-8')

	# Formacoes de entrada
    def _pad(self, s):
        return s + (self.bs - len(s) % self.bs) * chr(self.bs - len(s) % self.bs)

    @staticmethod
    def _unpad(s):
        return s[:-ord(s[len(s)-1:])]

def constColorUnravelIndex(pxNum, imgShape):
	invIndex = np.unravel_index(pxNum, (imgShape[2], imgShape[0], imgShape[1]))
	return (invIndex[1], invIndex[2], invIndex[0])
	
# Convolui
def convolve(img, filter):
	f = np.zeros((img.shape[0], img.shape[1]))	
	f[:filter.shape[0],:filter.shape[1]] = filter
	
	FTFilter = np.fft.fft2(f)
	res = np.empty_like(img)
	for i in range(3):
		res[:,:,i] = np.real(np.fft.ifft2(np.multiply(np.fft.fft2(img[:,:,i]), FTFilter)))
	
	return res

# Cria o filtro com as regioes
def writeMap(image, threshold):
	edgeDetectionKernel = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])
	edgeImage = convolve(image, edgeDetectionKernel)
	
	blurKernel = np.ones((20,20))/400
	blurredEdges = convolve(edgeImage, blurKernel)
	
	blurrRad = blurredEdges.copy()
	blurrRad[blurrRad > 255 * threshold] = 255
	blurrRad[blurrRad <= 255 * threshold] = 0
	
	seqStart = 0
	for pxNum in range(blurrRad.shape[2] * blurrRad.shape[0] * blurrRad.shape[1]):
		index = constColorUnravelIndex(pxNum, blurrRad.shape)
		
		if blurrRad[index] == 255:
			blurrRad[index] = 0
		else:
			if pxNum-seqStart >= 9:
				for i in range(seqStart, pxNum):
					blurrRad[constColorUnravelIndex(i, blurrRad.shape)] = pxNum-i
			if (pxNum-seqStart)%2 == 0:
				blurrRad[constColorUnravelIndex(seqStart, blurrRad.shape)] = 0
			seqStart = pxNum
	
	return blurrRad

# Escreve no pixel encontrado
def usePixel(pxNum, writeSpots):
	if(writeSpots[constColorUnravelIndex(pxNum, writeSpots.shape)] != 0):
		currPx = pxNum
		while(writeSpots[constColorUnravelIndex(currPx, writeSpots.shape)] != 0):
			writeSpots[constColorUnravelIndex(currPx, writeSpots.shape)] = 0
			currPx += 1
		
		currPx = pxNum-1
		while(writeSpots[constColorUnravelIndex(currPx, writeSpots.shape)] != 0):
			writeSpots[constColorUnravelIndex(currPx, writeSpots.shape)] = 0
			currPx -= 1
			
# Esconder o texto
def hideText(text, image, key, threshold):
	resultImage = image.copy()

	imgSize = image.shape[0] * image.shape[1] * image.shape[2]
	random.seed(key)
	originPixels = []
	# Faz o mapa de onde escrever
	writeSpots = writeMap(image, threshold)
	for i in range(16):
		px = random.randint(0, imgSize)
		while(px in originPixels):
			px = random.randint(0, imgSize)
		originPixels.append(px)
		usePixel(px, writeSpots)
	
	firstAvailablePixel = -1
	for currPx in range(writeSpots.shape[0] * writeSpots.shape[1] * writeSpots.shape[2]):
		if(writeSpots[constColorUnravelIndex(currPx, image.shape)] != 0):
			firstAvailablePixel = currPx
			break
	
	firstAvailablePixelEncode = firstAvailablePixel + random.randint(0, 2**32)
	firstAvailablePixelEncode %= 2**32
	
	for i in range(len(originPixels)):
		pxPos = constColorUnravelIndex(originPixels[i], image.shape)
		resultImage[pxPos] &= 0xfc
		resultImage[pxPos] |= (firstAvailablePixelEncode >> 2*i) & 3
	
	textPos = 0
	jumpRand = 0
	jumpAmount = firstAvailablePixel
	lastEnd = -1
	currPx = firstAvailablePixel
	while currPx < (writeSpots.shape[0] * writeSpots.shape[1] * writeSpots.shape[2]):
		val = writeSpots[constColorUnravelIndex(currPx, image.shape)]
		
		if(val != 0):
			if lastEnd != -1:
				jumpAmount = currPx - lastEnd
				jumpAmount += jumpRand
				jumpAmount %= 2**20
				
				for i in range(5):
					resultImage[constColorUnravelIndex(lastEnd+i, image.shape)] &= 0xf0
					resultImage[constColorUnravelIndex(lastEnd+i, image.shape)] |= ((jumpAmount >> 4*i) & 0x0f)
					
				lastEnd = -1
		
			resultImage[constColorUnravelIndex(currPx, image.shape)] &= 0xf0
			resultImage[constColorUnravelIndex(currPx+1, image.shape)] &= 0xf0
			if val >= 9:
				charVal = ord(text[int(textPos)])
				charVal += random.randint(0, 2**8)
				charVal %= 2**8
				resultImage[constColorUnravelIndex(currPx, image.shape)] |= (charVal & 0x0f)
				resultImage[constColorUnravelIndex(currPx+1, image.shape)] |= ((charVal >> 4) & 0x0f)
				textPos += 1
				currPx += 2
				if(textPos >= len(text)):
					break
			elif val == 7:
				charVal = ord('|')
				charVal += random.randint(0, 2**8)
				charVal %= 2**8
				resultImage[constColorUnravelIndex(currPx, image.shape)] |= (charVal & 0x0f)
				resultImage[constColorUnravelIndex(currPx+1, image.shape)] |= ((charVal >> 4) & 0x0f)
				currPx += 2
			elif val == 5:
				jumpRand = random.randint(0, 2**20)
				for i in range(5):
					resultImage[constColorUnravelIndex(currPx+i, image.shape)] &= 0xf0
				lastEnd = currPx
				currPx += 5
		else:
			currPx += 1
	
	if(writeSpots[constColorUnravelIndex(currPx, image.shape)] >= 7):
		charVal = ord('|')
		charVal += random.randint(0, 2**8)
		charVal %= 2**8
		resultImage[constColorUnravelIndex(currPx, image.shape)] &= 0xf0
		resultImage[constColorUnravelIndex(currPx+1, image.shape)] &= 0xf0
		
		resultImage[constColorUnravelIndex(currPx, image.shape)] |= (charVal & 0x0f)
		resultImage[constColorUnravelIndex(currPx+1, image.shape)] |= ((charVal >> 4) & 0x0f)
		
		currPx += 2
	
	if(writeSpots[constColorUnravelIndex(currPx, image.shape)] >= 5):
		jumpAmount = random.randint(0, 2**20)
		jumpAmount %= 2**20
		for i in range(5):
			resultImage[constColorUnravelIndex(currPx+i, image.shape)] &= 0xf0
			resultImage[constColorUnravelIndex(currPx+i, image.shape)] |= ((jumpAmount >> 4*i) & 0x0f)
	
	# Verifica se o texto nao coube na imagem
	if(textPos < len(text)):
		print("Esse texto não cabe nessa imagem com a densidade atual")
		print("Caracteres suportados pela imagem com a densidade atual:", textPos)
		print("Caracteres no seu texto:", len(text))
	
	return resultImage, writeSpots

# Estima a capacidade de armazenamento da img
def getCapacity(image, threshold):
	writeSpots = writeMap(image, threshold)
	cap = 0
	justOut = False
	for currPx in range(writeSpots.shape[0] * writeSpots.shape[1] * writeSpots.shape[2]):
		if(writeSpots[constColorUnravelIndex(currPx, image.shape)] != 0):
			cap += 1
			justOut = True
		elif justOut:
			justOut = False
			cap -= 7
	return int((cap*0.9)/2)
		
# Recupera texto dentro da img
def getText(image, key):
	imgSize = image.shape[0] * image.shape[1] * image.shape[2]
	random.seed(key)
	originPixels = []
	for i in range(16):
		px = random.randint(0, imgSize)
		while(px in originPixels):
			px = random.randint(0, imgSize)
		originPixels.append(px)
	
	firstAvailablePixel = 0
	for i in range(len(originPixels)):
		pxPos = constColorUnravelIndex(originPixels[i], image.shape)
		firstAvailablePixel |= (image[pxPos] & 0x03) << 2*i
	
	firstAvailablePixel -= random.randint(0, 2**32)
	firstAvailablePixel %= 2**32
	
	resultString = ""
	jumpAmount = -1
	currPx = firstAvailablePixel
	while jumpAmount != 0:
		jumpAmount = 0
		currChar = 0
		while(chr(currChar) != '|'):
			currChar = 0	
			currChar |= image[constColorUnravelIndex(currPx, image.shape)] & 0x0f
			currChar |= ((image[constColorUnravelIndex(currPx + 1, image.shape)] & 0x0f) << 4)
			
			currChar -= random.randint(0, 2**8)
			currChar %= 2**8
			
			if(chr(currChar) != '|'):
				resultString += chr(currChar)
			currPx += 2
		
		for i in range(5):
			jumpAmount |= ((image[constColorUnravelIndex(currPx+i, image.shape)] & 0xf) << (4*i))
		
		jumpAmount -= random.randint(0, 2**20)
		jumpAmount %= 2**20
		currPx += jumpAmount
	
	return resultString

# Compara as imgs e retorna diferenca
def compara(imgReferencia, imgCriada):
    return (np.sqrt(((imgReferencia - imgCriada) ** 2).mean()))

# Le a entrada e carrega as imagens
while True:
	print('1) Ler texto codificado\n2) Codificar mensagem\n', end='')
	option = input()
	if option != '1' and option != '2':
		print("Opção inválida!")
		continue
	break

# Tenta abrir a imagem
while True:
	try:
		print('Imagem: ', end='')
		image = imageio.imread(input())
	except OSError:
		print('Imagem não encontrada')
		continue
	break

# Pega chave pra decodificar, garante que seja utf-8
while True:
	try:
		print('Chave de codificação: ', end='')
		key = input()
		bytes(key, 'ascii')
	except UnicodeEncodeError:
		print('Chave não é do tipo utf-8')
		continue
	break

# Cria o encriptador com a key 
encriptador = AESCipher(key)

if(option == "1"):
	try:
		text = bytes(getText(image, key), "utf-8")
		text = encriptador.decrypt(text)
		print('Texto recuperado: ' + text)
	except:
		print('Não foi possivel recuperar texto dessa imagem')
else:
	print('Densidade do texto (entre 0 e 1): ', end='')
	threshold = 1-float(input())
	
	# Pega o texto e garante que seja utf-8
	while True:
		try:
			print('Texto a codificar: ', end='')
			text = input()
			bytes(text, 'ascii')
		except UnicodeEncodeError:
			print("Texto nao é do tipo UTF-8")
			continue
		break

	print()	# Pula linha

	textE = encriptador.encrypt(text).decode("utf-8")
	hiddenImage, writeSpots = hideText(textE, image, key, threshold)
	imageio.imwrite('result.png', hiddenImage)

	print('Imagem resultante salva em \"result.png\"')
	print('Texto codificado com sucesso!')
	print('RMSE: %lf' % compara(image, hiddenImage)); 	# Exibe o erro 
	# Mostra comparacoes
	plt.subplot(311).set_title('Imagem Original')
	plt.imshow(image)
	plt.subplot(312).set_title('Imagem Criada')
	plt.imshow(hiddenImage)
	plt.subplot(313).set_title('Regiões que podiam ser sobrescritas')
	plt.imshow(writeSpots)
	plt.tight_layout()
	plt.show()