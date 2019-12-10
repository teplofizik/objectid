from PIL import Image, ImageDraw, ImageFont, ImageOps
import numpy as np
from matplotlib import pyplot as plt

class Fonts:
  def __init__(self,folder,size):
    self.fonts = ['851MkPOP_002.ttf', 'AsobiMemogaki-Regular-1-01.ttf', 'Hosohuwafont.ttf', 'umeboshi_.ttf']
    self.loaded = []
    for f in self.fonts:
      self.loaded.append(ImageFont.truetype(folder + '/' + f, size))

  def GetRandomFont(self):
    id = np.random.randint(0,len(self.loaded))
    return self.loaded[id]
    
fonts = Fonts('fonts', 88)

def showImage(img):
  plt.imshow(np.divide(img.reshape(96,96,4),255), interpolation='nearest')

def AddRotatedText(image,text,font,angle, color):
  txt=Image.new('L', (100,100))

  d = ImageDraw.Draw(txt)
  d.text((0,0), text, font = font, fill=255)
  w=txt.rotate(angle, expand=1)

  image.paste( ImageOps.colorize(w, (0,0,0,0), color), (0,0), w)

def GenerateKanji(kanji,fonts):
  img = Image.new(mode='RGBA', size=(96,96), color=(255,255,255))
  fnt = fonts.GetRandomFont()
  
  AddRotatedText(img,kanji,fnt,np.random.randint(-100,100) / 10.0, (0,0,0));

  return np.asarray(img).astype(float).reshape((1,96,96,4))
