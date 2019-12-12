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
    

def showImage(img):
  plt.imshow(np.divide(img.reshape(96,96,4),255), interpolation='nearest')

class KanjiGenerator:
  def __init__(self):
    self.fonts = Fonts('fonts', 88)
    self.scalerange = [1, 1]
    self.rotaterange = [0, 0]
    self.size = (96, 96)
    
  def AddRotatedText(self,image,text,font,angle, color):
    txt=Image.new('L', (100,100))

    d = ImageDraw.Draw(txt)
    d.text((0,0), text, font = font, fill=255)
    w=txt.rotate(angle, expand=1)

    image.paste( ImageOps.colorize(w, (0,0,0,0), color), (0,0), w)

  def getRotateAngle(self):
    if self.rotaterange[0] == self.rotaterange[1]:
      return self.rotaterange[0]
    else:
      return np.random.randint(self.rotaterange[0]*10,self.rotaterange[1]*10) / 10.0

  def generate(self,kanji):
    img = Image.new(mode='RGBA', size=self.size, color=(255,255,255))
    fnt = self.fonts.GetRandomFont()
    
    
    self.AddRotatedText(img,kanji,fnt, self.getRotateAngle(), (0,0,0));

    return np.asarray(img).astype(float)