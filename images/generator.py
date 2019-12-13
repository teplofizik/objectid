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
    self.scalerange = [0.8, 1.1]
    self.moverange = [[-3,4], [-3,4]]
    self.rotaterange = [-15, 15]
    self.size = (96, 96)
    
  def AddRotatedText(self,image,text,font,angle,pos,size,color):
    txt=Image.new('L', (100,100))

    d = ImageDraw.Draw(txt)
    d.text((0,0), text, font = font, fill=np.random.randint(200,255))
    if size is not None:
      txt=txt.resize(size)

    w=txt.rotate(angle, expand=1)
    image.paste( ImageOps.colorize(w, (0,0,0,0), color), pos, w)

  def getRotateAngle(self):
    if self.rotaterange[0] == self.rotaterange[1]:
      return self.rotaterange[0]
    else:
      return np.random.randint(self.rotaterange[0]*10,self.rotaterange[1]*10) / 10.0

  def getRandomOverlay(self):
    imarray = np.random.rand(96,96,4) * 255
    return Image.fromarray(imarray.astype('uint8'))
    
  def generate(self,kanji):
    img = Image.new(mode='RGBA', size=self.size, color=(np.random.randint(150,256),np.random.randint(150,256),np.random.randint(150,256)))
    overlay = self.getRandomOverlay()
    img = Image.blend(img, overlay, np.random.randint(25,55) / 100.0);
    # todo: freetype2: set_transform
    fnt = self.fonts.GetRandomFont()
    
    x = np.random.randint(self.moverange[0][0],self.moverange[0][1])
    y = np.random.randint(self.moverange[1][0],self.moverange[1][1])
    
    txto=Image.new('RGBA', size=self.size, color=(0,0,0,0))
    scalex = np.random.randint(int(100*self.scalerange[0]),int(100*self.scalerange[1]))
    scaley = np.random.randint(int(100*self.scalerange[0]),int(100*self.scalerange[1]))
    self.AddRotatedText(txto,kanji,fnt, self.getRotateAngle(),(x,y),(scalex,scaley), (0,0,0));
    img = Image.blend(img, txto, np.random.randint(70,100) / 100.0);
    
    return np.asarray(img).astype(float)