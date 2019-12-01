#######################################################################################
# Class for group for similar object images
# filenames: full path to images
# key: text presentation*
# id: numeric id of group*
# ----------------------------
# * only for debug and external visualization tools and export id
#######################################################################################

import numpy as np
from PIL import Image

class ObjectGroup:
  def __init__(self):
    self.filenames = []
    self.key = ""
    self.id = 0
    self.kid = None
    return

class ObjectDataset:
  def __init__(self,localdir,type):
    self.localdir = localdir
    self.groups = []
    self.type   = type
    self.total  = 0
    return

  # filename: path to metadat file
  def load(self):
    try:
      fp = open(self.getListFilename(), 'r')
      for cnt, line in enumerate(fp):
        ingroup = line.strip().split(",")
        group = ObjectGroup()

        for g in ingroup:
          splitted = g.split(":")
          if len(splitted) == 1:
            filename = g
            path = self.getImageFilename(filename);
            group.filenames.append(filename) # filename
            self.total = self.total + 1
          else:
            key = splitted[0]
            value = splitted[1]
            if key == 'K':   # Key
              group.key = value
            elif key == 'I': # Key Id
              group.id = int(value)

        self.groups.append(group)
    finally:
      fp.close()
    
  # valpart: (0-1) percent of val data
  def splitTrainVal(self,valpart):
    traincount = self.total * (1 - valpart)
    trainmode = True
    train = ObjectDataset(self.localdir,"train") 
    val = ObjectDataset(self.localdir,"val") 
    for g in self.groups:
       if trainmode:
         train.total += len(g.filenames)
         train.groups.append(g)

         if(train.total > traincount):
            trainmode = False

       else:
         val.total += len(g.filenames)
         val.groups.append(g)

    return (train, val)

  # get path to dataset metadata file
  # each group is a line in file: comma separated string: 00001.png,00002.png,00003.png
  # line also can contain tags instead files: K:国,I:12345,00001.png,00002.png,00003.png
  # 
  def getListFilename(self):
    return self.localdir+"/{}.txt".format(self.type)

  # get image path
  def getImageFilename(self,filename):
    return self.localdir+"/images/{}/{}".format(self.type,filename);

  # Generate pair of not same random id in range 0..range
  def generateRandomPair(self,range):
    id1 = np.random.randint(0,range);
    id2 = id1
    while (id2 == id1):
      id2 = np.random.randint(0,range);
    return [id1, id2]

  # Generate random id in range 0..range
  def generateRandomId(self,range):
    return np.random.randint(0,range);
    
  def loadImage(self,path):
    img = np.asarray(Image.open(path)).astype(float)
    return img

  # Generate couple with same chars
  def getCouple(self):
    gid = self.generateRandomId(len(self.groups))
    ids = self.generateRandomPair(len(self.groups[gid].filenames))
    img1 = self.loadImage(self.groups[gid].filenames[ids[0]])
    img2 = self.loadImage(self.groups[gid].filenames[ids[1]])
    return np.array([img1,img2])

  # Generate couple with different chars
  def getWrong(self):
    gids = self.generateRandomPair(len(self.groups))
    id0 = self.generateRandomId(len(self.groups[gids[0]].filenames))
    id1 = self.generateRandomId(len(self.groups[gids[1]].filenames))
    img1 = self.loadImage(self.groups[gids[0]].filenames[id0])
    img2 = self.loadImage(self.groups[gids[1]].filenames[id1])
    return np.array([img1,img2])

  def getKanji(self,groupid):
    if groupid < len(self.groups):
      return self.groups[groupid].kanji
    else:
      return None

  def getSingle(self,groupid,id):
    if ((groupid < len(self.groups)) and (id < len(self.groups[groupid].filenames))):
      return np.array([self.loadImage(self.groups[groupid].filenames[id])])[0]
    else:
      return None