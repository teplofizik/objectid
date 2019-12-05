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
from tensorflow.keras.utils import Sequence
import tensorflow as tf
import gc

class ObjectGroup:
  def __init__(self):
    self.filenames = []
    self.key = ""
    self.id = 0
    return

class ObjectDataset:
  def __init__(self,localdir,type):
    self.localdir = localdir
    self.groups = []
    self.type   = type
    self.total  = 0
    self.load()
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
            group.filenames.append(path) # filename
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
  
  def join(self,dataset):
    for ig in dataset.groups:
      localgroup = self.getGroupByKey(ig.key)
      if localgroup == None:
        #print("Object group '{}' not found, add all group".format(ig.key))
        self.groups.append(ig)
      else:
        #print("Add to group '{}' {} filenames".format(ig.key,len(ig.filenames)))
        for fn in ig.filenames:
          localgroup.filenames.append(fn)
  
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

  def getPairImage(self,pair):
    img1 = self.loadImage(self.groups[pair[0][0]].filenames[pair[0][1]])
    img2 = self.loadImage(self.groups[pair[1][0]].filenames[pair[1][1]])
    return np.array([img1,img2])

  # Generate couple with same chars (only id)
  def getCoupleId(self):
    gid = self.generateRandomId(len(self.groups))
    ids = self.generateRandomPair(len(self.groups[gid].filenames))
    return [[gid,ids[0]],[gid,ids[1]]]

  # Generate couple with different chars (only id)
  def getWrongId(self):
    gids = self.generateRandomPair(len(self.groups))
    id0 = self.generateRandomId(len(self.groups[gids[0]].filenames))
    id1 = self.generateRandomId(len(self.groups[gids[1]].filenames))
    return [[gids[0],id0],[gids[1],id1]]

  # Generate couple with same chars
  def getCouple(self):
    cp = self.getCoupleId()
    return self.getPairImage(cp)

  # Generate couple with different chars
  def getWrong(self):
    cp = self.getWrongId()
    return self.getPairImage(cp)

  def getGroupIdByKey(self,key):
    for gid, g in enumerate(self.groups):
      if g.key == key:
        return gid
    
    return None
  def getGroupByKey(self,key):
    for g in self.groups:
      if g.key == key:
        return g
    
    return None

  def getKey(self,groupid):
    if groupid < len(self.groups):
      return self.groups[groupid].key
    else:
      return None

  def getSingle(self,groupid,id):
    if ((groupid < len(self.groups)) and (id < len(self.groups[groupid].filenames))):
      return np.array([self.loadImage(self.groups[groupid].filenames[id])])[0]
    else:
      return None

  def getKID(self,gid,id,model):
    image = self.getSingle(gid,id).reshape((1,96,96,4))
    # use model(image), model.predict(image) cause memory leak
    kid = tf.math.l2_normalize(model(image),axis=1)
    del image
    return kid

  def getAverageKID(self,gid,model):
    temp = 0
    count = len(self.groups[gid].filenames)
    for id in range(0,count):
      kid = self.getKID(gid,id,model)
      temp = np.add(temp, kid)
 
    return np.divide(temp, count)

  def calcKidDistance(self,kid1,kid2):
    return np.sqrt(np.sum((np.power(kid1 - kid2, 2))))

class KIDObjectGroup:
  def __init__(self,og):
    self.filenames = og.filenames
    self.key = og.key
    self.id = og.id
    self.kid = None
    self.diff = None
    return

class KIDObjectDataset(ObjectDataset):
  def __init__(self,dataset,model,maxcount=0):
    self.mindist = 0.15
    self.maxdist = 0.85
    
    self.localdir = dataset.localdir
    self.groups = []
    self.type   = dataset.type
    self.total  = dataset.total
    self.model  = model
    
    if (maxcount == 0) or (maxcount > len(dataset.groups)):
      maxcount = len(dataset.groups)

    for gid in range(0,maxcount):
      self.groups.append(KIDObjectGroup(dataset.groups[gid]))

    self.generateId(model)
    return

  def generateId(self,model):
    for gid in range(0,len(self.groups)):
      print("{}: generate id {}/{}".format(self.groups[gid].key,gid+1,len(self.groups)))
      # Iterate all keys
      count = len(self.groups[gid].filenames)
      temp = 0
      kid = None
      kids = []
      for id in range(0,count):
        kid = self.getKID(gid,id,model)
        kids.append(kid)
        temp = np.add(temp, kid)

      groupkid = np.divide(temp, count)
      self.groups[gid].kid = groupkid
      diff = 0
      for id in range(0,count):
        diff += self.calcKidDistance(kids[id],groupkid)
      self.groups[gid].diff = diff
      self.groups[gid].kids = kids

      del kids, diff, temp, count, kid

    self.kidgenerated = True

  # Generate couple with same chars
  def getCoupleId(self):
    return self.getCoupleIdDist(self.mindist)

  # Generate couple with different chars
  def getWrongId(self):
    return self.getWrongIdDist(self.maxdist)

  def calcDistanceOfPair(self,pair):
      kid1 = self.groups[pair[0][0]].kids[pair[0][1]]
      kid2 = self.groups[pair[1][0]].kids[pair[1][1]]
      return self.calcKidDistance(kid1,kid2)

  def getCoupleIdDist(self,mindist):
    for i in range(0, 10):
      pair = super().getCoupleId()
      dist = self.calcDistanceOfPair(pair)
      if dist > mindist:
        break
    return pair

  def getWrongIdDist(self,maxdist):
    for i in range(0, 10):
      pair = super().getWrongId()
      dist = self.calcDistanceOfPair(pair)
      if dist < maxdist:
        break
    return pair
  # Generate couple with same chars
  def getCoupleDist(self,mindist):
    cp = self.getCoupleIdDist(mindist)
    return self.getPairImage(cp)

  # Generate couple with different chars
  def getWrongDist(self,maxdist):
    cp = self.getWrongIdDist(maxdist)
    return self.getPairImage(cp)

class DatasetSequence(Sequence):
  def __init__(self, dataset, count, batch_size):
    self.dataset = dataset
    self.epoch = 0
    self.count = count
    self.batch_size = batch_size
    self.epochdataX1 = []
    self.epochdataX2 = []
    self.epochdataY = []
    self.updateDataset()

  def __len__(self):
    return int(np.ceil(self.count / float(self.batch_size)))
    
  def __getitem__(self, idx):
    vfrom = idx * self.batch_size
    vto = (idx + 1) * self.batch_size
    #print("X:{}, Y:{}".format(len(self.epochdataX1),len(self.epochdataY)))

    arra = np.asarray(self.epochdataX1[vfrom:vto])
    arrb = np.asarray(self.epochdataX2[vfrom:vto])
    arry = np.asarray(self.epochdataY[vfrom:vto])

    return [arra,arrb],arry

  def on_epoch_end(self):
    if self.epoch % 20 == 0:
       self.epoch += 1
    else:
       # modify data
       self.updateDataset()
       self.epoch += 1

  def updateDataset(self):
    X1=[]
    X2=[]
    y=[]
    switch=True
    for _ in range(self.count):
      if switch:
        couple = self.dataset.getCouple()
        X1.append(couple[0])
        X2.append(couple[1])
        y.append(np.array([0.]))
      else:
        wrong = self.dataset.getWrong()
        X1.append(wrong[0])
        X2.append(wrong[1])
        y.append(np.array([1.]))
      switch=not switch
    
    self.epochdataX1 = np.asarray(X1)
    self.epochdataX2 = np.asarray(X2)
    self.epochdataY = np.asarray(y)

class DatasetLongSequence(Sequence):
  def __init__(self, dataset, count, batch_size):
    self.dataset = dataset
    self.epoch = 0
    self.count = count
    self.batch_size = batch_size
    self.epochdataX = []
    self.epochdataY = []
    self.updateDataset()

  def __len__(self):
    return int(np.ceil(self.count / float(self.batch_size)))
    
  def __getitem__(self, idx):
    vfrom = idx * self.batch_size
    vto = (idx + 1) * self.batch_size
    #print("X:{}, Y:{}".format(len(self.epochdataX1),len(self.epochdataY)))
    X1 = []
    X2 = []
    arra = self.epochdataX[vfrom:vto]
    arry = np.asarray(self.epochdataY[vfrom:vto])

    for pairidx in arra:
       pair = self.dataset.getPairImage(pairidx)
       X1.append(pair[0])
       X2.append(pair[1])

    return [np.asarray(X1),np.asarray(X2)],arry

  def on_epoch_end(self):
    if self.epoch % 20 == 0:
       self.epoch += 1
    else:
       # modify data
       #self.updateDataset()
       self.epoch += 1

  def updateDataset(self):
    X=[]
    y=[]
    switch=True
    for _ in range(self.count):
      if switch:
        couple = self.dataset.getCoupleId()
        X.append(couple)
        y.append(np.array([0.]))
      else:
        wrong = self.dataset.getWrongId()
        X.append(wrong)
        y.append(np.array([1.]))
      switch=not switch
    
    self.epochdataX = np.asarray(X)
    self.epochdataY = np.asarray(y)
    
    self.epochdataY = np.asarray(y)
