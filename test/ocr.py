from .kidbase import KidBase

class DetectionResult:
  def __init__(self,key,dist):
    self.key = key
    self.dist = dist

class OCR:
  def __init__(self,dataset):
    self.dataset = dataset
    self.idbase = KidBase()
    
  def calcid(self, model, keys):
    self.ids = keys
    for key in keys:
      gid = self.dataset.getGroupIdByKey(key)
      self.idbase.set(key, self.dataset.getAverageKID(gid,model))
    
  def detectone(self,id,distrange):
    list = self.detect(id,distrange)
    if len(list) > 0:
      minindex = 0
      mindist = list[0].dist
      for i, dr in enumerate(list):
        if dr.dist < mindist:
          minindex = i
          mindist = dr.dist
          
      return list[minindex].key
    else:
      return None
    
  def detect(self,id,distrange):
    mgid = self.dataset.getGroupIdByKey(key)
    res = []
    if mgid is not None:
      for bkey in ids:
        baseid = self.idbase.get(bkey)
        dist = self.dataset.calcKidDistance(baseid,id)
        if dist < distrange:
          res.append(DetectionResult(bkey, dist))
          
      return res
    else:
      print("Not found: {}".format(key))
      return None