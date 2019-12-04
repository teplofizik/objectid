

class DistanceTester:
  def __init__(self,keylist):
    self.keys = keylist

  def test(self,key,model,dataset):
    mgid = dataset.getGroupIdByKey(key)
    if mgid == None:
      print("Key {}: not available\n".format(key));
      return

    basekid = dataset.getAverageKID(mgid,model)
    for k in self.keys:
      gid = dataset.getGroupIdByKey(k)
      if gid == None:
        print("Key {}: not available\n".format(k));
        continue

      kid = dataset.getAverageKID(gid,model)
      print("Key {}: {}".format(k,self.calcKidDistance(basekid,kid)))

  def calcKidDistance(self,kid1,kid2):
    return np.sqrt(np.sum((np.power(kid1 - kid2, 2))))
