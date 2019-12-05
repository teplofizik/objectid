
class KidRecord:
  def __init__(self,key,id):
    self.key = key
    self.id = id

class KidBase:
  def __init__(self):
    self.database = []
    
  def set(self, key, id):
    if self.get(key) is None:
      self.database.append(KidRecord(key,id))
      
  def get(self, key):
    for record in self.database:
      if record.key == key:
        return record.id
    return None
