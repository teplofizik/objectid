import os
import requests, zipfile, io

#######################################################################################
# class for downloading and unpacking zip archives from external server (with caching):
# example: 
#   downloader = Downloader("http://sample.com/data/","data");
#   downloader.download(["train1.zip", "val1.zip", "train2.zip", "val2.zip"]);
# http://sample.com/data/train1.zip and other files will be downloaded and unpacked to 
# "data" directory, downloaded archives will skip on all next calls 
# (useful for google CoLab)
#######################################################################################
class Downloader:
  # base: Part of URL with archives: "http://sample.com/data/"
  # localdir: Local directory to save unpacked data: "unpacked",""
  def __init__(self,base,localdir):
    self.urlbase = base
    self.dir = localdir
    self.indexdir = self.dir + "/index/"
    if not os.path.isdir(self.dir):
      os.mkdir(self.dir)
    if not os.path.isdir(self.indexdir):
      os.mkdir(self.indexdir)

  # check mark: is archive already downloaded and unpacked
  def hasMark(self,name):
    return os.path.isfile(self.indexdir+name);

  # mark file as downloaded
  def makeMark(self,name):
    f = open(self.indexdir+name,"w+")
    f.write("ok")
    f.close()

  # data_list: list of archives in "base" folder: ["train.zip","val.zip","checkid.zip"] etc
  def download(self,data_list):
    for data in data_list:
      if self.hasMark(data):
        print("Already: {}".format(data))
      else:
        r = requests.get(self.urlbase + data, stream=True)
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall("kanjiid")
        self.makeMark(data)
        print("Download: {}".format(data))