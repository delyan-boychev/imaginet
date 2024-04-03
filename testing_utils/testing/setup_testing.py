import subprocess
import gdown
import shutil
import os
import urllib.request

os.mkdir("required_libs")



subprocess.run(["git", "clone", "https://github.com/grip-unina/DMimageDetection.git"], cwd="required_libs") 
subprocess.run(["git", "clone", "https://github.com/HighwayWu/LASTED"], cwd="./required_libs")
subprocess.run(["git", "clone", "https://github.com/grip-unina/GANimageDetection"], cwd="./required_libs")

urllib.request.urlretrieve("https://www.grip.unina.it/download/prog/GANdetection/weights/gandetection_resnet50nodown_stylegan2.pth", "./required_libs/grag2021_stylegan2.pth")
gdown.download(id="1sAoAuOGCWS4dAMBhDkRHgBf4SgBgvkVf", output="./required_libs/weights.zip")
shutil.unpack_archive("./required_libs/weights.zip", "./required_libs/")
os.remove("./required_libs/weights.zip")
gdown.download(id="1fXBKvT0Z-2pwL0i035T5phBfq11e3GHn", output="./required_libs/imaginet_weights.pt")
gdown.download(id="1xrBOQQfDUUQQ19TakIMwdyDtHLmGa25c", output="./required_libs/LASTED_pretrained.pt")
