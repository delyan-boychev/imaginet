import subprocess
import gdown
import shutil
import os
import urllib.request

os.mkdir("required_libs")



subprocess.run(["git", "clone", "https://github.com/grip-unina/DMimageDetection.git"], cwd="./required_libs") 
subprocess.run(["git", "reset", "--hard", "745ad9e"], cwd="./required_libs/DMimageDetection")
subprocess.run(["git", "clone", "https://github.com/HighwayWu/LASTED"], cwd="./required_libs")
subprocess.run(["git", "clone", "https://github.com/grip-unina/GANimageDetection"], cwd="./required_libs")
subprocess.run(["git", "apply", "../../get_method_here.patch"], cwd="./required_libs/DMimageDetection")

print("Downloading Grag2021 weights...")
urllib.request.urlretrieve("https://www.grip.unina.it/download/prog/GANdetection/weights/gandetection_resnet50nodown_stylegan2.pth", "./required_libs/grag2021_stylegan2.pth")
print("Downloading Corvi2022 weights...")
gdown.download(id="1sAoAuOGCWS4dAMBhDkRHgBf4SgBgvkVf", output="./required_libs/weights.zip")
shutil.unpack_archive("./required_libs/weights.zip", "./required_libs/")
os.remove("./required_libs/weights.zip")
print("Downloading ImagiNet weights...")
gdown.download(id="1kI-j2J8JinyRBdemfPyPMhNRE79FlsgX", output="./required_libs/imaginet_weights.pt")
gdown.download(id="19MzsJpPs9i0nFO1vB63i-T0iZ48aHIoA", output="./required_libs/model_epoch_best.pth")
print("Downloading LASTED weights...")
gdown.download(id="1xrBOQQfDUUQQ19TakIMwdyDtHLmGa25c", output="./required_libs/LASTED_pretrained.pt")
