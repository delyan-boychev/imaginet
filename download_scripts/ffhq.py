import io
import sys
import os
import hashlib
import numpy as np
from PIL import Image
import multiprocessing
from multiprocessing import Value
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload


jobs = []
redo = []
os.mkdir("ffhq")
#redo = Value("list", [])
success = Value("i", 0)
creds = service_account.Credentials.from_service_account_file("YOUR CREDENTIALS HERE")
service = build("drive", "v3", credentials=creds)
file =  open("./ffhq_filenames.txt")
files = [k.strip().split(",") for k in file.readlines()]
def download_file(real_file_id):
  try:
    file_id = real_file_id
    request = service.files().get_media(fileId=file_id)
    file = io.BytesIO()
    downloader = MediaIoBaseDownload(file, request)
    done = False
    while done is False:
      status, done = downloader.next_chunk()
  except HttpError as error:
    print(f"An error occurred: {error}")
    file = None

  return file.getvalue()

def download(file_inf):
        file_id = file_inf[0]
        file_uuid = file_inf[1].split(".")[0]
        completed = False
        try:
            file = download_file(file_id)
            f = open(f"./ffhq/{file_uuid}.tmp.png", "wb")
            f.write(file)
            f.close()
            image = Image.open(f"./ffhq/{file_uuid}.tmp.png")
            if(hashlib.md5(np.array(image)).hexdigest() == file_inf[2]):
                image.save(f"./ffhq/{file_uuid}.png")
                completed = True
            else:
                completed = False
        except:
            completed = False
        if os.path.exists(f"./ffhq/{file_uuid}.tmp.png"):
            os.remove(f"./ffhq/{file_uuid}.tmp.png") 
        return completed
            
num_workers = 20

image_queue = multiprocessing.Queue()
def worker(queue, completed, lock):
    while True:
        item = queue.get()
        if item is None:
            break
        compl = download(item)
        with lock:
            if compl:
                completed.value += 1
            else:
                image_queue.put(item)
            sys.stdout.write(f"\rCompleted {completed.value}")
            sys.stdout.flush()
        queue.task_done()

num_workers = 20

image_queue = multiprocessing.JoinableQueue()
completed = multiprocessing.Value("i", 0)
failed = multiprocessing.JoinableQueue()
lock = multiprocessing.Lock()

for file_inf in files:
    image_queue.put(file_inf)

processes = []
for _ in range(num_workers):
    p = multiprocessing.Process(target=worker, args=(image_queue, completed, lock))
    p.start()
    processes.append(p)
image_queue.join()

for _ in range(num_workers):
    image_queue.put(None)

for p in processes:
    p.join()