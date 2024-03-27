import os
file = open("./annotations/train_annotations.txt", "r")

file2 = open("./annotations/test_annotations.txt", "r")

lines = file.readlines()
lines2 = file2.readlines()
directory_to_list = "./" #REPLACE WITH THE FOLDER WHERE ARE DOWNLOADED ARE IMAGE FOLDERS
items = [os.path.normpath(line.split(',')[0].strip()) for line in lines]
items += [os.path.normpath(line.split(',')[0].strip())for line in lines2]
items = set(items)

num = 0
itrt = os.walk(directory_to_list)
def delete_unused_files(annotations_set, directory):
    k = 0
    to_remove = []
    referenced_files = annotations_set
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            relative_path = os.path.relpath(file_path, directory)

            if relative_path not in referenced_files:
                to_remove.append(file_path)
            else:
                k += 1
    return k, to_remove
preserved, to_remove = delete_unused_files(items, directory_to_list)

def yes_or_no(question):
    while "The answer is invalid":
        reply = str(input(question+' (y/n): ')).lower().strip()
        if reply[:1] == 'y':
            return True
        if reply[:1] == 'n':
            return False

if preserved != 200000:
    print("Error: not found all files")
else:
    print(f"Preserved files: {preserved}")
    check = yes_or_no("Are you sure that you want to remove not needed files?")
    if check:
        for f in to_remove:
            os.remove(f)
        print("Deleted")
