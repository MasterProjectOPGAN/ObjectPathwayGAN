import os

for count, filename in enumerate(os.listdir(".")):
    if filename == "rename.py":
        continue
    dst = filename[0:5:]+filename[8::]
    os.rename(filename, dst)
