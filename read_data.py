import os
import gzip

folder = "GSE42568_RAW"

files = [f for f in os.listdir(folder) if f.endswith(".CEL.gz")]

print("Total samples:", len(files))

# read one file
path = os.path.join(folder, files[0])

with gzip.open(path, "rt", errors="ignore") as f:
    lines = f.readlines()

print("\nFirst 50 lines:\n")
for line in lines[:50]:
    print(line.strip())