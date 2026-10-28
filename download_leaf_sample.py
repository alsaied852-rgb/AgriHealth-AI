import requests
import struct
import os
import random
import time

URL = "https://huggingface.co/datasets/mohanty/PlantVillage/resolve/main/data.zip"
ZIP_SIZE = 2184723441
DIR_OFFSET = 2156772140
DIR_SIZE = 27951203
INDEX_START = ZIP_SIZE - 65536

os.makedirs("data/plant_leaf", exist_ok=True)

session = requests.Session()

print("Downloading ZIP index...")
r = session.get(
    URL,
    headers={"Range": f"bytes={DIR_OFFSET}-{DIR_OFFSET + DIR_SIZE - 1}"},
    timeout=120
)
r.raise_for_status()
index = r.content

entries = []
pos = 0

while True:
    pos = index.find(b"PK\x01\x02", pos)
    if pos < 0 or pos + 46 > len(index):
        break

    name_len = struct.unpack_from("<H", index, pos + 28)[0]
    extra_len = struct.unpack_from("<H", index, pos + 30)[0]
    comment_len = struct.unpack_from("<H", index, pos + 32)[0]
    compressed_size = struct.unpack_from("<I", index, pos + 20)[0]
    local_offset = struct.unpack_from("<I", index, pos + 42)[0]
    method = struct.unpack_from("<H", index, pos + 10)[0]

    end = pos + 46 + name_len + extra_len + comment_len
    if end > len(index):
        break

    name = index[pos + 46:pos + 46 + name_len].decode("utf-8", errors="replace")

    if (
        name.startswith("raw/color/")
        and name.lower().endswith((".jpg", ".jpeg", ".png"))
        and method == 8
        and compressed_size > 0
        and compressed_size < 2_000_000
    ):
        parts = name.split("/")
        if len(parts) >= 4:
            class_name = parts[2]
            entries.append((name, class_name, compressed_size, local_offset))

    pos = end

healthy = [e for e in entries if "healthy" in e[1].lower()]
diseased = [e for e in entries if "healthy" not in e[1].lower()]

random.seed(42)
random.shuffle(healthy)
random.shuffle(diseased)
selected = healthy[:20] + diseased[:20]

print("Healthy candidates:", len(healthy))
print("Diseased candidates:", len(diseased))
print("Selected images:", len(selected))

# Stop before downloading images until the ZIP format is verified.
print("Index parsed successfully. Image extraction needs ZIP local-header verification.")
print("No images have been downloaded yet.")
