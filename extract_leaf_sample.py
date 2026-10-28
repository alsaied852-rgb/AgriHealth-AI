import requests
import struct
import os
import random
import zlib
import time

URL = "https://huggingface.co/datasets/mohanty/PlantVillage/resolve/main/data.zip"
ZIP_SIZE = 2184723441
DIR_OFFSET = 2156772140
DIR_SIZE = 27951203

os.makedirs("data/plant_leaf", exist_ok=True)
session = requests.Session()

def get_range(start, end):
    r = session.get(
        URL,
        headers={"Range": f"bytes={start}-{end}"},
        timeout=90
    )
    r.raise_for_status()
    if r.status_code != 206:
        raise RuntimeError(f"Expected HTTP 206, got {r.status_code}")
    return r.content

print("Reading ZIP index...")
index = get_range(DIR_OFFSET, DIR_OFFSET + DIR_SIZE - 1)

entries = []
pos = 0

while True:
    pos = index.find(b"PK\x01\x02", pos)
    if pos < 0 or pos + 46 > len(index):
        break

    name_len = struct.unpack_from("<H", index, pos + 28)[0]
    extra_len = struct.unpack_from("<H", index, pos + 30)[0]
    comment_len = struct.unpack_from("<H", index, pos + 32)[0]
    method = struct.unpack_from("<H", index, pos + 10)[0]
    compressed_size = struct.unpack_from("<I", index, pos + 20)[0]
    uncompressed_size = struct.unpack_from("<I", index, pos + 24)[0]
    local_offset = struct.unpack_from("<I", index, pos + 42)[0]
    crc = struct.unpack_from("<I", index, pos + 16)[0]

    end = pos + 46 + name_len + extra_len + comment_len
    if end > len(index):
        break

    name = index[pos + 46:pos + 46 + name_len].decode("utf-8", errors="replace")

    if (name.startswith("raw/color/")
        and name.lower().endswith((".jpg", ".jpeg", ".png"))
        and method == 8 and compressed_size > 0
        and compressed_size < 2_000_000):
        parts = name.split("/")
        if len(parts) >= 4:
            entries.append({
                "name": name,
                "class": parts[2],
                "compressed": compressed_size,
                "size": uncompressed_size,
                "offset": local_offset,
                "crc": crc
            })
    pos = end

healthy = [e for e in entries if "healthy" in e["class"].lower()]
diseased = [e for e in entries if "healthy" not in e["class"].lower()]

random.seed(42)
random.shuffle(healthy)
random.shuffle(diseased)
selected = healthy[:20] + diseased[:20]

print("Selected:", len(selected), "images")

for i, item in enumerate(selected, 1):
    try:
        offset = item["offset"]
        header = get_range(offset, offset + 29)

        if header[:4] != b"PK\x03\x04":
            print(f"{i}/40: Invalid ZIP header; skipped")
            continue

        name_len = struct.unpack_from("<H", header, 26)[0]
        extra_len = struct.unpack_from("<H", header, 28)[0]
        data_start = offset + 30 + name_len + extra_len
        data_end = data_start + item["compressed"] - 1

        compressed = get_range(data_start, data_end)
        image_bytes = zlib.decompress(compressed, -15)

        if len(image_bytes) != item["size"]:
            print(f"{i}/40: Size check failed; skipped")
            continue

        filename = f"sample_{i:02d}_{os.path.basename(item['name']).replace(' ', '_')}"
        path = os.path.join("data", "plant_leaf", filename)

        with open(path, "wb") as f:
            f.write(image_bytes)

        print(f"{i}/40 saved: {filename} ({item['class']})")
        time.sleep(0.2)

    except Exception as e:
        print(f"{i}/40 failed: {type(e).__name__}: {e}")

print("\nSaved image count:", len(os.listdir("data/plant_leaf")))
print("Finished.")
