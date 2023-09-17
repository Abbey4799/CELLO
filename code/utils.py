import json
import os
import re

def readjsonl(datapath):
    res = []
    with open(datapath, "r", encoding='utf-8') as f:
        for line in f.readlines():
            res.append(json.loads(line))
    return res


def writejsonl(data, datapath):
    with open(datapath, "w", encoding='utf-8') as f:
        for item in data:
            json_item = json.dumps(item, ensure_ascii=False)
            f.write(json_item + "\n")

def writejson(data, json_path):
    json_str = json.dumps(data, indent=4, ensure_ascii=False)
    with open(json_path, "w", encoding='utf-8') as json_file:
        json_file.write(json_str)

def readjson(datapath):
    with open(datapath, "r", encoding='utf-8') as f:
        res = json.load(f)
    return res


def check_folder(path):
    if not os.path.exists(path):
        print(f"{path} not exists, create it")
        os.makedirs(path)

def get_name(name, pattern, mode = 0):
    match = re.search(pattern, name)

    if match:
        extracted_content = match.group(mode)
        return extracted_content
    else:
        print("未找到匹配的内容")