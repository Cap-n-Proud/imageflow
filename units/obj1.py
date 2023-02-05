import json

data = {
    "input": {
        "image": "/tmp/tmpo3_6sppd202301013862.jpg",
        "model_name": "yolox-s",
        "confidence": 0.6,
        "nms": 0.3,
        "tsize": 640,
        "return_json": true,
    },
    "output": {
        "img": "/tmp/tmpo3_6sppd202301013862.jpg",
        "inference": "[{'cls': 'bed', 'confidence': '0.68586016'}, {'cls': 'person', 'confidence': '0.6823821'}]",
    },
}

data.replace("true", "'True'")
print(data)
# inference = json.loads(data["output"]["inference"].replace("'", '"'))
# unique_cls = list(set([d["cls"] for d in inference]))
inference = json.loads(data["output"]["inference"])
unique_cls = list(set([d["cls"] for d in inference]))

print(unique_cls)
