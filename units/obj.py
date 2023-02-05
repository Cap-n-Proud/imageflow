import ast
import json

response = {
    "input": {
        "image": "/tmp/tmpczzpx4uk202301013862.jpg",
        "model_name": "yolox-s",
        "confidence": 0.6,
        "nms": 0.3,
        "tsize": 640,
        "return_json": true,
    },
    "output": {
        "img": "/tmp/tmpczzpx4uk202301013862.jpg",
        "inference": "[{'cls': 'bed', 'confidence': '0.68586016'}, {'cls': 'person', 'confidence': '0.6823821'}]",
    },
    "id": null,
    "created_at": null,
    "started_at": "2023-02-05T00:12:48.285817+00:00",
    "completed_at": "2023-02-05T00:12:48.534006+00:00",
    "logs": "2023-02-05 00:12:48.531 | INFO     | tools.demo:inference:165 - Infer time: 0.0957s\n",
    "error": null,
    "status": "succeeded",
    "webhook": null,
    "output_file_prefix": null,
    "webhook_events_filter": ["output", "logs", "start", "completed"],
}


response_str = json.dumps(response)
response_str = response_str.replace("True", '"True"')
response_str = response_str.replace("False", '"False"')
response_str = response_str.replace("null", '"Nul"')

data = json.loads(response_str)

inference = json.loads(data["output"]["inference"].replace("'", '"'))
unique_cls = list(set([d["cls"] for d in inference]))

print(unique_cls)
