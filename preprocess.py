from datasets import load_dataset
import json
import re
import ndjson

add_prompt = "I would like you to write a verilog code for the following description:\n"

dataset = load_dataset("json", data_files="./verilog_data.jsonl", split="train")
with open("all.jsonl", "w", encoding="utf-8") as f:
    out = []
    for data in dataset:
        try:
            parse = json.loads(data["description"])
#        parse = re.search(r'"description":\s*"(.+?)",\s*"rank"', data["description"])
            prompt = add_prompt + parse["description"]
            code = data["code"]
            out.append({"messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": code}]})
        except Exception as e:
            print(f"Error parsing data: {e}")
        finally:
            pass
    ndjson.dump(out, f)