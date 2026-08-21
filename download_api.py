import requests
import json
import os

def download_msmarco_api():
    url = "https://datasets-server.huggingface.co/rows?dataset=ai4bharat/MSMARCO-XI&config=default&split=train&offset=0&length=50"
    print(f"Fetching from {url}")
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()
    
    os.makedirs("./data", exist_ok=True)
    out_path = "./data/msmarco_xi_sample.jsonl"
    
    with open(out_path, "w", encoding="utf-8") as f:
        for row in data["rows"]:
            item = row["row"]
            # Convert to our schema
            record = {"metadata": {}}
            if "query" in item and "passage" in item:
                record["id"] = str(item.get("passage_id", item.get("id", row["row_idx"])))
                record["text"] = str(item["passage"])
                record["metadata"]["query"] = str(item["query"])
                record["metadata"]["query_id"] = str(item.get("query_id", ""))
            elif "text" in item:
                record["id"] = str(item.get("id", row["row_idx"]))
                record["text"] = str(item["text"])
            else:
                record["id"] = str(item.get("id", row["row_idx"]))
                record["text"] = json.dumps(item)
                
            record["metadata"]["language"] = "hi"
            for k, v in item.items():
                if k not in ["text", "passage"] and k not in record["metadata"]:
                    record["metadata"][k] = v
                    
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            
    print(f"Saved 50 records to {out_path}")
    
    # Print the schema of the first row
    print("Schema of first raw item from MSMARCO-XI:")
    print(json.dumps(data["rows"][0]["row"], indent=2))
    return out_path

if __name__ == "__main__":
    download_msmarco_api()
