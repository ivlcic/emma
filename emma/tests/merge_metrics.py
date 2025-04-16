import itertools
import zipfile
import os
import json
import pandas as pd


# Path to the new uploaded zip file
new_zip_path = 'metrics.zip'
new_extracted_folder_path = 'tmp'
new_merged_json_path = 'merged_metrics.json'
new_csv_path = 'metrics'
groups = [
    ['newsmon', 'eurlex'],
    ['All', 'Frequent', 'Rare'],
    ['weak', 'svm', 'logreg', 'xlmrb', 'zshot', 'mlknn', 'raexmcsim', 'raexmc'],
    ['all', 'random', 'majority', 'gte', 'jina', '_bge_m3_', '_ftbge_m3_']
]
all_groups = set(itertools.chain.from_iterable(groups))

if os.path.exists(new_zip_path):
    # Extract the zip file
    with zipfile.ZipFile(new_zip_path, 'r') as zip_ref:
        zip_ref.extractall(new_extracted_folder_path)

# List all JSON files in the extracted folder
new_json_files = [
    os.path.join(new_extracted_folder_path, file)
    for file in os.listdir(new_extracted_folder_path) if file.endswith('.json')
]

# Merge JSON content
new_merged_data = []
for file_path in new_json_files:
    with open(file_path, 'r') as f:
        try:
            data = json.load(f)
            if isinstance(data, list):
                new_merged_data.extend(data)
            else:
                new_merged_data.append(data)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from {file_path}: {e}")

# Write the merged data to a new JSON file
with open(new_merged_json_path, 'w') as f:
    json.dump(new_merged_data, f, indent=4)

all_metrics = {}
for entry in new_merged_data:
    model_name = entry.get("model_name", "Unknown")
    model_name = model_name.replace('mulabel', 'newsmon')
    model_name = model_name.replace('eurlex_all', 'eurlex')
    model_name = model_name.replace('_e_bge_m3_', '_ftbge_m3_')
    model_name = model_name.replace('_n_bge_m3_', '_ftbge_m3_')
    metrics = entry.get("epochs", [])
    if metrics:
        all_metrics[model_name] = metrics[0]
        all_metrics[model_name]['model_name'] = model_name

sorted = {}
def group_sort(curr_level, data: dict):
    global groups
    if curr_level == len(groups):
        return

    default_name = groups[curr_level][0]
    for g in groups[curr_level]:
        data[g] = {}
        for m_name in list(data):
            if g not in m_name or m_name in all_groups:
                continue
            data[g][m_name] = data.pop(m_name)
    for m_name in list(data):
        if m_name in all_groups:
            continue
        data[default_name][m_name] = data.pop(m_name)

    for g in groups[curr_level]:
        group_sort(curr_level + 1, data[g])

sorted_metrics = []
def flatten_dict(data):
    for k, v in data.items():
        if k not in all_groups:
            sorted_metrics.append(v)
        else:
            flatten_dict(v)

group_sort(0, all_metrics)
flatten_dict(all_metrics)

new_df = pd.DataFrame(sorted_metrics)


new_df.columns = new_df.columns.str.replace('test/', '')

#df.rename(columns={"A": "a", "B": "c"})
new_df = new_df[
    [
        'model_name',
        'micro.f1', 'micro.p', 'micro.r',
        'acc',
        'macro.f1', 'macro.p', 'macro.r',
        'weighted.f1', 'weighted.p', 'weighted.r',
        'hamming_loss',
        'r-p@1', 'r-p@3', 'r-p@5', 'r-p@7', 'r-p@9',
        'p@1', 'p@3', 'p@5', 'p@7', 'p@9', 'r@1',
        'r@3', 'r@5', 'r@7', 'r@9',
        'ndcg', 'ndcg@1', 'ndcg@3', 'ndcg@5', 'ndcg@7', 'ndcg@9',
        'r-p', 'p', 'r'
    ]
]

new_df.to_csv(new_csv_path + '.csv', index=False)
#new_df = pd.DataFrame(rare_flattened_data)
#new_df.to_csv(new_csv_path + '_rare.csv', index=False)
#new_df = pd.DataFrame(frequent_flattened_data)
#new_df.to_csv(new_csv_path + '_frequent.csv', index=False)
