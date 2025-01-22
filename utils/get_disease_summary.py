import json
import os

import pandas as pd


def load_human_do(json_file_path):
    with open(json_file_path, 'r') as file:
        human_do = json.load(file)
    return human_do


def get_disease_summary(doid, human_do):
    """Fetches the summary ('val') for a given DOID."""
    try:
        doid_cleaned = doid.replace("DOID:", "")
        doid_uri = f"http://purl.obolibrary.org/obo/DOID_{doid_cleaned}"

        # Search for the matching DOID in the JSON data
        # Search for the matching DOID in the JSON data
        for entry in human_do['graphs'][0]['nodes']:
            if entry['id'] == doid_uri:
                # Extract the 'val' field under 'definition'
                meta = entry.get('meta', {})
                definition = meta.get('definition', {})
                val = definition.get('val', None)
                if val:
                    return val.strip()  # Return the full summary text
                return "Summary not available"
        return "DOID not found in HumanDO.json"
    except Exception as e:
        return f"Error fetching summary for DOID {doid}: {e}"


def create_disease_summary_df(dd_data):
    """Creates a DataFrame with DOIDs and their corresponding summaries."""
    disease_summaries_path = 'disease_summaries.csv'
    if os.path.exists(disease_summaries_path):
        disease_sum = pd.read_csv(disease_summaries_path)
    else:
        json_file_path = "./BioSNAP/HumanDO.json"
        human_do = load_human_do(json_file_path)

        # Extract unique DOIDs from the edge list
        unique_doids = pd.concat([dd_data['did1'], dd_data['did2']]).unique()

        # Fetch summaries for all unique DOIDs
        disease_summaries = []
        for index, doid in enumerate(unique_doids):
            summary = get_disease_summary(doid, human_do)
            disease_summaries.append({"DOID": doid, "Summary": summary})
            print(f"Processing index: {index}, DOID: {doid}")

        # Convert to DataFrame
        disease_sum = pd.DataFrame(disease_summaries)
        disease_sum.to_csv("disease_summaries.csv", index=False)
    return disease_sum
