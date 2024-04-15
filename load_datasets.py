import numpy as np
import pandas as pd


def load_datasets():
    # Paths to the data files
    ppi_path = 'BioSNAP/PP-Decagon_ppi.csv'
    dd_path = 'BioSNAP/DD-Miner_miner-disease-disease.tsv'
    dg_path = 'BioSNAP/DG-AssocMiner_miner-disease-gene.tsv'
    mapping_path = 'BioSNAP/disease_mappings.tsv'
    # Load and display the first few rows of each file to understand their structure
    ppi_data = pd.read_csv(ppi_path, header=None, names=['Gene 1', 'Gene 2'], skiprows=lambda i: np.random.rand() > 0.01)
    dd_data = pd.read_csv(dd_path, sep='\t', skiprows=lambda i: i > 0 and np.random.rand() > 0.1)
    dg_data = pd.read_csv(dg_path, sep='\t')
    mapping_data = pd.read_csv(mapping_path, sep='\t')

    # Remove the "DOID:" prefix from the relevant columns
    dd_data.columns = ['Disease 1', 'Disease 2']
    dd_data['Disease 1'] = dd_data['Disease 1'].str.replace('DOID:', '')
    dd_data['Disease 2'] = dd_data['Disease 2'].str.replace('DOID:', '')

    dd_data.columns = ['Disease 1 DOID', 'Disease 2 DOID']

    dg_data.columns = ['Disease ID', 'Disease Name', 'Gene ID']

    filtered_mapping_data = mapping_data[mapping_data['vocabulary'] == 'DO']

    # Create a mapping from Disease ID to DOID code
    disease_id_to_code = filtered_mapping_data.set_index('code')['diseaseId'].to_dict()

    # Map the Disease IDs in dg_data to DOID codes
    dd_data['Disease ID1'] = dd_data['Disease 1 DOID'].map(disease_id_to_code)
    dd_data['Disease ID2'] = dd_data['Disease 2 DOID'].map(disease_id_to_code)

    # Drop rows where either 'DiseaseID1' or 'DiseaseID2' is NaN
    dd_data_filtered = dd_data.dropna(subset=['Disease ID1', 'Disease ID2'])

    # Display the data
    # print("Protein-Protein Interaction Data:")
    # print(ppi_data.head())
    # print("\nDisease-Disease Interaction Data:")
    # print(dd_data_filtered.head())
    # print("\nDisease-Gene Interaction Data:")
    # print(dg_data.head())
    # print("\nDisease Mapping Data:")
    # print(mapping_data.head())

    return ppi_data, dd_data_filtered, dg_data
