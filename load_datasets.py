import numpy as np
import pandas as pd


def load_datasets():
    # Paths to the data files
    ppi_path = 'BioSNAP/PP-Decagon_ppi.csv'
    dd_path = 'BioSNAP/DD-Miner_miner-disease-disease.tsv'
    dg_path = 'BioSNAP/DG-AssocMiner_miner-disease-gene.tsv'
    mapping_path = 'BioSNAP/disease_mappings.tsv'
    # Load datasets
    ppi_data = pd.read_csv(ppi_path, header=None, names=['Gene ID 1', 'Gene ID 2'])
    # Create a unique set of genes
    unique_genes = pd.concat([ppi_data['Gene ID 1'], ppi_data['Gene ID 2']]).unique()

    gene_to_index = {gene: idx for idx, gene in enumerate(unique_genes)}

    # Map the genes to indices in the DataFrame
    ppi_data['Gene 1'] = ppi_data['Gene ID 1'].map(gene_to_index)
    ppi_data['Gene 2'] = ppi_data['Gene ID 2'].map(gene_to_index)

    dd_data = pd.read_csv(dd_path, sep='\t')
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
    dd_data_filtered = dd_data.dropna(subset=['Disease ID1', 'Disease ID2']).copy()

    # Create a unique set of genes
    unique_diseases = pd.concat([dd_data_filtered['Disease ID1'], dd_data_filtered['Disease ID2']]).unique()

    disease_to_index = {disease: idx for idx, disease in enumerate(unique_diseases)}

    # Map the genes to indices in the DataFrame
    dd_data_filtered['Disease 1'] = dd_data_filtered['Disease ID1'].map(disease_to_index)
    dd_data_filtered['Disease 2'] = dd_data_filtered['Disease ID2'].map(disease_to_index)

    # Filter the Disease-Gene Data
    # Filter diseases and genes based on what's available in disease-disease and gene-gene networks
    dg_data = dg_data[(dg_data['Disease ID'].isin(unique_diseases)) & (dg_data['Gene ID'].isin(unique_genes))]

    # Map the genes to indices in the DataFrame
    dg_data['Gene'] = dg_data['Gene ID'].map(gene_to_index)
    dg_data['Disease'] = dg_data['Disease ID'].map(disease_to_index)

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
