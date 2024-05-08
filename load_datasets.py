import os

import numpy as np
import pandas as pd


def load_datasets():
    # Paths to the data files
    ppi_path = 'BioSNAP/PP-Decagon_ppi.csv'
    dd_path = 'BioSNAP/DD-Miner_miner-disease-disease.tsv'
    dg_path = 'BioSNAP/DG-AssocMiner_miner-disease-gene.tsv'
    mapping_path = 'BioSNAP/disease_mappings.csv'

    ppi_data_path = 'BioSNAP/pp_data.csv'
    dd_data_path = 'BioSNAP/dd_data.tsv'
    dg_data_path = 'BioSNAP/dg_data.tsv'
    # mapping_data_path = 'BioSNAP/mapping_data.csv'
    if ((os.path.exists(ppi_data_path) and os.path.exists(ppi_data_path)) and
            (os.path.exists(ppi_data_path) and os.path.exists(ppi_data_path))):

        # Read the saved data frame
        ppi_data = pd.read_csv(ppi_data_path)
        dd_data = pd.read_csv(dd_data_path)
        dg_data = pd.read_csv(dg_data_path)
        # mapping_data = pd.read_csv(mapping_data_path)

    else:

        # Load datasets
        ppi_data = pd.read_csv(ppi_path, header=None, names=['Gene ID 1', 'Gene ID 2'])
        dd_data = pd.read_csv(dd_path, sep='\t')
        dg_data = pd.read_csv(dg_path, sep='\t')
        mapping_data = pd.read_csv(mapping_path, sep=',')

        # Create a unique set of genes
        unique_genes = pd.concat([ppi_data['Gene ID 1'], ppi_data['Gene ID 2']]).unique()
        gene_to_index = {gene: idx for idx, gene in enumerate(unique_genes)}
        # Map the genes to indices in the DataFrame
        ppi_data['Gene 1'] = ppi_data['Gene ID 1'].map(gene_to_index)
        ppi_data['Gene 2'] = ppi_data['Gene ID 2'].map(gene_to_index)

        # Remove the "DOID:" prefix from the relevant columns
        # dd_data.columns = ['Disease 1', 'Disease 2']
        # unique_diseases = pd.concat([dd_data['Disease 1'], dd_data['Disease 2']]).unique()
        # dd_data['Disease 1'] = dd_data['Disease 1'].str.replace('DOID:', '')
        # dd_data['Disease 2'] = dd_data['Disease 2'].str.replace('DOID:', '')
        dd_data.columns = ['DOID 1', 'DOID 2']
        mapping_data.columns = ['DOID', 'Name', 'UML']
        mapping_data['UML'] = mapping_data['UML'].str.replace('UMLS_CUI:', '')
        mapping_data['UML trim'] = mapping_data['UML'].str.split('|')

        # disease_mapping_dict = mapping_data.set_index('DOID')['UML'].to_dict()

        # Map the DOIDs in the DD file to their corresponding CUI IDs
        # dd_data['Disease ID1'] = dd_data['Disease 1 DOID'].map(disease_mapping_dict)
        # dd_data['Disease ID2'] = dd_data['Disease 2 DOID'].map(disease_mapping_dict)

        # # Map the Disease IDs in dg_data to DOID codes
        # dd_data['Disease ID1'] = dd_data['Disease 1 DOID'].map(disease_mapping_dict)
        # dd_data['Disease ID2'] = dd_data['Disease 2 DOID'].map(disease_mapping_dict)

        # Drop rows where either 'DiseaseID1' or 'DiseaseID2' is NaN
        # dd_data_filtered = dd_data.dropna(subset=['Disease ID1', 'Disease ID2']).copy().reset_index(drop=True)

        unique_diseases = pd.concat([dd_data['DOID 1'], dd_data['DOID 2']]).unique()
        disease_to_index = {disease: idx for idx, disease in enumerate(unique_diseases)}
        # Map the genes to indices in the DataFrame
        dd_data['Disease 1'] = dd_data['DOID 1'].map(disease_to_index)
        dd_data['Disease 2'] = dd_data['DOID 2'].map(disease_to_index)

        # Filter the Disease-Gene Data
        dg_data.columns = ['Disease ID', 'Disease Name', 'Gene ID']

        uml_to_doid = {}
        for _, map_row in mapping_data.iterrows():
            for uml in map_row['UML trim']:
                uml_to_doid[uml] = map_row['DOID']

        # Use this optimized dictionary to map Disease ID to DOID
        dg_data['DOID'] = dg_data['Disease ID'].map(uml_to_doid)

        # Filter diseases and genes based on what's available in disease-disease and gene-gene networks
        dg_data = dg_data[(dg_data['DOID'].isin(unique_diseases)) & (dg_data['Gene ID'].isin(unique_genes))]
        # Map the genes to indices in the DataFrame
        dg_data['Gene'] = dg_data['Gene ID'].map(gene_to_index)
        dg_data['Disease'] = dg_data['DOID'].map(disease_to_index)

        unique_diseases_dg = dg_data['Disease'].unique()
        unique_genes_dg = dg_data['Gene'].unique()

        disease_to_index_dg = {disease: idx for idx, disease in enumerate(unique_diseases_dg)}
        gene_to_index_dg = {disease: idx for idx, disease in enumerate(unique_genes_dg)}

        dg_data['Gene_idx'] = dg_data['Gene'].map(gene_to_index_dg)
        dg_data['Disease_idx'] = dg_data['Disease'].map(disease_to_index_dg)

        # Display the data
        # print("Protein-Protein Interaction Data:")
        # print(ppi_data.head())
        # print("\nDisease-Disease Interaction Data:")
        # print(dd_data_filtered.head())
        # print("\nDisease-Gene Interaction Data:")
        # print(dg_data.head())
        # print("\nDisease Mapping Data:")
        # print(mapping_data.head())

        ppi_data.to_csv(ppi_data_path, index=False)
        dd_data.to_csv(dd_data_path, index=False)
        dg_data.to_csv(dg_data_path, index=False)

    return ppi_data, dd_data, dg_data
