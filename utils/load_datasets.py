import os
import pandas as pd


def load_datasets():
    # Paths to the data files
    ppi_path = './BioSNAP/PP-Decagon_ppi.csv'
    dd_path = './BioSNAP/DD-Miner_miner-disease-disease.tsv'
    dg_path = './BioSNAP/DG-AssocMiner_miner-disease-gene.tsv'
    mapping_path = './BioSNAP/disease_mapping.csv'

    ppi_edge_list_path = './BioSNAP/ppi_data.csv'
    dd_edge_list_path = './BioSNAP/dd_data.csv'
    dg_edge_list_path = './BioSNAP/dg_data.csv'
    mapping_data_path = './BioSNAP/mapping_data.csv'

    if ((os.path.exists(ppi_edge_list_path) and os.path.exists(dd_edge_list_path)) and
            (os.path.exists(dg_edge_list_path) and os.path.exists(mapping_data_path))):

        # Read the saved data frame
        ppi_data = pd.read_csv(ppi_edge_list_path)
        dd_data = pd.read_csv(dd_edge_list_path)
        dg_data = pd.read_csv(dg_edge_list_path)
        mapping_data = pd.read_csv(mapping_data_path)

    else:

        # Load datasets
        ppi_data = pd.read_csv(ppi_path, header=None, names=['gid1', 'gid2'])
        dd_data = pd.read_csv(dd_path, sep='\t')
        dg_data = pd.read_csv(dg_path, sep='\t')
        mapping_data = pd.read_csv(mapping_path, sep=',')

        # Unique genes
        unique_genes = pd.concat([ppi_data['gid1'], ppi_data['gid2']]).unique()
        gene_to_index = {gene: idx for idx, gene in enumerate(unique_genes)}

        # Map the genes to indices in the DataFrame
        ppi_data['g1'] = ppi_data['gid1'].map(gene_to_index)
        ppi_data['g2'] = ppi_data['gid2'].map(gene_to_index)

        dd_data.columns = ['did1', 'did2']
        mapping_data.columns = ['DOID', 'Name', 'UML']
        mapping_data['UML'] = mapping_data['UML'].str.replace('UMLS_CUI:', '')
        disease_mapping_dict = mapping_data.set_index('UML')['DOID'].to_dict()

        # Unique diseases
        unique_diseases = pd.concat([dd_data['did1'], dd_data['did2']]).unique()
        disease_to_index = {disease: idx for idx, disease in enumerate(unique_diseases)}

        # Map the genes to indices in the DataFrame
        dd_data['d1'] = dd_data['did1'].map(disease_to_index)
        dd_data['d2'] = dd_data['did2'].map(disease_to_index)

        dd_data['d1_ind'] = dd_data['d1'] + len(unique_genes)
        dd_data['d2_ind'] = dd_data['d2'] + len(unique_genes)

        # Filter the Disease-Gene Data
        dg_data.columns = ['xref', 'dname', 'gid']

        # Use this optimized dictionary to map Disease ID to DOID
        dg_data['did'] = dg_data['xref'].map(disease_mapping_dict)

        # Filter diseases and genes based on what's available in disease-disease and gene-gene networks
        dg_data = dg_data[(dg_data['did'].isin(unique_diseases)) & (dg_data['gid'].isin(unique_genes))]
        # Map the genes to indices in the DataFrame
        dg_data['g'] = dg_data['gid'].map(gene_to_index)
        # print(len(unique_genes))
        # dg_data['d'] = dg_data['did'].map(disease_to_index) + len(unique_genes)
        dg_data['d'] = dg_data['did'].map(disease_to_index)
        # dg_data.reset_index(drop=True, inplace=True)

        ppi_data.to_csv(ppi_edge_list_path, index=False)
        dd_data.to_csv(dd_edge_list_path, index=False)
        dg_data.to_csv(dg_edge_list_path, index=False)
        mapping_data.to_csv(mapping_data_path, index=False)

    return ppi_data, dd_data, dg_data, mapping_data
