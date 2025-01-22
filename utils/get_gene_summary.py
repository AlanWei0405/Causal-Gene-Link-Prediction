import os
import re
import pandas as pd
from Bio import Entrez
import xml.etree.ElementTree as ET


def fetch_gene_summary(gene_id):
    # Set your email address (required by NCBI)
    Entrez.email = "alanzy.wei@mail.utoronto.ca"
    """Fetches the gene summary from NCBI for a given Gene ID."""
    try:
        # Convert Gene ID to string (if not already)
        gene_id = str(gene_id)
        # Fetch data from the NCBI Gene database
        handle = Entrez.efetch(db="gene", id=gene_id, retmode="xml")
        records = handle.read()
        handle.close()

        # Parse the XML content
        root = ET.fromstring(records)

        # Find the <Entrezgene_summary> tag
        summary = root.find(".//Entrezgene_summary")
        if summary is not None:
            return summary.text.strip()
        else:
            return f"Summary not found for Gene ID {gene_id}"
    except Exception as e:
        return f"Error fetching data for Gene ID {gene_id}: {e}"


def create_gene_summary_df(ppi_data):

    gene_summaries_path = 'gene_summaries.csv'
    if os.path.exists(gene_summaries_path):
        gene_sum = pd.read_csv(gene_summaries_path)

    else:
        """Creates a DataFrame with gene IDs and their corresponding summaries."""
        # Flatten the edge list to extract unique gene IDs
        unique_gene_ids = pd.concat([ppi_data['gid1'], ppi_data['gid2']]).unique()

        # Fetch summaries for all unique gene IDs
        gene_summaries = []
        for index, gene_id in enumerate(unique_gene_ids):
            summary = fetch_gene_summary(gene_id)
            gene_summaries.append({"Gene ID": gene_id, "Summary": summary})
            print(index)

        # Convert to DataFrame
        gene_sum = pd.DataFrame(gene_summaries)
        gene_sum.to_csv("gene_summaries.csv", index=False)
    return gene_sum
