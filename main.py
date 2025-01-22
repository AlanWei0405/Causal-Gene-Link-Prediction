from models.heterogeneous.run_onHeteroGraph import run_onheterograph
from models.run_logistic_regression import run_logistic_regression
from utils.build_networks import build_networks
from utils.get_biobert_em import get_biobert_em
from utils.get_disease_summary import create_disease_summary_df
from utils.get_gene_summary import create_gene_summary_df
from utils.load_datasets import load_datasets
from models.homogeneous.run_onSingleGraph import run_onsinglegraph
from synthetic_data import generate_ppi_network_data, generate_dd_network_data, generate_dg_data
from utils.meta2vec import get_features


def main():

    data_mode = "real data"
    # data_mode = "synthetic data"

    if data_mode == "real data":
        ppi_data, dd_data, dg_data, mapping_data = load_datasets()
    else:
        ppi_data = generate_ppi_network_data(1000, edge_density=0.1)
        dd_data = generate_dd_network_data(1000, edge_density=0.1)
        dg_data = generate_dg_data(1000, 1000, edge_density=0.0005)

    epoch = 100
    gene_sum = create_gene_summary_df(ppi_data)
    disease_sum = create_disease_summary_df(dd_data)
    gene_sum, disease_sum = get_biobert_em(gene_sum, disease_sum)

    ppi_nt, dd_nt, dg_nt, dgt_nt, all_nt, hetero_data = build_networks(ppi_data, dd_data, dg_data)

    # run_logistic_regression(hetero_data, epoch)
    # run_onsinglegraph(hetero_data, epoch)
    run_onheterograph(hetero_data, gene_sum, disease_sum, epoch)


if __name__ == "__main__":
    main()
