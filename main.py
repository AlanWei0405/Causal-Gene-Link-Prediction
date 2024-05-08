from build_networks import build_networks
from load_datasets import load_datasets
from run_vgae import run_vgae


def main():
    ppi_data, dd_data, dg_data = load_datasets()
    (ppi_graph, dd_graph, dg_graph,
     ppi_data, dd_data, dg_data, dgt_data) = build_networks(ppi_data, dd_data, dg_data)

    epoch = 100

    adj = run_vgae(ppi_data, dd_data, dg_data, dgt_data, epoch)
    print(adj)


if __name__ == "__main__":
    main()
