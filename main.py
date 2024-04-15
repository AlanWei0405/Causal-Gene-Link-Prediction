from build_networks import build_networks
from load_datasets import load_datasets


def main():
    ppi_data, dd_data, dg_data = load_datasets()
    build_networks(ppi_data, dd_data, dg_data)


if __name__ == "__main__":
    main()
