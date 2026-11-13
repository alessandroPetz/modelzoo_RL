from seanergys_modelzoo.datasets.common.seanergys_dataset import SeanergysDataset
from seanergys_modelzoo.datasets.pm100 import PM100Dataset
from seanergys_modelzoo.datasets.f_data import FDataDataset
from seanergys_modelzoo.dataloader.seanergys_dataloader import SeanergysDataloader
import pandas as pd

def print_n_elements(data_class:SeanergysDataset, n:int = 10):
    for i in range(n):
        print(data_class.__getitem__(idx = i))

if __name__ == "__main__":
    # PM100 Loading Example
    # Load dataset class with parameters (taken from JPCP model)
    data_class = PM100Dataset(
        use_zenodo_url=True, # This will be replaced with different sources from where data can be pulled, e.g. from_data_plane, from_monitoring, etc.
        filters=[
            [
                ("submit_time", ">=", pd.Timestamp("2020-05-01", tz="UTC")),
                ("submit_time", "<=", pd.Timestamp("2020-05-10", tz="UTC")),
            ]
        ],
        columns=[
            "num_nodes_req",
            "user_id",
            "node_power_consumption",
            "num_nodes_alloc",
        ],
        input_features=["num_nodes_req", "user_id"],
        output_features=["node_power_consumption"],
    )

    # Print first 10 elements
    print_n_elements(data_class=data_class)

    # F-DATA Loading Example
    # Load dataset class with parameters (taken from JPCP model)
    data_class = FDataDataset(
            use_zenodo_url=True,
            files=["23_12"],
            filters=[
                ("adt", ">=", "2023-12-01"),
                ("adt", "<=", "2023-12-10"),
            ],
            input_features=["embedding"],
            output_features=["avgpcon", "nnuma"],
        )

    # Print first 10 elements
    print_n_elements(data_class=data_class)

    # M100 Loading Example 1
    # Loading 15-min aggregated dataset of compute node 36.
    data_class =  M100Dataset(
        dataset_path="./data",
        aggregated=True,
        nodes=36,
    )

    # Print first 10 elements
    print_n_elements(data_class=data_class)

    # M100 Loading Example 2
    # Loading raw data for a specific plugin and metric
    data_class =  M100Dataset(
        yy_mm="22-02",
        dataset_path="./data",
        plugins=["ipmi_pub"],
        metrics=["total_power"],
    )

    # Print first 10 elements
    print_n_elements(data_class=data_class)


