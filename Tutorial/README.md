# FedScale Setup

This repository contains commands to set up the FedScale project and download the FemNIST dataset.


## Quick start
To start the quick configuration of fedscale you can open a new terminal and type in this command line:

chmod +x Tutorial/setup.sh && ./Tutorial/setup.sh


## Setup

1. Clone the FedScale repository:

    ```
    git clone https://github.com/SymbioticLab/FedScale.git
    ```

2. Navigate to the cloned directory:

    ```
    cd FedScale
    ```

3. Run the following commands in sequence:

    ```
    source install.sh --cuda
    pip install -e .
    conda activate fedscale
    bash benchmark/dataset/download.sh download femnist
    ```

These commands will install the necessary dependencies, set up the FedScale environment, and download the FemNIST dataset.

## Usage

Once the setup is complete, you can proceed with using the FedScale project for your experiments or research in federated learning.

For more information on how to use FedScale, please refer to the project documentation.

