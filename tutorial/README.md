# FedScale Setup

This repository contains commands to set up the FedScale project and download the FemNIST dataset.


## Quick start
To start the quick configuration of fedscale you can download [tutorial/setup.sh](https://github.com/AyoubSaidane/FedScale/edit/master/tutorial/setup.sh) open a new terminal in the same directory as setup.sh and type in this command line:

```
chmod +x setup.sh && ./setup.sh
```

## Setup

1. Clone the FedScale repository:

    ```
    git clone https://github.com/AyoubSaidane/FedScale.git
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

### Configuration

Change the training and configuration parameters in [conf.yml](https://github.com/AyoubSaidane/FedScale/blob/master/benchmark/configs/femnist/conf.yml), by specifying the IP address of server and clients and their number. You can change any training parameter too.  

### Execution

Once the setup is complete, you can proceed with using the FedScale project for your experiments or research in federated learning.
To train from Femnist dataset run this command line:
```
python docker/driver.py start benchmark/configs/femnist/conf.yml
```
To stop the training process run this command:
```
python docker/driver.py stop femnist
```
And to visualize the training results using Tensorboard use this command:
```
%load_ext tensorboard
%tensorboard --logdir /content/FedScale/benchmark/logs/femnist
```

For more information on how to use FedScale, please refer to the project documentation.

