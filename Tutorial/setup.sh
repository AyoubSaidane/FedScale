# Clone the repository
git clone https://github.com/SymbioticLab/FedScale.git && \
# Navigate to the cloned directory
cd FedScale && \
# Run the installation script with CUDA support
source install.sh --cuda && \
# Install the package
pip install -e . && \
# Activate the conda environment
conda activate fedscale && \
# Download the FemNIST dataset
bash benchmark/dataset/download.sh download femnist && \
# Successfull installation
echo "Successfully installed FedScale and FemNIST dataset"