sudo apt install python3-pip

sudo apt-get install libncurses-dev
sudo apt install libyaml-cpp-dev
sudo apt-get install libconfig++-dev
sudo apt-get install libboost-all-dev
sudo apt-get install scons
sudo apt install libgraphviz-dev
sudo apt install graphviz

pip3 install torch torchvision
pip3 install transformers
pip3 install graphviz pygraphviz

# Megatron specific installations
######################################################
#pip3 install megatron-lm
pip3 install git+https://github.com/huggingface/Megatron-LM.git
pip3 install six
pip3 install pybind11
pip3 install ninja
pip3 install Wandb

pip3 install scipy
pip3 install networkx
pip install gurobipy

git clone https://github.com/NVIDIA/apex
cd apex
pip3 install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
cp -r third_party_for_phaze/phaze-megatron-lm/megatron/*  ~/.local/lib/python3.8/site-packages/megatron/

######################################################

git clone https://github.com/Accelergy-Project/accelergy.git
cd accelergy
pip3 install .
accelergy
cd ..

git clone https://github.com/Accelergy-Project/accelergy-cacti-plug-in.git
cd accelergy-cacti-plug-in
pip3 install .

git clone https://github.com/HewlettPackard/cacti.git 
cd cacti
make
export PATH=$(pwd):${PATH}
cd ..

git clone https://github.com/Accelergy-Project/accelergy-table-based-plug-ins.git
cd accelergy-table-based-plug-ins/
pip3 install .
cd ..

git clone https://github.com/Accelergy-Project/accelergy-library-plug-in.git
cd accelergy-library-plug-in/
pip3 install .
cd ..

cp Estimator/arch_configs/area_files/*.csv ~/.local/share/accelergy/estimation_plug_ins/accelergy-table-based-plug-ins/set_of_table_templates/data/.
cp -r Estimator/arch_configs/area_files/tablePluginData ~/.local/share/accelergy/estimation_plug_ins/accelergy-library-plugin/library

cd Solver/device_placement/
g++ device_placement.cpp -o device_placement

export THIRD_PARTY_PATH=$(pwd)/Phaze/third_party_for_phaze
export WHAM_PATH=$THIRD_PARTY_PATH/wham/
export SUNSTONE_PATH=$THIRD_PARTY_PATH/sunstone/
export PYTHONPATH=$THIRD_PARTY_PATH:$WHAM_PATH:$SUNSTONE_PATH:$PYTHONPATH
