python train.py --directed_dataset f --directed f --n_hid_conv 64 --snr 20 --n_hid_mlp 64 --n_layers 3 --nEpochs 100 --batchSize 100 --iterations 5 --lr 0.01
python train.py --directed_dataset f --directed t --n_hid_conv 64 --snr 20 --n_hid_mlp 64 --n_layers 3 --nEpochs 100 --batchSize 100 --iterations 5 --lr 0.01
python train.py --directed_dataset t --directed f --n_hid_conv 64 --snr 20 --n_hid_mlp 64 --n_layers 3 --nEpochs 100 --batchSize 100 --iterations 5 --lr 0.01
python train.py --directed_dataset t --directed t --n_hid_conv 64 --snr 20 --n_hid_mlp 64 --n_layers 3 --nEpochs 100 --batchSize 100 --iterations 5 --lr 0.01
python train_gnn.py --directed_dataset f --directed f --n_hid_conv 32 --snr 20 --n_hid_mlp 32 --n_layers 3 --nEpochs 100 --batchSize 100 --iterations 5 --lr 0.01
python train_gnn.py --directed_dataset f --directed t --n_hid_conv 64 --snr 20 --n_hid_mlp 64 --n_layers 3 --nEpochs 100 --batchSize 100 --iterations 5 --lr 0.01
python train_gnn.py --directed_dataset t --directed f --n_hid_conv 64 --snr 20 --n_hid_mlp 64 --n_layers 3 --nEpochs 100 --batchSize 100 --iterations 5 --lr 0.01
python train_gnn.py --directed_dataset t --directed t --n_hid_conv 64 --snr 20 --n_hid_mlp 64 --n_layers 2 --nEpochs 100 --batchSize 100 --iterations 5 --lr 0.01
