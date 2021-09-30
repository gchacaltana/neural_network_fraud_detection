config = {
    'dataset_source':"data/creditcard.csv",
    'dataset_preprocessed':"data/creditcard_preprocessed.csv",
    'dataset_x_train':"data/dataset_x_train.csv",
    'dataset_x_test':"data/dataset_x_test.csv",
    'dataset_y_train':"data/dataset_y_train.csv",
    'dataset_y_test':"data/dataset_y_test.csv",
    'pct_data_train': 0.8, # 80%
    'pct_data_test': 0.2, # 20%
    'nodes_input': 37,
    'multiplier':1.5,
    'nodes_layer_hidden': 18,
    'training_epochs': 5, # 2000
    'training_dropout': 0.9,
    'display_step': 1, # 10
    'batch_size': 2048,
    'learning_rate': 0.005
}