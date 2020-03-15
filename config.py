squad_cfg = {
    'raw_folder': 'squad',
    'train_file': 'train-v1.1.json',
    'processed_folder': 'squad_pro',
    'saved_datasets': 'squad_datasets.pt',
    'saved_field': 'squad_field.pt',
    'val_file': 'dev-v1.1.json',
    'language': 'English',
    'context_threshold': 500,
    'batch_size': 64
}

dureader_cfg = {
    'raw_folder': 'dureader',
    'train_file': None,
    'processed_folder': 'dureader_pro',
    'saved_datasets': 'dureader_datasets.pt',
    'saved_field': 'dureader_field.pt',
    'language': 'Chinese',
    'context_threshold': 500,
    'val_file': None,
    'batch_size': 64
}

model_cfg = {
    'cxt_emb': None,
    'hidden_dim': 100 + 0,
    'dropout': 0.2
}

train_cfg = {
    'lr': 3e-4,
    'num_epochs': 10,
    'batch_per_disp': 100,
}
