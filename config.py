squad_cfg = {
    'raw_folder': 'squad',
    'train_file': 'squad_train.json',
    'processed_folder': 'squad_pro',
    'saved_datasets': 'squad_datasets.pt',
    'saved_field': 'squad_field.pt',
    'val_file': 'squad_val.json',
    'language': 'English',
    'context_threshold': 500,
    'batch_size': 64
}

dureader_cfg = {
    'raw_folder': 'dureader',
    'train_file': 'dureader_train.json',
    'processed_folder': 'dureader_pro',
    'saved_datasets': 'dureader_datasets.pt',
    'saved_field': 'dureader_field.pt',
    'language': 'Chinese',
    'context_threshold': 400,
    'val_file': 'dureader_val.json',
    'batch_size': 128
}

model_cfg = {
    'word_emb_size': 200,
    'cxt_emb': 'mt_emb',
    'cxt_emb_size': 200,
    'cxt_emb_pretrained': 'translate-best.th',
    'dropout': 0.2,
}

train_cfg = {
    'lr': 0.5,
    'num_epochs': 50,
    'batch_per_disp': 100,
    'ckpoint_file': 'ckpoint.pt',
    'exp_decay_rate': 0.999
}
