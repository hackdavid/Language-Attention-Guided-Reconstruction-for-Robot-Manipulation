## C5_ema_teacher_masks_20260410T130021_10ba0d4a
{
  "_runtime": 13291.800713297,
  "_step": 6189,
  "_timestamp": 1775839314.0573895,
  "_wandb.runtime": 13291,
  "checkpoint/saved_best": 1,
  "epoch": 3,
  "train/action_loss": 0.2451026439666748,
  "train/best_metric": 0.2435416785536954,
  "train/epoch_mean_loss": 0.2435416785536954,
  "train/recon_loss": 0.05228385701775551,
  "train/total_loss": 0.2712445855140686,
  "val/action_mae_dim_0": 0.8341502743959427,
  "val/action_mae_dim_1": 0.7923856174945831,
  "val/action_mae_dim_2": 0.8138385879993438,
  "val/action_mae_dim_3": 0.7958601075410843,
  "val/action_mae_dim_4": 0.8044362449645996,
  "val/action_mae_dim_5": 0.7846895134449006,
  "val/action_mae_dim_6": 0.8393954074382782,
  "val/action_mae_mean": 0.8092508218969617,
  "val/total_loss": 1.084127711057663
}

config : 
{
  "data": {
    "value": {
      "libero": {
        "seed": 42,
        "shuffle": true,
        "batch_size": 8,
        "pin_memory": true,
        "num_workers": 2,
        "skip_snapshot_download": true
      }
    }
  },
  "model": {
    "value": {
      "ema": {
        "decay": 0.999,
        "enabled": true
      },
      "masking": {
        "mode": "attention_selected",
        "topology": "default",
        "mask_ratio": 0.25,
        "mask_source": "ema_teacher",
        "selected_heads": [
          0,
          1,
          2
        ],
        "attention_heads": "all",
        "attention_layers": "last_3",
        "head_selection_file": null
      },
      "backbone": {
        "model_id": "/kaggle/input/models/google/paligemma-2/transformers/paligemma2-3b-mix-224/1",
        "device_map": null,
        "torch_dtype": "bfloat16"
      },
      "patch_size": 14,
      "num_patches": 256,
      "reconstruction": {
        "enabled": true,
        "decoder_dim": 256,
        "lambda_recon": 0.5,
        "decoder_heads": 8,
        "decoder_layers": 4
      },
      "freeze_backbone": true,
      "finetune_last_n_layers": 2
    }
  },
  "_wandb": {
    "value": {
      "m": [],
      "t": {
        "1": [
          1,
          5,
          11,
          41,
          49,
          51,
          53,
          71,
          105
        ],
        "2": [
          1,
          5,
          11,
          41,
          49,
          51,
          53,
          71,
          105
        ],
        "3": [
          2,
          13,
          15,
          16,
          61
        ],
        "4": "3.12.12",
        "5": "0.24.2",
        "6": "5.0.0",
        "8": [
          2
        ],
        "12": "0.24.2",
        "13": "linux-x86_64"
      },
      "cli_version": "0.24.2",
      "python_version": "3.12.12"
    }
  },
  "logging": {
    "value": {
      "wandb": {
        "tags": [
          "C5"
        ],
        "enabled": true,
        "project": "La-ReconVLA"
      }
    }
  },
  "training": {
    "value": {
      "seed": 42,
      "device": "cuda",
      "epochs": 3,
      "batch_size": 16,
      "resume_from": null,
      "val_batches": 50,
      "weight_decay": 0.01,
      "learning_rate": 0.0001,
      "max_grad_norm": 1,
      "checkpoint_dir": "./checkpoints/C5",
      "mixed_precision": true,
      "batches_per_epoch": 500,
      "log_every_n_steps": 100,
      "use_experiment_preset": false,
      "best_checkpoint_metric": "train_loss"
    }
  },
  "experiment": {
    "value": {
      "name": "C5_ema_teacher_masks",
      "notes": "",
      "condition": "C5"
    }
  }
}

## C4_selected_heads_mae_20260408T134240_8c94e40b
{
  "_runtime": 8401.73958682,
  "_step": 6189,
  "_timestamp": 1775664162.7764354,
  "_wandb.runtime": 8401,
  "checkpoint/saved_best": 1,
  "epoch": 3,
  "train/action_loss": 0.2451036125421524,
  "train/best_metric": 0.2441725844586872,
  "train/epoch_mean_loss": 0.2441725844586872,
  "train/recon_loss": 0.05213457718491554,
  "train/total_loss": 0.2711709141731262,
  "val/action_mae_dim_0": 0.8341497403383255,
  "val/action_mae_dim_1": 0.7923859226703643,
  "val/action_mae_dim_2": 0.813838642835617,
  "val/action_mae_dim_3": 0.7958602410554886,
  "val/action_mae_dim_4": 0.8044363021850586,
  "val/action_mae_dim_5": 0.7846893799304963,
  "val/action_mae_dim_6": 0.8393953311443328,
  "val/action_mae_mean": 0.8092507943085262,
  "val/total_loss": 1.085895665884018
}

config : 
{
  "data": {
    "value": {
      "libero": {
        "seed": 42,
        "shuffle": true,
        "batch_size": 8,
        "pin_memory": true,
        "num_workers": 2,
        "skip_snapshot_download": true
      }
    }
  },
  "model": {
    "value": {
      "ema": {
        "decay": 0.999,
        "enabled": false
      },
      "masking": {
        "mode": "attention_selected",
        "topology": "default",
        "mask_ratio": 0.25,
        "mask_source": "student",
        "selected_heads": [
          0,
          1,
          2
        ],
        "attention_heads": "all",
        "attention_layers": "last_3",
        "head_selection_file": null
      },
      "backbone": {
        "model_id": "/kaggle/input/models/google/paligemma-2/transformers/paligemma2-3b-mix-224/1",
        "device_map": null,
        "torch_dtype": "bfloat16"
      },
      "patch_size": 14,
      "num_patches": 256,
      "reconstruction": {
        "enabled": true,
        "decoder_dim": 256,
        "lambda_recon": 0.5,
        "decoder_heads": 8,
        "decoder_layers": 4
      },
      "freeze_backbone": true,
      "finetune_last_n_layers": 2
    }
  },
  "_wandb": {
    "value": {
      "m": [],
      "t": {
        "1": [
          1,
          5,
          11,
          41,
          49,
          51,
          53,
          71,
          105
        ],
        "2": [
          1,
          5,
          11,
          41,
          49,
          51,
          53,
          71,
          105
        ],
        "3": [
          2,
          13,
          15,
          16,
          61
        ],
        "4": "3.12.12",
        "5": "0.24.2",
        "6": "5.0.0",
        "8": [
          2
        ],
        "12": "0.24.2",
        "13": "linux-x86_64"
      },
      "cli_version": "0.24.2",
      "python_version": "3.12.12"
    }
  },
  "logging": {
    "value": {
      "wandb": {
        "tags": [
          "C4"
        ],
        "enabled": true,
        "project": "La-ReconVLA"
      }
    }
  },
  "training": {
    "value": {
      "seed": 42,
      "device": "cuda",
      "epochs": 3,
      "batch_size": 16,
      "resume_from": null,
      "val_batches": 50,
      "weight_decay": 0.01,
      "learning_rate": 0.0001,
      "max_grad_norm": 1,
      "checkpoint_dir": "./checkpoints/C4",
      "mixed_precision": true,
      "batches_per_epoch": 500,
      "log_every_n_steps": 100,
      "use_experiment_preset": false,
      "best_checkpoint_metric": "train_loss"
    }
  },
  "experiment": {
    "value": {
      "name": "C4_selected_heads_mae",
      "notes": "",
      "condition": "C4"
    }
  }
}


## C3_attention_naive_mae_20260408T111502_24042fa7
{
  "_runtime": 8485.747161073,
  "_step": 8253,
  "_timestamp": 1775655388.913484,
  "_wandb.runtime": 8485,
  "checkpoint/saved_best": 1,
  "epoch": 3,
  "train/action_loss": 0.23897352814674375,
  "train/best_metric": 0.24373097305014457,
  "train/epoch_mean_loss": 0.24373097305014457,
  "train/recon_loss": 0.04842156544327736,
  "train/total_loss": 0.2631843090057373,
  "val/action_mae_dim_0": 0.8342397671937942,
  "val/action_mae_dim_1": 0.792891446352005,
  "val/action_mae_dim_2": 0.813940167427063,
  "val/action_mae_dim_3": 0.7958690625429153,
  "val/action_mae_dim_4": 0.8044415855407715,
  "val/action_mae_dim_5": 0.7847019684314728,
  "val/action_mae_dim_6": 0.8401070439815521,
  "val/action_mae_mean": 0.809455863067082,
  "val/total_loss": 1.0849330806732178
}

config: 
{
  "data": {
    "value": {
      "libero": {
        "seed": 42,
        "shuffle": true,
        "batch_size": 6,
        "pin_memory": true,
        "num_workers": 2,
        "skip_snapshot_download": true
      }
    }
  },
  "model": {
    "value": {
      "ema": {
        "decay": 0.999,
        "enabled": false
      },
      "masking": {
        "mode": "attention_naive",
        "topology": "default",
        "mask_ratio": 0.25,
        "mask_source": "student",
        "selected_heads": null,
        "attention_heads": "all",
        "attention_layers": "last_3",
        "head_selection_file": null
      },
      "backbone": {
        "model_id": "/kaggle/input/models/google/paligemma-2/transformers/paligemma2-3b-mix-224/1",
        "device_map": null,
        "torch_dtype": "bfloat16"
      },
      "patch_size": 14,
      "num_patches": 256,
      "reconstruction": {
        "enabled": true,
        "decoder_dim": 256,
        "lambda_recon": 0.5,
        "decoder_heads": 8,
        "decoder_layers": 4
      },
      "freeze_backbone": true,
      "finetune_last_n_layers": 2
    }
  },
  "_wandb": {
    "value": {
      "m": [],
      "t": {
        "1": [
          1,
          5,
          11,
          41,
          49,
          51,
          53,
          71,
          105
        ],
        "2": [
          1,
          5,
          11,
          41,
          49,
          51,
          53,
          71,
          105
        ],
        "3": [
          2,
          13,
          15,
          16,
          61
        ],
        "4": "3.12.12",
        "5": "0.24.2",
        "6": "5.0.0",
        "8": [
          2
        ],
        "12": "0.24.2",
        "13": "linux-x86_64"
      },
      "cli_version": "0.24.2",
      "python_version": "3.12.12"
    }
  },
  "logging": {
    "value": {
      "wandb": {
        "tags": [
          "C3"
        ],
        "enabled": true,
        "project": "La-ReconVLA"
      }
    }
  },
  "training": {
    "value": {
      "seed": 42,
      "device": "cuda",
      "epochs": 3,
      "batch_size": 16,
      "resume_from": null,
      "val_batches": 50,
      "weight_decay": 0.01,
      "learning_rate": 0.0001,
      "max_grad_norm": 1,
      "checkpoint_dir": "./checkpoints/C3",
      "mixed_precision": true,
      "batches_per_epoch": 500,
      "log_every_n_steps": 100,
      "use_experiment_preset": false,
      "best_checkpoint_metric": "train_loss"
    }
  },
  "experiment": {
    "value": {
      "name": "C3_attention_naive_mae",
      "notes": "",
      "condition": "C3"
    }
  }
}

## C2_random_mae_20260407T203403_df19b5d9
{
  "_runtime": 8262.691370872,
  "_step": 6189,
  "_timestamp": 1775602307.3007133,
  "_wandb.runtime": 8262,
  "checkpoint/saved_best": 1,
  "epoch": 3,
  "train/action_loss": 0.2445082813501358,
  "train/best_metric": 0.24305064939377785,
  "train/epoch_mean_loss": 0.24305064939377785,
  "train/recon_loss": 0.04636845365166664,
  "train/total_loss": 0.267692506313324,
  "val/action_mae_dim_0": 0.8111789703369141,
  "val/action_mae_dim_1": 0.7859559035301209,
  "val/action_mae_dim_2": 0.8210531461238861,
  "val/action_mae_dim_3": 0.7972615051269532,
  "val/action_mae_dim_4": 0.7865269184112549,
  "val/action_mae_dim_5": 0.8074352896213531,
  "val/action_mae_dim_6": 0.8246913909912109,
  "val/action_mae_mean": 0.8048718748773848,
  "val/total_loss": 1.0671397185325622
}

config : 
{
  "data": {
    "value": {
      "libero": {
        "seed": 42,
        "shuffle": true,
        "batch_size": 8,
        "pin_memory": true,
        "num_workers": 4,
        "skip_snapshot_download": true
      }
    }
  },
  "model": {
    "value": {
      "ema": {
        "decay": 0.999,
        "enabled": false
      },
      "masking": {
        "mode": "random",
        "topology": "default",
        "mask_ratio": 0.25,
        "mask_source": "student",
        "selected_heads": null,
        "attention_heads": "all",
        "attention_layers": "last_3",
        "head_selection_file": null
      },
      "backbone": {
        "model_id": "/kaggle/input/models/google/paligemma-2/transformers/paligemma2-3b-mix-224/1",
        "device_map": null,
        "torch_dtype": "bfloat16"
      },
      "patch_size": 14,
      "num_patches": 256,
      "reconstruction": {
        "enabled": true,
        "decoder_dim": 256,
        "lambda_recon": 0.5,
        "decoder_heads": 8,
        "decoder_layers": 4
      },
      "freeze_backbone": true,
      "finetune_last_n_layers": 2
    }
  },
  "_wandb": {
    "value": {
      "m": [],
      "t": {
        "1": [
          1,
          5,
          11,
          41,
          49,
          51,
          53,
          71,
          105
        ],
        "2": [
          1,
          5,
          11,
          41,
          49,
          51,
          53,
          71,
          105
        ],
        "3": [
          2,
          13,
          15,
          16,
          61
        ],
        "4": "3.12.12",
        "5": "0.24.2",
        "6": "5.0.0",
        "8": [
          2
        ],
        "12": "0.24.2",
        "13": "linux-x86_64"
      },
      "cli_version": "0.24.2",
      "python_version": "3.12.12"
    }
  },
  "logging": {
    "value": {
      "wandb": {
        "tags": [
          "C2"
        ],
        "enabled": true,
        "project": "La-ReconVLA"
      }
    }
  },
  "training": {
    "value": {
      "seed": 42,
      "device": "cuda",
      "epochs": 3,
      "batch_size": 32,
      "resume_from": null,
      "val_batches": 50,
      "weight_decay": 0.01,
      "learning_rate": 0.0001,
      "max_grad_norm": 1,
      "checkpoint_dir": "./checkpoints/C2",
      "mixed_precision": true,
      "batches_per_epoch": 500,
      "log_every_n_steps": 10,
      "use_experiment_preset": false,
      "best_checkpoint_metric": "train_loss"
    }
  },
  "experiment": {
    "value": {
      "name": "C2_random_mae",
      "notes": "",
      "condition": "C2"
    }
  }
}

## C1_action_only_20260407T141132_0089169c
{
  "_runtime": 6201.90889906,
  "_step": 5930,
  "_timestamp": 1775577285.9402173,
  "_wandb.runtime": 6201,
  "checkpoint/saved_best": 1,
  "epoch": 2,
  "train/action_loss": 0.23353971540927887,
  "train/best_metric": 0.21908093206733412,
  "train/epoch_mean_loss": 0.21908093206733412,
  "train/recon_loss": 0,
  "train/total_loss": 0.23353971540927887,
  "val/action_mae_dim_0": 0.8650337618589401,
  "val/action_mae_dim_1": 0.8129968786239624,
  "val/action_mae_dim_2": 0.8088088274002075,
  "val/action_mae_dim_3": 0.791849792599678,
  "val/action_mae_dim_4": 0.7664434707164764,
  "val/action_mae_dim_5": 0.7708389097452164,
  "val/action_mae_dim_6": 0.8421022421121598,
  "val/action_mae_mean": 0.8082962690080916,
  "val/total_loss": 1.0255842328071594
}

config : 
{
  "data": {
    "value": {
      "libero": {
        "seed": 42,
        "shuffle": true,
        "batch_size": 6,
        "local_root": "/kaggle/working/my_local_libero_data",
        "pin_memory": true,
        "num_workers": 4,
        "skip_snapshot_download": true
      }
    }
  },
  "model": {
    "value": {
      "ema": {
        "decay": 0.999,
        "enabled": false
      },
      "masking": {
        "mode": "none",
        "topology": "default",
        "mask_ratio": 0.25,
        "mask_source": "student",
        "selected_heads": null,
        "attention_heads": "all",
        "attention_layers": "last_3",
        "head_selection_file": null
      },
      "backbone": {
        "model_id": "/kaggle/input/models/google/paligemma-2/transformers/paligemma2-3b-mix-224/1",
        "device_map": null,
        "torch_dtype": "bfloat16"
      },
      "patch_size": 16,
      "num_patches": 196,
      "reconstruction": {
        "enabled": false,
        "decoder_dim": 256,
        "lambda_recon": 0,
        "decoder_heads": 8,
        "decoder_layers": 4
      },
      "freeze_backbone": true,
      "finetune_last_n_layers": 2
    }
  },
  "_wandb": {
    "value": {
      "m": [],
      "t": {
        "1": [
          1,
          5,
          11,
          41,
          49,
          51,
          53,
          71,
          105
        ],
        "2": [
          1,
          5,
          11,
          41,
          49,
          51,
          53,
          71,
          105
        ],
        "3": [
          13,
          15,
          16,
          61
        ],
        "4": "3.12.12",
        "5": "0.24.2",
        "6": "5.0.0",
        "8": [
          2
        ],
        "12": "0.24.2",
        "13": "linux-x86_64"
      },
      "cli_version": "0.24.2",
      "python_version": "3.12.12"
    }
  },
  "logging": {
    "value": {
      "wandb": {
        "tags": [
          "C1"
        ],
        "enabled": true,
        "project": "La-ReconVLA"
      }
    }
  },
  "training": {
    "value": {
      "seed": 42,
      "device": "cuda",
      "epochs": 20,
      "batch_size": 16,
      "resume_from": null,
      "val_batches": 50,
      "weight_decay": 0.01,
      "learning_rate": 0.0001,
      "max_grad_norm": 1,
      "checkpoint_dir": "./checkpoints/C1",
      "mixed_precision": true,
      "batches_per_epoch": 500,
      "log_every_n_steps": 10,
      "use_experiment_preset": false,
      "best_checkpoint_metric": "train_loss"
    }
  },
  "experiment": {
    "value": {
      "name": "C1_action_only",
      "notes": "",
      "condition": "C1"
    }
  }
}