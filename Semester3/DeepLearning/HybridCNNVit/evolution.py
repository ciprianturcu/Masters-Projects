import random
import copy


def get_default_config():
    return {
        "lr": 1e-4,
        "weight_decay": 1e-4,
        "batch_size": 64,
        "dropout": 0.1,
        "optimizer": "adamw",
        "scheduler": "cosine",
        "cnn_channels": [32, 64, 128, 256, 384],
        "embed_dim": 384,
        "trans_depth": 6,
        "num_heads": 6,
        "mlp_ratio": 4.0,
    }


_VALID_EMBED_DIMS = [192, 256, 384]
_VALID_NUM_HEADS = {192: [3, 4, 6], 256: [4, 8], 384: [4, 6, 8]}


def mutate_config(config):
    new_config = copy.deepcopy(config)
    new_config["architecture_changed"] = False

    mutation_type = random.choice([
        "change_lr",
        "change_weight_decay",
        "change_batch_size",
        "change_dropout",
        "change_optimizer",
        "change_scheduler",
        "change_trans_depth",
        "change_embed_dim",
        "change_num_heads",
        "change_mlp_ratio",
        "change_cnn_channels",
    ])

    print(f"  --> Mutation: {mutation_type}")

    if mutation_type == "change_lr":
        factor = random.choice([0.4, 0.6, 0.8, 1.25, 1.5, 2.5])
        new_config["lr"] = max(1e-6, min(1e-2, new_config["lr"] * factor))

    elif mutation_type == "change_weight_decay":
        new_config["weight_decay"] = random.choice([0.0, 1e-5, 1e-4, 1e-3, 1e-2])

    elif mutation_type == "change_batch_size":
        new_config["batch_size"] = random.choice([32, 48, 64, 96])

    elif mutation_type == "change_dropout":
        new_config["dropout"] = random.choice([0.0, 0.05, 0.1, 0.15, 0.2])

    elif mutation_type == "change_optimizer":
        new_config["optimizer"] = random.choice(["adam", "adamw"])

    elif mutation_type == "change_scheduler":
        new_config["scheduler"] = random.choice(["none", "cosine", "plateau"])

    elif mutation_type == "change_trans_depth":
        new_config["trans_depth"] = random.choice([4, 6, 8])
        new_config["architecture_changed"] = True

    elif mutation_type == "change_embed_dim":
        new_dim = random.choice([d for d in _VALID_EMBED_DIMS if d != new_config["embed_dim"]])
        new_config["embed_dim"] = new_dim
        new_config["cnn_channels"][-1] = new_dim
        valid_heads = _VALID_NUM_HEADS[new_dim]
        if new_config["num_heads"] not in valid_heads:
            new_config["num_heads"] = random.choice(valid_heads)
        new_config["architecture_changed"] = True

    elif mutation_type == "change_num_heads":
        dim = new_config["embed_dim"]
        valid_heads = _VALID_NUM_HEADS.get(dim, [6])
        candidates = [h for h in valid_heads if h != new_config["num_heads"]]
        if candidates:
            new_config["num_heads"] = random.choice(candidates)
            new_config["architecture_changed"] = True

    elif mutation_type == "change_mlp_ratio":
        new_config["mlp_ratio"] = random.choice([2.0, 3.0, 4.0])
        new_config["architecture_changed"] = True

    elif mutation_type == "change_cnn_channels":
        presets = [
            [16, 32, 64, 128],
            [32, 64, 128, 256],
            [48, 96, 192, 384],
            [64, 128, 256, 512],
        ]
        choice = random.choice(presets)
        new_config["cnn_channels"] = choice + [new_config["embed_dim"]]
        new_config["architecture_changed"] = True

    return new_config
