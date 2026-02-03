import random
import copy


def get_default_config():
    return {
        "lr": 2.5e-6,
        "weight_decay": 1e-4,
        "batch_size": 64,
        "n2": 4,
        "dropout": 0.0,
        "optimizer": "adam",
        "scheduler": "none",
    }


def mutate_config(config):
    new_config = copy.deepcopy(config)

    mutation_type = random.choice([
        "change_lr",
        "change_weight_decay",
        "change_batch_size",
        "change_n2",
        "change_dropout",
        "change_optimizer",
        "change_scheduler",
    ])

    print(f"  --> Mutation: {mutation_type}")

    if mutation_type == "change_lr":
        factor = random.choice([0.4, 0.6, 0.8, 1.25, 1.5, 2.5])
        new_config["lr"] = new_config["lr"] * factor
        new_config["lr"] = max(1e-7, min(1e-3, new_config["lr"]))

    elif mutation_type == "change_weight_decay":
        new_config["weight_decay"] = random.choice([0.0, 1e-5, 1e-4, 1e-3, 1e-2])

    elif mutation_type == "change_batch_size":
        new_config["batch_size"] = random.choice([32, 48, 64])

    elif mutation_type == "change_n2":
        new_config["n2"] = random.choice([2, 3, 4, 6])

    elif mutation_type == "change_dropout":
        new_config["dropout"] = random.choice([0.0, 0.05, 0.1, 0.15, 0.2])

    elif mutation_type == "change_optimizer":
        new_config["optimizer"] = random.choice(["adam", "adamw"])

    elif mutation_type == "change_scheduler":
        new_config["scheduler"] = random.choice(["none", "cosine", "plateau"])

    return new_config
