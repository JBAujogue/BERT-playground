from fire import Fire

from bertools.tasks.mlm.train import train

if __name__ == "__main__":
    Fire({"train": train})
