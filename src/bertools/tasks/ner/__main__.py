from fire import Fire

from bertools.tasks.ner.train import train


if __name__ == '__main__':
    Fire({"train": train})