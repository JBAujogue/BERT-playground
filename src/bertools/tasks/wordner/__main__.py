from fire import Fire

from bertools.tasks.wordner.evaluate import evaluate
from bertools.tasks.wordner.train import train

if __name__ == '__main__':
    Fire({
        'train': train,
        'evaluate': evaluate,
    })
