import os
import csv
from pathlib import Path

def save_returns(episode, ret, fpath: Path = Path('../logs')):

    if not fpath.exists():
        fpath.mkdir(parents=True, exist_ok=True)
    path = fpath / 'returns.csv'

    file_exists = os.path.isfile(str(path))
    with open(path, 'a', newline='') as file:
        writer = csv.writer(file)

        if not file_exists:
            writer.writerow(["episode", "return"])
        writer.writerow([episode, ret])


def save_loss(episode, actor_loss, critic_loss, entropy, fpath: Path = Path('../logs')):

    if not fpath.exists():
        fpath.mkdir(parents=True, exist_ok=True)
    path = fpath / 'loss.csv'

    file_exists = os.path.isfile(str(path))
    with open(path, 'a', newline='') as file:
        writer = csv.writer(file)

        if not file_exists:
            writer.writerow(["episode", 'actor_loss', 'critic_loss', 'entropy'])
        writer.writerow([episode, actor_loss, critic_loss, entropy])



