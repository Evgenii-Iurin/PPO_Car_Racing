import cv2
import os
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def visualize_rollout(storage, path):
    """
    Save rollouts (frame by frame) from the storage
    """
    for ind, frame in enumerate(storage.obs):
        print(f"reward {ind} : {storage.rewards[ind]}")
        actual_frame = frame[3::]
        frame_np = actual_frame.numpy().squeeze()
        sample = frame_np * 255
        save_path = Path(path) / f"sample_{ind}.jpg"
        cv2.imwrite(str(save_path), sample)


def convert_rollouts_to_gif(image_folder, output_gif = "../rollout.gif"):
    images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
    images.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
    frames = [Image.open(os.path.join(image_folder, img)) for img in images]
    frames[0].save(output_gif, format='GIF', append_images=frames[1:], save_all=True, duration=70, loop=0)


def plot_return_values(file: str, save_path: str):
    df = pd.read_csv(file)

    plt.figure(figsize=(10, 6))
    plt.plot(df['episode'], df['return'], color='blue', linestyle='-')
    plt.title('Reward')
    plt.xlabel('Episode')
    plt.ylabel('Return')
    plt.grid(False)
    plt.savefig(save_path)
    print(f"Figure was saved to {save_path}")


def plot_losses(csv_file, save_path):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file)

    # Extracting columns
    episodes = df['episode']
    actor_loss = df['actor_loss']
    critic_loss = df['critic_loss']
    entropy = df['entropy']

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(episodes, actor_loss, label='Actor Loss', color='blue', marker='o', linestyle='-')
    plt.plot(episodes, critic_loss, label='Critic Loss', color='red', marker='s', linestyle='-')
    plt.plot(episodes, entropy, label='Entropy', color='green', marker='^', linestyle='-')

    plt.title('Training Metrics')
    plt.xlabel('Episode')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(False)
    plt.savefig(save_path)

    print(f"Plot was saved to {save_path}")


def plot_actor_loss(file: str, save_path: str):
    df = pd.read_csv(file)

    plt.figure(figsize=(10, 6))
    plt.plot(df['episode'], df['actor_loss'], color='blue', linestyle='-')
    plt.title('Actor loss')
    plt.xlabel('Episode')
    plt.ylabel('Actor Loss')
    plt.grid(False)
    plt.savefig(save_path)
    print(f"Figure was saved to {save_path}")


def plot_critic_loss(file: str, save_path: str):
    df = pd.read_csv(file)

    plt.figure(figsize=(10, 6))
    plt.plot(df['episode'], df['critic_loss'], color='blue', linestyle='-')
    plt.title('Critic loss')
    plt.xlabel('Episode')
    plt.ylabel('Critic Loss')
    plt.grid(False)
    plt.savefig(save_path)
    print(f"Figure was saved to {save_path}")


def plot_entropy(file: str, save_path: str):
    df = pd.read_csv(file)

    plt.figure(figsize=(10, 6))
    plt.plot(df['episode'], df['entropy'], color='blue', linestyle='-')
    plt.title('Entropy')
    plt.xlabel('Episode')
    plt.ylabel('Entropy')
    plt.grid(False)
    plt.savefig(save_path)
    print(f"Figure was saved to {save_path}")

if __name__ == "__main__":

    exp = "exp_7"
    path = f"result_exp/{exp}/logs/loss.csv"
    path_return = f"result_exp/{exp}/logs/returns.csv"
    save_path = f"result_exp/{exp}/"

    plot_entropy(path, save_path + "entropy.jpg")
    plot_critic_loss(path, save_path + "critic_loss.jpg")
    plot_actor_loss(path, save_path + "actor_loss.jpg")
    plot_losses(path, save_path + "losses.jpg")

    plot_return_values(path_return, save_path + "return_value.jpg")


    # convert_rollouts_to_gif("eval_replays/exp_7/rollout_3", "rollout_exp_7_3.gif")
