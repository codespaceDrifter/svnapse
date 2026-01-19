import os
import re
import json
import torch
import matplotlib.pyplot as plt

def extract_batch(filename):
    match = re.search(r"batch_(\d+)", filename)
    if match:
        return int(match.group(1))
    return -1

def cleanup_checkpoints(folder, keep_last_n=5):
    files = [f for f in os.listdir(folder) if f.endswith(".pt")]
    files_sorted = sorted(files, key=extract_batch, reverse=True)  # newest first

    for f in files_sorted[keep_last_n:]:
        os.remove(os.path.join(folder, f))

def save_checkpoint(path, model, optimizer, batch, quiz_history, total_train_time, scheduler=None):
    """Atomic save - weights to .pt, everything else to metadata json."""
    temp_path = path + ".tmp"

    # .pt only has weights (assumes compiled model with _orig_mod. prefix)
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()

    torch.save(checkpoint, temp_path)
    os.rename(temp_path, path)

    # all training state goes in metadata json
    folder = os.path.dirname(path)
    meta_dir = os.path.join(folder, "metadata")
    os.makedirs(meta_dir, exist_ok=True)

    # quiz_history: [(batch_idx, {lesson: loss, ...}), ...]
    # convert to json-friendly format
    quiz_history_json = [(b, losses) for b, losses in quiz_history]

    metadata = {
        'batch': batch,
        'quiz_history': quiz_history_json,
        'total_train_time': total_train_time,
    }
    json_path = os.path.join(meta_dir, "training_metadata.json")
    with open(json_path, "w") as f:
        json.dump(metadata, f)

    # line graph with different colors per lesson
    if quiz_history:
        plt.figure(figsize=(10, 6))

        # collect all unique lessons across all entries, preserve order
        all_lessons = []
        for _, losses_dict in quiz_history:
            for lesson in losses_dict:
                if lesson not in all_lessons:
                    all_lessons.append(lesson)

        # consistent colors per lesson (hsv cycles through hues, works for any count)
        cmap = plt.cm.get_cmap('hsv', len(all_lessons) + 1)

        for i, lesson in enumerate(all_lessons):
            # only include points where this lesson was quizzed
            batches = [b for b, losses_dict in quiz_history if lesson in losses_dict]
            losses = [losses_dict[lesson] for _, losses_dict in quiz_history if lesson in losses_dict]
            plt.plot(batches, losses, marker='o', markersize=3, label=lesson, color=cmap(i))

        plt.xlabel("batch")
        plt.ylabel("loss")
        plt.ylim(0, 10)
        plt.title("quiz loss per lesson")
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
        plt.savefig(os.path.join(meta_dir, "loss.png"), dpi=150, bbox_inches='tight')
        plt.close()


def load_latest_checkpoint(folder, model, optimizer=None, scheduler=None):
    """Load checkpoint and training metadata. Returns (batch, quiz_history, total_train_time)."""
    # find .pt files
    files = [f for f in os.listdir(folder) if f.endswith(".pt") and not f.endswith(".tmp")]
    if not files:
        return 0, [], 0.0

    files_sorted = sorted(files, key=extract_batch, reverse=True)

    # load metadata if exists
    json_path = os.path.join(folder, "metadata", "training_metadata.json")
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            metadata = json.load(f)
        batch = metadata.get('batch', 0)
        quiz_history = [(b, losses) for b, losses in metadata.get('quiz_history', [])]
        total_train_time = metadata.get('total_train_time', 0.0)
    else:
        # no metadata - extract batch from filename, reset quiz history
        batch = extract_batch(files_sorted[0])
        quiz_history = []
        total_train_time = 0.0

    # load weights
    for checkpoint_file in files_sorted[:2]:
        try:
            path = os.path.join(folder, checkpoint_file)
            print(f"Loading checkpoint: {checkpoint_file}")
            checkpoint = torch.load(path)

            model.load_state_dict(checkpoint['model_state_dict'])
            if optimizer and 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if scheduler and 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

            return batch, quiz_history, total_train_time

        except Exception as e:
            print(f"Error loading {checkpoint_file}: {e}")
            continue

    return 0, [], 0.0
