# curriculum learning - each bin file is a lesson
# mix adjusted based on per-lesson eval loss

import time
import random
import torch
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader, Sampler

from utils.training.saves import cleanup_checkpoints, load_latest_checkpoint, save_checkpoint


class BinTokenDataset(Dataset):
    def __init__(self, bin_path, seq_len):
        self.data = np.memmap(bin_path, dtype=np.int32, mode='r')
        self.seq_len = seq_len
        self.num_samples = len(self.data) // seq_len

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        start = idx * self.seq_len
        # (seq_len,) int64
        return torch.from_numpy(self.data[start:start + self.seq_len].copy()).long()


class CurriculumDataset(Dataset):
    """
    Combined dataset from multiple lesson .bin files.
    Example with 2 lessons: {"story": 100 samples, "code": 50 samples}
      lessons = ["story", "code"]
      sizes = {"story": 100, "code": 50}
      boundaries = [0, 100, 150]  -> story is idx 0-99, code is idx 100-149
    """
    def __init__(self, lesson_bin_paths, seq_len):
        # ["story", "code", ...] - ordered list of lesson names
        self.lessons = list(lesson_bin_paths.keys())
        # {"story": BinTokenDataset, "code": BinTokenDataset, ...}
        self.datasets = {l: BinTokenDataset(p, seq_len) for l, p in lesson_bin_paths.items()}
        # {"story": 100, "code": 50, ...} - num samples per lesson
        self.sizes = {l: len(ds) for l, ds in self.datasets.items()}

        # [0, 100, 150] - cumulative boundaries for global idx -> lesson lookup
        self.boundaries = [0]
        for l in self.lessons:
            self.boundaries.append(self.boundaries[-1] + self.sizes[l])

    def __len__(self):
        return self.boundaries[-1]

    def __getitem__(self, idx):
        # find which lesson this idx belongs to
        for i, l in enumerate(self.lessons):
            if idx < self.boundaries[i + 1]:
                local_idx = idx - self.boundaries[i]
                return self.datasets[l][local_idx]


class CurriculumSampler(Sampler):
    """
    Infinite sampler - picks lesson first (based on mix), then random sample within lesson.
    Rebuilt when mix changes after quiz.
    """
    def __init__(self, dataset, mix):
        self.dataset = dataset
        # [("story", 0.7), ("code", 0.3), ...] - lessons with non-zero mix
        self.lesson_weights = [(l, mix[l]) for l in dataset.lessons if mix[l] > 0]

    def __iter__(self):
        lessons = [l for l, _ in self.lesson_weights]
        weights = [w for _, w in self.lesson_weights]

        while True:
            # pick lesson based on mix probabilities
            lesson = random.choices(lessons, weights=weights, k=1)[0]
            # pick random sample within that lesson
            lesson_idx = self.dataset.lessons.index(lesson)
            start = self.dataset.boundaries[lesson_idx]
            end = self.dataset.boundaries[lesson_idx + 1]
            yield random.randint(start, end - 1)


def make_curriculum_loader(dataset, mix, minibatch_size):
    """Build DataLoader with CurriculumSampler from mix ratios."""
    sampler = CurriculumSampler(dataset, mix)
    return DataLoader(dataset, batch_size=minibatch_size, sampler=sampler, num_workers=2, pin_memory=True)


def compute_mix_ratios(quiz_losses, acceptable_loss, priority_order):
    """
    raw_ratio = (loss - acceptable) / acceptable, capped [0, 1]
    fill in priority order until sum = 1.0
    scale up if total < 1.0

    Returns (mix, all_passed) where all_passed is True if all lessons < acceptable_loss
    """
    all_passed = all(loss < acceptable_loss for loss in quiz_losses.values())

    # if all passed, uniform mix (keep training everything equally)
    if all_passed:
        n = len(priority_order)
        mix = {lesson: 1.0 / n for lesson in priority_order}
        return mix, True

    raw = {}
    for lesson, loss in quiz_losses.items():
        r = (loss - acceptable_loss) / acceptable_loss
        raw[lesson] = max(0.0, min(1.0, r))

    mix = {lesson: 0.0 for lesson in priority_order}
    remaining = 1.0

    for lesson in priority_order:
        if remaining <= 0:
            break
        take = min(raw[lesson], remaining)
        mix[lesson] = take
        remaining -= take

    total = sum(mix.values())
    if 0< total < 1.0:
        scale = 1.0 / total
        mix = {k: v * scale for k, v in mix.items()}

    return mix, False


def quiz_lessons(model, dataset, batch_size, quiz_batches):
    """Sample loss on each lesson from training data, return dict of losses."""
    model.eval()
    quiz_losses = {}

    with torch.no_grad():
        for lesson in dataset.lessons:
            ds = dataset.datasets[lesson]
            loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

            total_loss = 0.0
            count = 0
            for tokens in loader:
                if count >= quiz_batches:
                    break
                # (batch, seq_len)
                tokens = tokens.cuda()
                _, loss = model(tokens, labels=tokens)
                total_loss += loss.item()
                count += 1

            assert count > 0
            quiz_losses[lesson] = total_loss / count

    model.train()
    return quiz_losses


def train_curriculum(
    lesson_bin_paths,       # {"lesson1": "path.bin", "lesson2": "path.bin", ...}
    priority_order,         # ["lesson1", "lesson2", ...] - order to fill mix
    model,
    optimizer,
    batch_size,
    seq_len,
    save_folder_path,
    acceptable_loss=3.0,
    stop_streak=10,         # stop after all lessons < acceptable_loss for this many quizzes
    tokenizer=None,
    accumulation_steps=8,
    scheduler=None,
    clip_grad_norm=1.0,
    batches_per_save=100,
    batches_per_log=10,
    batches_per_quiz=100,
    quiz_batches=5,
):
    assert batch_size % accumulation_steps == 0
    minibatch_size = batch_size // accumulation_steps

    model.cuda().bfloat16()
    os.makedirs(save_folder_path, exist_ok=True)

    # combined dataset
    dataset = CurriculumDataset(lesson_bin_paths, seq_len)
    for l in dataset.lessons:
        print(f"{l}: {dataset.sizes[l]} samples")
    print(f"Total: {len(dataset)} samples")

    # checkpoint
    # quiz_history: [(batch_idx, {lesson: loss, ...}), ...]
    last_batch, quiz_history, prev_train_time = load_latest_checkpoint(
        save_folder_path, model, optimizer, scheduler
    )
    batch_idx = last_batch
    session_start_time = time.time()

    # streak counter - how many consecutive quizzes with all lessons below acceptable_loss
    streak = 0

    # initial quiz
    print("\n=== Initial quiz ===")
    quiz_losses = quiz_lessons(model, dataset, minibatch_size, quiz_batches)
    for l, loss in quiz_losses.items():
        print(f"  {l}: {loss:.4f}")
    quiz_history.append((batch_idx, quiz_losses.copy()))

    mix, _ = compute_mix_ratios(quiz_losses, acceptable_loss, priority_order)
    print(f"Initial mix: {mix}")

    model.train()
    loader = make_curriculum_loader(dataset, mix, minibatch_size)
    minibatch_iter = iter(loader)
    batches_since_quiz = 0

    while True:
        optimizer.zero_grad()
        acc_loss = 0.0

        for _ in range(accumulation_steps):
            # (minibatch, seq_len)
            tokens = next(minibatch_iter).cuda()
            logits, loss = model(tokens, labels=tokens)
            (loss / accumulation_steps).backward()
            acc_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm)
        optimizer.step()
        if scheduler:
            scheduler.step()

        batch_idx += 1
        batches_since_quiz += 1
        avg_loss = acc_loss / accumulation_steps

        # logging
        if batch_idx % batches_per_log == 0:
            lr = optimizer.param_groups[0]['lr']

            def fmt(s):
                d, h, m, sec = int(s // 86400), int((s % 86400) // 3600), int((s % 3600) // 60), int(s % 60)
                p = []
                if d: p.append(f"{d}d")
                if h: p.append(f"{h}h")
                if m: p.append(f"{m}m")
                p.append(f"{sec}s")
                return " ".join(p)

            elapsed = time.time() - session_start_time
            total_time = prev_train_time + elapsed
            print(f"Time: {fmt(total_time)} | Batch {batch_idx} | Loss: {avg_loss:.4f} | LR: {lr:.2e}")
            print(f"Mix: {mix}")

            if tokenizer:
                pred = logits[0, :50].argmax(dim=-1).tolist()
                tgt = tokens[0, 1:51].tolist()
                print(f"  Target: {repr(tokenizer.decode(tgt)[:100])}")
                print(f"  Pred:   {repr(tokenizer.decode(pred)[:100])}")

        # re-quiz and adjust mix
        if batches_since_quiz >= batches_per_quiz:
            print("\n--- Quiz ---")
            quiz_losses = quiz_lessons(model, dataset, minibatch_size, quiz_batches)
            for l, loss in quiz_losses.items():
                print(f"  {l}: {loss:.4f}")
            quiz_history.append((batch_idx, quiz_losses.copy()))

            mix, all_passed = compute_mix_ratios(quiz_losses, acceptable_loss, priority_order)

            if all_passed:
                streak += 1
                print(f"All lessons passed! Streak: {streak}/{stop_streak}")
                if streak >= stop_streak:
                    print(f"\nTraining complete - {stop_streak} consecutive passes!")
                    # final save
                    path = f"{save_folder_path}/batch_{batch_idx}.pt"
                    total_time = prev_train_time + (time.time() - session_start_time)
                    save_checkpoint(path, model, optimizer, batch_idx, quiz_history, total_time, scheduler)
                    return
            else:
                if streak > 0:
                    print(f"Streak reset (was {streak})")
                streak = 0

            print(f"New mix: {mix}")

            loader = make_curriculum_loader(dataset, mix, minibatch_size)
            minibatch_iter = iter(loader)
            batches_since_quiz = 0

        # save
        if batch_idx % batches_per_save == 0:
            path = f"{save_folder_path}/batch_{batch_idx}.pt"
            total_time = prev_train_time + (time.time() - session_start_time)
            save_checkpoint(path, model, optimizer, batch_idx, quiz_history, total_time, scheduler)
            print(f"Saved: {path}")
            cleanup_checkpoints(save_folder_path)
