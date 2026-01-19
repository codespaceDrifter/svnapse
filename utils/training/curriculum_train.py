# each batch is broken into minibatches and gradient accumulation to fit in vram
# curriculum learning. each bin file is a lesson and can be chunked into multiple lessons
# we set mix lesson percentage progressively using training loss (quizes)

import time
import random
import torch
import os
import numpy as np

from utils.training.saves import cleanup_checkpoints, load_latest_checkpoint, save_checkpoint


class Curriculum:
    """
    Holds multiple lessons (each a memmap .bin file), supports chunking.
    Directly samples batches based on mix ratios.
    """
    def __init__(self, lesson_to_bin, seq_len, lesson_to_chunks=None):
        if lesson_to_chunks is None:
            lesson_to_chunks = {}

        # ["story", "wikipedia_0", "wikipedia_1", ...] - expanded lesson names in order
        self.lessons = []
        # {"story": (memmap, start_sample, num_samples), ...}
        self.data = {}
        self.seq_len = seq_len

        for lesson, path in lesson_to_bin.items():
            mmap = np.memmap(path, dtype=np.int32, mode='r')
            full_samples = len(mmap) // seq_len
            n_chunks = lesson_to_chunks.get(lesson, 1)

            if n_chunks == 1:
                self.lessons.append(lesson)
                self.data[lesson] = (mmap, 0, full_samples)
            else:
                chunk_size = full_samples // n_chunks
                for i in range(n_chunks):
                    chunk_name = f"{lesson}_{i}"
                    start = i * chunk_size
                    end = full_samples if i == n_chunks - 1 else start + chunk_size
                    self.lessons.append(chunk_name)
                    self.data[chunk_name] = (mmap, start, end - start)

        # {"story": 100, ...} - num samples per lesson
        self.sizes = {l: self.data[l][2] for l in self.lessons}

    def sample_batch(self, lesson_to_mix, batch_size):
        """Sample batch according to mix ratios. Returns (batch_size, seq_len) tensor."""
        lessons = [l for l in self.lessons if lesson_to_mix.get(l, 0) > 0]
        ratios = [lesson_to_mix[l] for l in lessons]

        batch = []
        for _ in range(batch_size):
            lesson = random.choices(lessons, weights=ratios, k=1)[0]
            mmap, start_sample, num_samples = self.data[lesson]
            idx = random.randint(0, num_samples - 1)
            offset = (start_sample + idx) * self.seq_len
            # (seq_len,)
            tokens = torch.from_numpy(mmap[offset:offset + self.seq_len].copy()).long()
            batch.append(tokens)

        # (batch_size, seq_len)
        return torch.stack(batch)

    def quiz_batch(self, lesson, batch_size):
        """Sample batch from specific lesson for quizzing."""
        mmap, start_sample, num_samples = self.data[lesson]
        batch = []
        for _ in range(batch_size):
            idx = random.randint(0, num_samples - 1)
            offset = (start_sample + idx) * self.seq_len
            tokens = torch.from_numpy(mmap[offset:offset + self.seq_len].copy()).long()
            batch.append(tokens)
        return torch.stack(batch)


def fmt_mix(lesson_to_mix, priority_order):
    """Format mix showing only first non-zero to last non-zero range."""
    # find first and last non-zero indices
    first, last = -1, -1
    for i, lesson in enumerate(priority_order):
        if lesson_to_mix.get(lesson, 0) > 0:
            if first == -1:
                first = i
            last = i
    if first == -1:
        return "{}"
    subset = priority_order[first:last + 1]
    return "{" + ", ".join(f"{l}: {lesson_to_mix[l]:.3f}" for l in subset) + "}"


def quiz_until_mix_full(model, curriculum, batch_size, quiz_batches, acceptable_loss, priority_order):
    """
    Quiz in priority order, stop early once mix is determined.
    Returns (quiz_losses, all_passed, lesson_to_mix)
    """
    model.eval()
    quiz_losses = {}
    # {lesson: mix_ratio} - built gradually during quiz
    lesson_to_mix = {l: 0.0 for l in priority_order}
    # how much mix budget remains
    remaining = 1.0
    all_passed = True

    loss_fn = torch.nn.CrossEntropyLoss()
    samples_per_lesson = batch_size * quiz_batches

    with torch.no_grad():
        for lesson in priority_order:
            # (samples, seq_len)
            tokens = curriculum.quiz_batch(lesson, samples_per_lesson).cuda()

            total_loss = 0.0
            for start in range(0, samples_per_lesson, batch_size):
                end = min(start + batch_size, samples_per_lesson)
                # (chunk, seq_len)
                chunk = tokens[start:end]
                # (chunk, seq_len, vocab)
                logits, _ = model(chunk, labels=chunk)
                total_loss += loss_fn(
                    logits[:, :-1].reshape(-1, logits.size(-1)),
                    chunk[:, 1:].reshape(-1)
                ).item() * (end - start)

            loss = total_loss / samples_per_lesson
            quiz_losses[lesson] = loss

            if loss >= acceptable_loss:
                all_passed = False
                raw = min(1.0, (loss - acceptable_loss) / acceptable_loss)
                # cap so we don't exceed 1.0 total
                take = min(raw, remaining)
                lesson_to_mix[lesson] = take
                remaining -= take

                if remaining <= 0:
                    break

    model.train()

    # if all passed, equal mix
    if all_passed:
        n = len(priority_order)
        return quiz_losses, True, {l: 1.0 / n for l in priority_order}

    # scale up if total < 1.0
    total = sum(lesson_to_mix.values())
    if 0 < total < 1.0:
        lesson_to_mix = {k: v / total for k, v in lesson_to_mix.items()}

    return quiz_losses, False, lesson_to_mix


def train_curriculum(
    # model
    model,
    optimizer,
    scheduler=None,
    # training
    batch_size=128,
    accumulation_steps=8,
    seq_len=2048,
    clip_grad_norm=1.0,
    # data
    lesson_to_bin=None,      # {"lesson1": "path.bin", ...}
    priority_order=None,        # ["lesson1", "lesson2", ...] - curriculum order
    save_folder_path="weights",
    # curriculum
    acceptable_loss=3.0,
    stop_streak=10,
    # logging
    batches_per_log=10,
    batches_per_save=100,
    batches_per_quiz=100,
    quiz_batches=5,
    tokenizer=None,
    # chunking (most niche)
    lesson_to_chunks=None,   # {"lesson1": 3, ...} - split lesson into N chunks
):
    assert batch_size % accumulation_steps == 0
    minibatch_size = batch_size // accumulation_steps

    # expand priority_order for chunked lessons
    if lesson_to_chunks is None:
        lesson_to_chunks = {}
    expanded_priority = []
    for lesson in priority_order:
        n_chunks = lesson_to_chunks.get(lesson, 1)
        if n_chunks == 1:
            expanded_priority.append(lesson)
        else:
            expanded_priority.extend(f"{lesson}_{i}" for i in range(n_chunks))
    priority_order = expanded_priority

    model.cuda().bfloat16()
    os.makedirs(save_folder_path, exist_ok=True)

    curriculum = Curriculum(lesson_to_bin, seq_len, lesson_to_chunks)
    for l in curriculum.lessons:
        print(f"{l}: {curriculum.sizes[l]} samples")
    print(f"Total: {sum(curriculum.sizes.values())} samples")

    # quiz_history: [(batch_idx, {lesson: loss, ...}), ...]
    last_batch, quiz_history, prev_train_time = load_latest_checkpoint(
        save_folder_path, model, optimizer, scheduler
    )
    batch_idx = last_batch
    session_start_time = time.time()
    streak = 0

    # initial quiz
    print("\n=== Initial quiz ===")
    quiz_losses, _, lesson_to_mix = quiz_until_mix_full(
        model, curriculum, minibatch_size, quiz_batches, acceptable_loss, priority_order
    )
    for l, loss in quiz_losses.items():
        print(f"  {l}: {loss:.4f}")
    quiz_history.append((batch_idx, quiz_losses.copy()))
    print(f"Initial mix: {fmt_mix(lesson_to_mix, priority_order)}")

    model.train()
    batches_since_quiz = 0

    while True:
        optimizer.zero_grad()
        acc_loss = 0.0

        for _ in range(accumulation_steps):
            # (minibatch, seq_len)
            tokens = curriculum.sample_batch(lesson_to_mix, minibatch_size).cuda()
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
            print(f"Mix: {fmt_mix(lesson_to_mix, priority_order)}")

            if tokenizer:
                pred = logits[0, :50].argmax(dim=-1).tolist()
                tgt = tokens[0, 1:51].tolist()
                print(f"  Target: {repr(tokenizer.decode(tgt)[:100])}")
                print(f"  Pred:   {repr(tokenizer.decode(pred)[:100])}")

        # re-quiz and adjust mix
        if batches_since_quiz >= batches_per_quiz:
            print("\n--- Quiz ---")
            quiz_losses, all_passed, lesson_to_mix = quiz_until_mix_full(
                model, curriculum, minibatch_size, quiz_batches, acceptable_loss, priority_order
            )
            for l, loss in quiz_losses.items():
                print(f"  {l}: {loss:.4f}")
            quiz_history.append((batch_idx, quiz_losses.copy()))

            if all_passed:
                streak += 1
                print(f"All lessons passed! Streak: {streak}/{stop_streak}")
                if streak >= stop_streak:
                    print(f"\nTraining complete - {stop_streak} consecutive passes!")
                    path = f"{save_folder_path}/batch_{batch_idx}.pt"
                    total_time = prev_train_time + (time.time() - session_start_time)
                    save_checkpoint(path, model, optimizer, batch_idx, quiz_history, total_time, scheduler)
                    return
            else:
                if streak > 0:
                    print(f"Streak reset (was {streak})")
                streak = 0

            print(f"New mix: {fmt_mix(lesson_to_mix, priority_order)}")
            batches_since_quiz = 0

        # save
        if batch_idx % batches_per_save == 0:
            path = f"{save_folder_path}/batch_{batch_idx}.pt"
            total_time = prev_train_time + (time.time() - session_start_time)
            save_checkpoint(path, model, optimizer, batch_idx, quiz_history, total_time, scheduler)
            print(f"Saved: {path}")
            cleanup_checkpoints(save_folder_path)
