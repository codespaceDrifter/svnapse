"""
Evolutionary Neural Architecture Search (NAS) - Primitive Level

WHAT THIS IS:
    Search over computation graphs built from PRIMITIVE OPERATIONS.
    No predefined attention, no MLP blocks. Just raw math.
    The goal: discover what architectures emerge from first principles.

PRIMITIVES:
    Operations:
        - matmul(a, b)    : matrix multiplication
        - add(a, b)       : element-wise addition
        - sub(a, b)       : element-wise subtraction
        - mul(a, b)       : element-wise multiplication
        - div(a, b)       : element-wise division
        - pow(a, n)       : element-wise power
        - softmax(a)      : softmax over last dim
        - relu(a)         : ReLU activation
        - silu(a)         : SiLU activation
        - transpose(a)    : swap last two dims
        - sum(a)          : reduce sum over last dim
        - mean(a)         : reduce mean over last dim

    Tensors:
        - Weight(shape)   : learnable parameter
        - Input(idx)      : reference to input tensor

GENOME:
    A computation graph represented as a list of nodes.
    Each node: {op: str, inputs: [int], shape: tuple}
    Inputs reference earlier nodes by index.

EXAMPLE - Attention discovered from primitives:
    node 0: Input(0)                          # x: (batch, seq, dim)
    node 1: Weight((dim, dim))                # W_q
    node 2: Weight((dim, dim))                # W_k
    node 3: Weight((dim, dim))                # W_v
    node 4: matmul(0, 1)                      # Q = x @ W_q
    node 5: matmul(0, 2)                      # K = x @ W_k
    node 6: matmul(0, 3)                      # V = x @ W_v
    node 7: transpose(5)                      # K.T
    node 8: matmul(4, 7)                      # Q @ K.T
    node 9: div(8, sqrt(dim))                 # scaled
    node 10: softmax(9)                       # attention weights
    node 11: matmul(10, 6)                    # attn @ V

USAGE:
    python -m utils.NAS --pop_size 10 --generations 100
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import random
import copy
import json
import math
from pathlib import Path
from dataclasses import dataclass, field
from typing import Callable


# =============================================================================
# PRIMITIVE OPERATIONS
# =============================================================================

# op_name -> (function, arity, description)
PRIMITIVES: dict[str, tuple[Callable, int, str]] = {
    # binary ops
    "matmul": (lambda a, b: a @ b, 2, "matrix multiply"),
    "add": (lambda a, b: a + b, 2, "element-wise add"),
    "sub": (lambda a, b: a - b, 2, "element-wise subtract"),
    "mul": (lambda a, b: a * b, 2, "element-wise multiply"),
    "div": (lambda a, b: a / (b + 1e-8), 2, "element-wise divide"),
    # unary ops
    "pow2": (lambda a: a ** 2, 1, "square"),
    "sqrt": (lambda a: torch.sqrt(a.abs() + 1e-8), 1, "square root"),
    "neg": (lambda a: -a, 1, "negate"),
    "relu": (lambda a: F.relu(a), 1, "ReLU"),
    "silu": (lambda a: F.silu(a), 1, "SiLU"),
    "tanh": (lambda a: torch.tanh(a), 1, "tanh"),
    "softmax": (lambda a: F.softmax(a, dim=-1), 1, "softmax last dim"),
    "transpose": (lambda a: a.transpose(-1, -2), 1, "swap last 2 dims"),
    "sum": (lambda a: a.sum(dim=-1, keepdim=True), 1, "sum last dim"),
    "mean": (lambda a: a.mean(dim=-1, keepdim=True), 1, "mean last dim"),
    "norm": (lambda a: a / (a.norm(dim=-1, keepdim=True) + 1e-8), 1, "L2 normalize"),
}


# =============================================================================
# COMPUTATION GRAPH NODES
# =============================================================================

@dataclass
class Node:
    """One node in the computation graph."""
    op: str  # operation name or "input" or "weight"
    # indices of input nodes (for ops)
    inputs: list[int] = field(default_factory=list)
    # for weight nodes: shape of the parameter
    shape: tuple = None
    # for input nodes: which input (0 = x, 1 = optional second input)
    input_idx: int = 0

    def to_dict(self):
        return {
            "op": self.op,
            "inputs": self.inputs,
            "shape": list(self.shape) if self.shape else None,
            "input_idx": self.input_idx,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(
            op=d["op"],
            inputs=d.get("inputs", []),
            shape=tuple(d["shape"]) if d.get("shape") else None,
            input_idx=d.get("input_idx", 0),
        )


@dataclass
class Genome:
    """Computation graph as a list of nodes."""
    # list of nodes in topological order
    nodes: list[Node] = field(default_factory=list)
    # which node is the output
    output_node: int = -1
    # metadata
    generation: int = 0
    parent_id: str = None
    mutation_applied: str = None
    # config
    embed_dim: int = 64
    vocab_size: int = 97

    def to_dict(self):
        return {
            "nodes": [n.to_dict() for n in self.nodes],
            "output_node": self.output_node,
            "generation": self.generation,
            "parent_id": self.parent_id,
            "mutation_applied": self.mutation_applied,
            "embed_dim": self.embed_dim,
            "vocab_size": self.vocab_size,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(
            nodes=[Node.from_dict(n) for n in d["nodes"]],
            output_node=d["output_node"],
            generation=d.get("generation", 0),
            parent_id=d.get("parent_id"),
            mutation_applied=d.get("mutation_applied"),
            embed_dim=d.get("embed_dim", 64),
            vocab_size=d.get("vocab_size", 97),
        )

    def save(self, path: Path):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path):
        with open(path) as f:
            return cls.from_dict(json.load(f))

    def describe(self) -> str:
        """Human readable summary."""
        n_weights = sum(1 for n in self.nodes if n.op == "weight")
        n_ops = sum(1 for n in self.nodes if n.op in PRIMITIVES)
        ops_used = set(n.op for n in self.nodes if n.op in PRIMITIVES)
        return f"nodes={len(self.nodes)}, weights={n_weights}, ops={n_ops}, types={ops_used}"

    def to_code(self) -> str:
        """Generate pseudocode representation."""
        lines = []
        for i, node in enumerate(self.nodes):
            if node.op == "input":
                lines.append(f"n{i} = INPUT[{node.input_idx}]")
            elif node.op == "weight":
                lines.append(f"n{i} = WEIGHT{list(node.shape)}")
            else:
                args = ", ".join(f"n{j}" for j in node.inputs)
                lines.append(f"n{i} = {node.op}({args})")
        lines.append(f"OUTPUT = n{self.output_node}")
        return "\n".join(lines)


def minimal_genome(embed_dim: int = 64, vocab_size: int = 97) -> Genome:
    """
    Simplest possible model: embed -> linear -> output.

    x: (batch, seq) -> embed -> (batch, seq, dim) -> W -> (batch, seq, vocab)
    """
    nodes = [
        # node 0: input tokens (will be embedded by the model wrapper)
        Node(op="input", input_idx=0),
        # node 1: weight for output projection
        Node(op="weight", shape=(embed_dim, vocab_size)),
        # node 2: matmul(input, weight) -> logits
        Node(op="matmul", inputs=[0, 1]),
    ]
    return Genome(
        nodes=nodes,
        output_node=2,
        embed_dim=embed_dim,
        vocab_size=vocab_size,
    )


# =============================================================================
# BUILD MODEL FROM GENOME
# =============================================================================

class GenomeModel(nn.Module):
    """
    Execute a computation graph genome.
    Wraps with embedding and handles weight parameters.
    """

    def __init__(self, genome: Genome):
        super().__init__()
        self.genome = genome

        # embedding layer (always present)
        self.embed = nn.Embedding(genome.vocab_size, genome.embed_dim)

        # create parameters for weight nodes
        self.weights = nn.ParameterDict()
        for i, node in enumerate(genome.nodes):
            if node.op == "weight":
                # initialize with small random values
                param = nn.Parameter(torch.randn(*node.shape) * 0.02)
                self.weights[str(i)] = param

    def forward(self, x):
        """
        x: (batch, seq) token ids
        returns: (batch, seq, vocab) logits
        """
        # embed input
        # (batch, seq) -> (batch, seq, embed_dim)
        embedded = self.embed(x)

        # cache for computed node values
        cache = {}

        for i, node in enumerate(self.genome.nodes):
            if node.op == "input":
                # input node references the embedded input
                cache[i] = embedded

            elif node.op == "weight":
                # weight node references learned parameter
                cache[i] = self.weights[str(i)]

            else:
                # operation node
                fn, arity, _ = PRIMITIVES[node.op]

                # gather inputs
                inputs = [cache[j] for j in node.inputs]

                # execute op
                try:
                    result = fn(*inputs)
                    cache[i] = result
                except Exception:
                    # shape mismatch or other error - return zeros
                    # (mutation might create invalid graphs)
                    cache[i] = torch.zeros_like(embedded)

        # return output node
        return cache[self.genome.output_node]

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build_model(genome: Genome) -> GenomeModel:
    return GenomeModel(genome)


# =============================================================================
# WEIGHT INHERITANCE
# =============================================================================

def inherit_weights(child: GenomeModel, parent: GenomeModel) -> tuple[int, int]:
    """
    Copy weights from parent to child where possible.
    Returns (inherited_count, new_count).
    """
    inherited = 0
    new = 0

    child_state = child.state_dict()
    parent_state = parent.state_dict()

    for name, child_param in child_state.items():
        if name in parent_state:
            parent_param = parent_state[name]
            if child_param.shape == parent_param.shape:
                child_state[name] = parent_param.clone()
                inherited += 1
            else:
                # shape mismatch - copy what fits
                slices = tuple(
                    slice(0, min(cs, ps))
                    for cs, ps in zip(child_param.shape, parent_param.shape)
                )
                child_state[name][slices] = parent_param[slices].clone()
                inherited += 1
        else:
            new += 1

    child.load_state_dict(child_state)
    return inherited, new


# =============================================================================
# MUTATIONS
# =============================================================================

def get_valid_inputs(genome: Genome, node_idx: int, arity: int) -> list[list[int]]:
    """Get valid input combinations for a node at given index."""
    # can reference any earlier node
    available = list(range(node_idx))
    if not available:
        return []

    if arity == 1:
        return [[i] for i in available]
    else:
        # pairs of inputs
        return [[i, j] for i in available for j in available]


def mutate(genome: Genome, genome_id: str = None) -> Genome:
    """Apply one random mutation."""
    genome = copy.deepcopy(genome)
    genome.generation += 1
    genome.parent_id = genome_id

    mutation_type = random.choice([
        "add_op",
        "add_weight",
        "remove_node",
        "change_op",
        "change_input",
        "change_output",
    ])

    genome.mutation_applied = mutation_type

    if mutation_type == "add_op":
        # add a new operation node
        op_name = random.choice(list(PRIMITIVES.keys()))
        _, arity, _ = PRIMITIVES[op_name]

        # insert at random position (after inputs/weights, before output)
        min_pos = sum(1 for n in genome.nodes if n.op in ["input", "weight"])
        max_pos = len(genome.nodes)

        if min_pos >= max_pos:
            pos = max_pos
        else:
            pos = random.randint(min_pos, max_pos)

        # pick random valid inputs
        valid = get_valid_inputs(genome, pos, arity)
        if valid:
            inputs = random.choice(valid)
            new_node = Node(op=op_name, inputs=inputs)
            genome.nodes.insert(pos, new_node)

            # update references in later nodes
            for i in range(pos + 1, len(genome.nodes)):
                genome.nodes[i].inputs = [
                    j + 1 if j >= pos else j
                    for j in genome.nodes[i].inputs
                ]

            # update output reference
            if genome.output_node >= pos:
                genome.output_node += 1

            genome.mutation_applied = f"add_{op_name}_at_{pos}"

    elif mutation_type == "add_weight":
        # add a new weight parameter
        dim = genome.embed_dim
        # random shape that might be useful
        shape = random.choice([
            (dim, dim),
            (dim, dim * 2),
            (dim * 2, dim),
            (dim, 1),
            (1, dim),
        ])

        # insert after other weights/inputs
        pos = sum(1 for n in genome.nodes if n.op in ["input", "weight"])
        new_node = Node(op="weight", shape=shape)
        genome.nodes.insert(pos, new_node)

        # update references
        for i in range(pos + 1, len(genome.nodes)):
            genome.nodes[i].inputs = [
                j + 1 if j >= pos else j
                for j in genome.nodes[i].inputs
            ]
        if genome.output_node >= pos:
            genome.output_node += 1

        genome.mutation_applied = f"add_weight_{shape}"

    elif mutation_type == "remove_node":
        # remove a non-essential node
        removable = []
        for i, node in enumerate(genome.nodes):
            # can't remove input or the output node
            if node.op == "input":
                continue
            if i == genome.output_node:
                continue
            # can't remove if other nodes depend on it (except output)
            dependents = [j for j, n in enumerate(genome.nodes) if i in n.inputs and j != genome.output_node]
            if not dependents:
                removable.append(i)

        if removable:
            idx = random.choice(removable)
            removed = genome.nodes.pop(idx)

            # update references
            for node in genome.nodes:
                node.inputs = [
                    j - 1 if j > idx else j
                    for j in node.inputs
                    if j != idx  # remove refs to deleted node
                ]

            if genome.output_node > idx:
                genome.output_node -= 1

            genome.mutation_applied = f"remove_{removed.op}_from_{idx}"

    elif mutation_type == "change_op":
        # change an operation to a different one with same arity
        op_nodes = [(i, n) for i, n in enumerate(genome.nodes) if n.op in PRIMITIVES]

        if op_nodes:
            idx, node = random.choice(op_nodes)
            old_op = node.op
            _, old_arity, _ = PRIMITIVES[old_op]

            # find ops with same arity
            same_arity = [name for name, (_, arity, _) in PRIMITIVES.items() if arity == old_arity]
            if len(same_arity) > 1:
                same_arity.remove(old_op)
                new_op = random.choice(same_arity)
                genome.nodes[idx].op = new_op
                genome.mutation_applied = f"change_{old_op}_to_{new_op}_at_{idx}"

    elif mutation_type == "change_input":
        # rewire an operation's input
        op_nodes = [(i, n) for i, n in enumerate(genome.nodes) if n.op in PRIMITIVES and n.inputs]

        if op_nodes:
            idx, node = random.choice(op_nodes)
            _, arity, _ = PRIMITIVES[node.op]

            # pick which input to change
            input_slot = random.randint(0, len(node.inputs) - 1)

            # pick new source (any earlier node)
            available = list(range(idx))
            if available:
                new_source = random.choice(available)
                old_source = node.inputs[input_slot]
                genome.nodes[idx].inputs[input_slot] = new_source
                genome.mutation_applied = f"rewire_node{idx}_input{input_slot}_{old_source}_to_{new_source}"

    elif mutation_type == "change_output":
        # change which node is the output
        # must be an op node (not input/weight)
        candidates = [i for i, n in enumerate(genome.nodes) if n.op in PRIMITIVES]
        if candidates:
            old_output = genome.output_node
            new_output = random.choice(candidates)
            genome.output_node = new_output
            genome.mutation_applied = f"output_{old_output}_to_{new_output}"

    return genome


# =============================================================================
# MODULAR ARITHMETIC TASK
# =============================================================================

class ModularArithmeticDataset(Dataset):
    """(a op b) mod p"""

    def __init__(self, p: int = 97, op: str = "add", split: str = "train", train_frac: float = 0.7):
        self.p = p

        all_pairs = []
        for a in range(p):
            for b in range(p):
                if op == "add":
                    c = (a + b) % p
                elif op == "mul":
                    c = (a * b) % p
                elif op == "sub":
                    c = (a - b) % p
                else:
                    raise ValueError(f"unknown op: {op}")
                all_pairs.append((a, b, c))

        random.seed(42)
        random.shuffle(all_pairs)

        n_train = int(len(all_pairs) * train_frac)
        self.data = all_pairs[:n_train] if split == "train" else all_pairs[n_train:]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        a, b, c = self.data[idx]
        return torch.tensor([a, b], dtype=torch.long), torch.tensor(c, dtype=torch.long)


def make_dataloaders(p: int = 97, op: str = "add", batch_size: int = 256):
    train_ds = ModularArithmeticDataset(p, op, "train")
    val_ds = ModularArithmeticDataset(p, op, "val")
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False),
    )


# =============================================================================
# TRAINING
# =============================================================================

def train_steps(model: nn.Module, loader: DataLoader, steps: int, lr: float = 1e-3, device: str = "cuda"):
    """Train for fixed steps, return avg loss."""
    model = model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    step = 0
    total_loss = 0.0

    while step < steps:
        for x, y in loader:
            if step >= steps:
                break
            x, y = x.to(device), y.to(device)

            # (batch, 2) -> (batch, 2, vocab)
            logits = model(x)

            # handle shape issues from broken genomes
            if logits.dim() == 2:
                # (batch, vocab) - missing seq dim, expand
                logits = logits.unsqueeze(1).expand(-1, 2, -1)
            elif logits.dim() == 1:
                # totally broken, skip
                continue

            # predict from last position
            # (batch, vocab)
            pred = logits[:, -1, :]

            if pred.shape[-1] != loader.dataset.p:
                # wrong vocab size, skip
                continue

            loss = F.cross_entropy(pred, y)

            if torch.isnan(loss) or torch.isinf(loss):
                continue

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            step += 1

    return total_loss / max(step, 1)


def evaluate(model: nn.Module, loader: DataLoader, device: str = "cuda") -> tuple[float, float]:
    """Return (loss, accuracy)."""
    model = model.to(device)
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            logits = model(x)

            # handle broken shapes
            if logits.dim() == 2:
                logits = logits.unsqueeze(1).expand(-1, 2, -1)
            elif logits.dim() != 3:
                continue

            pred = logits[:, -1, :]

            if pred.shape[-1] != loader.dataset.p:
                continue

            loss = F.cross_entropy(pred, y)

            if not (torch.isnan(loss) or torch.isinf(loss)):
                total_loss += loss.item() * x.size(0)
                correct += (pred.argmax(-1) == y).sum().item()
                total += x.size(0)

    if total == 0:
        return float("inf"), 0.0

    return total_loss / total, correct / total


# =============================================================================
# POPULATION EVOLUTION
# =============================================================================

@dataclass
class Individual:
    """One member of the population."""
    genome: Genome
    model: GenomeModel = None
    train_loss: float = float("inf")
    val_loss: float = float("inf")
    val_acc: float = 0.0
    id: str = None

    def __post_init__(self):
        if self.id is None:
            self.id = f"g{self.genome.generation}_{random.randint(0, 99999):05d}"


class Population:
    """Evolving population."""

    def __init__(
        self,
        size: int = 10,
        embed_dim: int = 64,
        vocab_size: int = 97,
        elite_frac: float = 0.2,
        tournament_size: int = 3,
        device: str = "cuda",
    ):
        self.size = size
        self.elite_frac = elite_frac
        self.tournament_size = tournament_size
        self.device = device

        # init population with minimal genomes
        self.individuals = []
        for _ in range(size):
            genome = minimal_genome(embed_dim, vocab_size)
            model = build_model(genome)
            self.individuals.append(Individual(genome=genome, model=model))

        self.generation = 0
        self.history = []

    def evaluate_all(self, train_loader: DataLoader, val_loader: DataLoader, train_steps_per_eval: int = 500):
        """Train and evaluate all individuals."""
        for ind in self.individuals:
            try:
                train_loss = train_steps(ind.model, train_loader, train_steps_per_eval, device=self.device)
                val_loss, val_acc = evaluate(ind.model, val_loader, device=self.device)
            except Exception as e:
                # broken genome
                train_loss = float("inf")
                val_loss = float("inf")
                val_acc = 0.0

            ind.train_loss = train_loss
            ind.val_loss = val_loss
            ind.val_acc = val_acc

            print(f"  {ind.id}: loss={val_loss:.4f}, acc={val_acc:.2%}, nodes={len(ind.genome.nodes)}, params={ind.model.count_params():,}")

    def select_and_reproduce(self):
        """Create next generation."""
        # sort by val loss
        ranked = sorted(self.individuals, key=lambda x: x.val_loss)

        # elites survive
        n_elites = max(1, int(self.size * self.elite_frac))
        elites = ranked[:n_elites]

        new_individuals = []

        # copy elites
        for elite in elites:
            new_individuals.append(Individual(
                genome=copy.deepcopy(elite.genome),
                model=copy.deepcopy(elite.model),
                val_loss=elite.val_loss,
                val_acc=elite.val_acc,
                id=elite.id,
            ))

        # fill rest with mutations
        while len(new_individuals) < self.size:
            # tournament selection
            contestants = random.sample(self.individuals, min(self.tournament_size, len(self.individuals)))
            winner = min(contestants, key=lambda x: x.val_loss)

            # mutate
            child_genome = mutate(winner.genome, winner.id)

            try:
                child_model = build_model(child_genome)
                inherited, new = inherit_weights(child_model, winner.model)
            except Exception:
                # broken genome, skip
                continue

            child = Individual(genome=child_genome, model=child_model)
            new_individuals.append(child)

            print(f"  {child.id} from {winner.id}: {child_genome.mutation_applied}")

        self.individuals = new_individuals
        self.generation += 1

    def get_best(self) -> Individual:
        return min(self.individuals, key=lambda x: x.val_loss)

    def log_generation(self):
        best = self.get_best()
        avg_loss = sum(i.val_loss for i in self.individuals if i.val_loss < float("inf")) / max(1, len(self.individuals))
        avg_acc = sum(i.val_acc for i in self.individuals) / len(self.individuals)

        stats = {
            "generation": self.generation,
            "best_loss": best.val_loss,
            "best_acc": best.val_acc,
            "best_id": best.id,
            "best_arch": best.genome.describe(),
            "avg_loss": avg_loss,
            "avg_acc": avg_acc,
        }
        self.history.append(stats)

        print(f"\n=== Generation {self.generation} ===")
        print(f"Best: {best.id} | loss={best.val_loss:.4f} | acc={best.val_acc:.2%}")
        print(f"Architecture: {best.genome.describe()}")
        print(f"Code:\n{best.genome.to_code()}")

        return stats


def evolve(
    pop_size: int = 10,
    generations: int = 100,
    train_steps_per_eval: int = 500,
    embed_dim: int = 64,
    p: int = 97,
    op: str = "add",
    device: str = "cuda",
    save_dir: Path = None,
):
    """Main evolution loop."""
    print(f"Starting primitive-level NAS")
    print(f"  pop={pop_size}, gens={generations}, task=({op} mod {p})")
    print(f"  embed_dim={embed_dim}")
    print(f"  primitives: {list(PRIMITIVES.keys())}")

    train_loader, val_loader = make_dataloaders(p, op)
    pop = Population(size=pop_size, embed_dim=embed_dim, vocab_size=p, device=device)

    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    for gen in range(generations):
        print(f"\n{'='*60}")
        print(f"GENERATION {gen}")
        print(f"{'='*60}")

        print("\nEvaluating:")
        pop.evaluate_all(train_loader, val_loader, train_steps_per_eval)

        stats = pop.log_generation()

        if save_dir:
            best = pop.get_best()
            best.genome.save(save_dir / f"best_gen{gen}.json")
            torch.save(best.model.state_dict(), save_dir / f"best_gen{gen}.pt")

        # solved?
        if pop.get_best().val_acc > 0.99:
            print(f"\n*** SOLVED at generation {gen}! ***")
            break

        if gen < generations - 1:
            print("\nReproducing:")
            pop.select_and_reproduce()

    # final
    print("\n" + "="*60)
    print("EVOLUTION COMPLETE")
    print("="*60)
    best = pop.get_best()
    print(f"Best: {best.id}")
    print(f"  Loss: {best.val_loss:.4f}")
    print(f"  Accuracy: {best.val_acc:.2%}")
    print(f"  Params: {best.model.count_params():,}")
    print(f"  Architecture: {best.genome.describe()}")
    print(f"\nDiscovered computation graph:")
    print(best.genome.to_code())

    if save_dir:
        best.genome.save(save_dir / "final_best.json")
        torch.save(best.model.state_dict(), save_dir / "final_best.pt")
        with open(save_dir / "history.json", "w") as f:
            json.dump(pop.history, f, indent=2)

    return pop, best


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Primitive-level Evolutionary NAS")
    parser.add_argument("--pop_size", type=int, default=10)
    parser.add_argument("--generations", type=int, default=100)
    parser.add_argument("--train_steps", type=int, default=500)
    parser.add_argument("--embed_dim", type=int, default=64)
    parser.add_argument("--p", type=int, default=97)
    parser.add_argument("--op", type=str, default="add", choices=["add", "mul", "sub"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_dir", type=str, default="nas_results")
    args = parser.parse_args()

    evolve(
        pop_size=args.pop_size,
        generations=args.generations,
        train_steps_per_eval=args.train_steps,
        embed_dim=args.embed_dim,
        p=args.p,
        op=args.op,
        device=args.device,
        save_dir=Path(args.save_dir),
    )
