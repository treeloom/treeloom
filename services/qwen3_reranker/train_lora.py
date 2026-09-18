"""Listwise LoRA fine-tune for Qwen3-Reranker-0.6B-seq-cls.

Consumes the mined training groups from `benchmarks/mine_reranker_training_data.py`
(one JSON object per line: {query, positive, negatives:[...], meta:{repo,...}})
and LoRA-fine-tunes the SAME seq-cls cross-encoder this directory's `app.py`
serves, with a listwise (group-softmax / InfoNCE) objective that directly
optimizes "rank the answer chunk above its hard negatives" — i.e. recall@1 / MRR.

Design decisions:
  * LoRA on the 0.6B, head trained in full (`modules_to_save=["score"]`). Small
    adapter, preserves the general code competence that makes it generalize to
    unseen repos. The `/rerank` contract is unchanged — still a single logit.
  * The prompt template is FROZEN to the serving format. `format_input` below is
    a verbatim copy of app.py's PREFIX / `<Instruct>/<Query>/<Document>` / SUFFIX.
    Train on exactly what the service serves or you've trained a different model
    than you deploy (the seq-cls port is only score-identical-to-official WITH
    this template). `_assert_template_matches_service()` fails loud on drift.
  * Listwise loss: per group, softmax over [positive, neg0, neg1, ...] logits,
    cross-entropy toward index 0. Variable group sizes handled by a masked
    [B, max_G] score matrix (-inf padding).
  * Split BY REPO, not by query (`--val-repos`) — measures generalization, not
    memorization.

Run in an ISOLATED env (this repo's venv has numpy 2.x, which transformers
rejects — same isolation pattern as the aider/cgr benchmark arms):

    uv run --with 'numpy<2' --with 'transformers>=4.51,<5' --with peft \
           --with accelerate --with safetensors \
      python services/qwen3_reranker/train_lora.py \
        --train benchmarks/reranker/*.jsonl \
        --val-repos featbit \
        --out services/qwen3_reranker/adapters/v1 --merge

Then point the service at the merged weights: `MODEL_ID=<out>/merged` (the
service loads `AutoModelForSequenceClassification` and does NOT apply a bare PEFT
adapter, so `--merge` is required for serving). Benchmark it the usual way
(clean-100, pool 50, deterministic flags) before trusting it — beat 0.70/0.90/0.786.
"""
from __future__ import annotations

import argparse
import ast
import glob
import json
import os
import random
import re
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

# ── FROZEN serving template — MUST stay byte-identical to services/qwen3_reranker/app.py ──
# (verified at runtime by _assert_template_matches_service)
MODEL_ID = os.environ.get("MODEL_ID", "tomaarsen/Qwen3-Reranker-0.6B-seq-cls")
MAX_LENGTH = int(os.environ.get("RERANK_MAX_LENGTH", "1600"))
INSTRUCTION = os.environ.get(
    "RERANK_INSTRUCTION",
    "Given a code search query, retrieve relevant code snippets",
)
PREFIX = (
    "<|im_start|>system\nJudge whether the Document meets the requirements "
    "based on the Query and the Instruct provided. Note that the answer can "
    'only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
)
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def format_input(query: str, doc: str) -> str:
    """The exact string app.py builds per (query, document) pair."""
    return f"{PREFIX}<Instruct>: {INSTRUCTION}\n<Query>: {query}\n<Document>: {doc}{SUFFIX}"


def _assert_template_matches_service() -> None:
    """Fail loud if PREFIX/SUFFIX/INSTRUCTION here drifted from app.py — training
    on a stale template silently produces a model that scores garbage in service."""
    app = Path(__file__).with_name("app.py").read_text()

    def literal(name: str) -> str | None:
        # Grab the python literal assigned to `name` and parse it. ast.literal_eval
        # (not eval) — these are plain string literals / parenthesized adjacent-
        # string concatenations, which the parser folds into one constant; no code
        # execution, so a tampered app.py can't run anything here.
        m = re.search(rf"^{name}\s*=\s*(\(.*?\)|\".*?\"|'.*?')",
                      app, re.MULTILINE | re.DOTALL)
        return ast.literal_eval(m.group(1)) if m else None

    drift = [n for n, v in (("PREFIX", PREFIX), ("SUFFIX", SUFFIX))
             if literal(n) != v]
    # INSTRUCTION is an os.environ.get default; check the default string only.
    mi = re.search(r'RERANK_INSTRUCTION",\s*\n\s*("(?:[^"\\]|\\.)*")', app)
    if mi and ast.literal_eval(mi.group(1)) != INSTRUCTION:
        drift.append("INSTRUCTION")
    if drift:
        raise SystemExit(
            f"TEMPLATE DRIFT vs app.py: {drift}. The trainer's serving template "
            "must match the service byte-for-byte. Sync the constants and re-run."
        )


# ──────────────────────────── data ────────────────────────────
def load_groups(patterns: list[str]) -> list[dict]:
    groups: list[dict] = []
    files: list[str] = []
    for p in patterns:
        files.extend(sorted(glob.glob(p)) or [p])
    for fp in files:
        with open(fp) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                g = json.loads(line)
                if g.get("query") and g.get("positive") and g.get("negatives"):
                    groups.append(g)
    return groups


def _repo_of(g: dict) -> str:
    return (g.get("meta") or {}).get("repo") or "_unknown"


def split_by_repo(
    groups: list[dict], val_repos: list[str] | None, val_frac: float, seed: int,
) -> tuple[list[dict], list[dict], list[str]]:
    """Hold out WHOLE repos for validation. Explicit --val-repos wins; otherwise
    pick repos (smallest-first, deterministic) until ~val_frac of groups are held
    out. Returns (train, val, val_repo_names)."""
    by_repo: dict[str, list[dict]] = {}
    for g in groups:
        by_repo.setdefault(_repo_of(g), []).append(g)
    repos = sorted(by_repo)
    if val_repos:
        chosen = [r for r in val_repos if r in by_repo]
        missing = [r for r in val_repos if r not in by_repo]
        if missing:
            print(f"WARNING: --val-repos not found in data: {missing} "
                  f"(available: {repos})", file=sys.stderr)
    else:
        rng = random.Random(seed)
        order = sorted(repos, key=lambda r: (len(by_repo[r]), r))
        target = val_frac * len(groups)
        chosen, held = [], 0
        for r in order:
            if held >= target and chosen:
                break
            chosen.append(r)
            held += len(by_repo[r])
        rng.shuffle(chosen)
    chosen_set = set(chosen)
    train = [g for g in groups if _repo_of(g) not in chosen_set]
    val = [g for g in groups if _repo_of(g) in chosen_set]
    return train, val, chosen


class GroupDataset(torch.utils.data.Dataset):
    """Each item is the group's documents as templated strings, POSITIVE FIRST
    (index 0 is the answer; the listwise target is always 0). Negatives are kept
    in mined order (hardest first) and capped to bound memory."""

    def __init__(self, groups: list[dict], max_negatives: int):
        self.items = []
        for g in groups:
            negs = g["negatives"][:max_negatives]
            if not negs:
                continue
            texts = [format_input(g["query"], g["positive"])]
            texts += [format_input(g["query"], d) for d in negs]
            self.items.append(texts)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> list[str]:
        return self.items[i]


def make_collate(tokenizer, max_length: int):
    def collate(batch: list[list[str]]) -> dict:
        counts = [len(texts) for texts in batch]
        flat = [t for texts in batch for t in texts]
        enc = tokenizer(flat, truncation=True, max_length=max_length,
                        padding=True, return_tensors="pt")
        enc["counts"] = torch.tensor(counts, dtype=torch.long)
        return enc
    return collate


# ──────────────────── loss + eval (pure, torch-only) ────────────────────
def scores_matrix(logits: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    """Scatter a flat [N] logit vector (groups concatenated, positive first in
    each) into a padded [B, max_G] matrix with -inf in padding slots."""
    counts = counts.tolist()
    b, max_g = len(counts), max(counts)
    out = logits.new_full((b, max_g), float("-inf"))
    off = 0
    for i, c in enumerate(counts):
        out[i, :c] = logits[off:off + c]
        off += c
    return out


def listwise_loss(logits: torch.Tensor, counts: torch.Tensor,
                  temperature: float) -> torch.Tensor:
    """Group-softmax cross-entropy toward the positive (index 0 of each group).
    -inf padding contributes 0 probability mass."""
    scores = scores_matrix(logits, counts) / temperature
    target = torch.zeros(scores.size(0), dtype=torch.long, device=scores.device)
    return F.cross_entropy(scores, target)


def rank_metrics(logits: torch.Tensor, counts: torch.Tensor) -> tuple[float, float]:
    """(mean reciprocal rank of the positive, accuracy@1) for one batch. The
    positive is row 0 of each group; rank = 1 + #negatives scoring >= positive."""
    scores = scores_matrix(logits, counts)
    rr_sum = acc = 0.0
    for i, c in enumerate(counts.tolist()):
        pos = scores[i, 0]
        negs = scores[i, 1:c]
        rank = 1 + int((negs >= pos).sum().item())
        rr_sum += 1.0 / rank
        acc += float(rank == 1)
    n = scores.size(0)
    return rr_sum / n, acc / n


# ──────────────────────────── model ────────────────────────────
def apply_lora(model, lora_r: int, lora_alpha: int, lora_dropout: float):
    """Wrap a Qwen3 seq-cls model in LoRA (attention + MLP projections) with the
    classification head trained in full. Factored out so the peft wiring can be
    smoke-tested on a tiny random-init Qwen3 without the real weights/GPU."""
    from peft import LoraConfig, TaskType, get_peft_model

    lora = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        modules_to_save=["score"],  # train the classification head in full
    )
    return get_peft_model(model, lora)


def build_model(args):
    from transformers import AutoModelForSequenceClassification

    dtype = (torch.bfloat16 if args.bf16 and torch.cuda.is_bf16_supported()
             else (torch.float16 if args.fp16 else torch.float32))
    base = AutoModelForSequenceClassification.from_pretrained(
        args.model_id, num_labels=1, dtype=dtype)
    model = apply_lora(base, args.lora_r, args.lora_alpha, args.lora_dropout)
    model.print_trainable_parameters()
    return model, dtype


@torch.no_grad()
def evaluate(model, loader, device, temperature) -> dict:
    model.eval()
    tot_loss = tot_rr = tot_acc = 0.0
    n = 0
    for batch in loader:
        counts = batch.pop("counts")
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(**batch).logits.squeeze(-1).float()
        tot_loss += listwise_loss(logits, counts, temperature).item()
        rr, acc = rank_metrics(logits, counts)
        tot_rr += rr
        tot_acc += acc
        n += 1
    return {"loss": tot_loss / max(n, 1), "mrr": tot_rr / max(n, 1),
            "acc@1": tot_acc / max(n, 1)}


def train(args):
    _assert_template_matches_service()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer

    groups = load_groups(args.train)
    if not groups:
        raise SystemExit(f"no training groups loaded from {args.train}")
    tr, va, val_repos = split_by_repo(groups, args.val_repos, args.val_frac, args.seed)
    print(f"loaded {len(groups)} groups | train {len(tr)} | val {len(va)} "
          f"(held-out repos: {val_repos or '—'})")

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, padding_side="left")
    collate = make_collate(tokenizer, args.max_length)
    tr_ds = GroupDataset(tr, args.max_negatives)
    va_ds = GroupDataset(va, args.max_negatives)
    tr_loader = DataLoader(tr_ds, batch_size=args.groups_per_batch, shuffle=True,
                           collate_fn=collate)
    va_loader = DataLoader(va_ds, batch_size=args.groups_per_batch, shuffle=False,
                           collate_fn=collate) if len(va_ds) else None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = build_model(args)
    # The seq-cls head pools the last non-pad token, so a pad id must be defined
    # for batched (>1) inputs. The served model sets one; this is insurance.
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    model.to(device)
    opt = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=args.lr, weight_decay=args.weight_decay)
    total_steps = max(1, (len(tr_loader) // args.grad_accum) * args.epochs)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=total_steps, pct_start=0.1)

    best = -1.0
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        opt.zero_grad()
        for step, batch in enumerate(tr_loader, 1):
            counts = batch.pop("counts")
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits.squeeze(-1).float()
            loss = listwise_loss(logits, counts, args.temperature) / args.grad_accum
            loss.backward()
            running += loss.item() * args.grad_accum
            if step % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    (p for p in model.parameters() if p.requires_grad), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad()
            if step % args.log_every == 0:
                print(f"epoch {epoch} step {step}/{len(tr_loader)} "
                      f"loss {running/step:.4f} lr {sched.get_last_lr()[0]:.2e}")
        msg = f"epoch {epoch} train_loss {running/max(len(tr_loader),1):.4f}"
        if va_loader is not None:
            metrics = evaluate(model, va_loader, device, args.temperature)
            msg += (f" | val_loss {metrics['loss']:.4f} mrr {metrics['mrr']:.4f} "
                    f"acc@1 {metrics['acc@1']:.4f}")
            score = metrics["mrr"]
        else:
            score = -running  # no val → keep last by lowest train loss
        print(msg)
        if score > best:
            best = score
            model.save_pretrained(out)
            tokenizer.save_pretrained(out)
            (out / "training_meta.json").write_text(json.dumps(
                {"args": vars(args), "epoch": epoch, "best_score": best,
                 "val_repos": val_repos, "n_train": len(tr_ds),
                 "n_val": len(va_ds)}, indent=2))
            print(f"  saved adapter -> {out} (best score {best:.4f})")

    if args.merge:
        _merge(args, out, device)


def _merge(args, adapter_dir: Path, device: str):
    """Merge LoRA into the base and save servable full weights at <out>/merged.
    The service can't load a bare adapter, so this is what MODEL_ID points at."""
    from peft import PeftModel
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    base = AutoModelForSequenceClassification.from_pretrained(
        args.model_id, num_labels=1, dtype=torch.float16)
    merged = PeftModel.from_pretrained(base, adapter_dir).merge_and_unload()
    md = adapter_dir / "merged"
    merged.save_pretrained(md)
    AutoTokenizer.from_pretrained(args.model_id, padding_side="left").save_pretrained(md)
    print(f"merged servable weights -> {md}  (serve with MODEL_ID={md})")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train", nargs="+", required=True,
                   help="Mined group JSONL file(s)/globs (from mine_reranker_training_data.py).")
    p.add_argument("--out", required=True, help="Output dir for the LoRA adapter.")
    p.add_argument("--model-id", default=MODEL_ID, help="Base seq-cls model (default: %(default)s).")
    p.add_argument("--val-repos", type=lambda s: [x for x in s.split(",") if x],
                   default=None, help="Comma-separated repos to hold out for validation.")
    p.add_argument("--val-frac", type=float, default=0.2,
                   help="If --val-repos unset, hold out whole repos to ~this fraction (default %(default)s).")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--groups-per-batch", type=int, default=4,
                   help="Listwise GROUPS per step (each expands to ~1+negatives docs).")
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--max-negatives", type=int, default=15,
                   help="Cap negatives per group in training to bound memory (hardest-first).")
    p.add_argument("--max-length", type=int, default=MAX_LENGTH,
                   help="Token truncation length of the full templated input (default %(default)s, = service).")
    p.add_argument("--temperature", type=float, default=1.0,
                   help="InfoNCE temperature. Lower (e.g. 0.05, per the jina-v3 "
                        "distillation recipe) sharpens the group softmax so correct "
                        "ranking is rewarded with wider logit margins; tune on the "
                        "held-out repo's MRR (default %(default)s).")
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--bf16", action="store_true", default=True)
    p.add_argument("--no-bf16", dest="bf16", action="store_false")
    p.add_argument("--fp16", action="store_true", help="Use fp16 instead of bf16/fp32.")
    p.add_argument("--merge", action="store_true",
                   help="Also save merged servable weights at <out>/merged (required to serve).")
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


if __name__ == "__main__":
    train(parse_args())
