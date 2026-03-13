#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Predict grouped output for dev.csv and test.csv and compute Exact Match.
- Loads a token-classification checkpoint (BIO: B/I/O)
- Writes predictions_dev.csv and predictions_test.csv
- Prints exact-match accuracy for both (if gold is present)

Usage examples:
  python predict_eval.py --checkpoint ./results/checkpoint-XXXX \
                         --dev dev.csv --test test.csv --batch_size 16
"""
import argparse
import os
import sys
from typing import List, Tuple

import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForTokenClassification


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', type=str, required=True,
                    help='Path to HF checkpoint directory (tokenizer+model).')
    ap.add_argument('--dev', type=str, default='dev.csv', help='Path to dev.csv')
    ap.add_argument('--test', type=str, default='test.csv', help='Path to test.csv')
    ap.add_argument('--batch_size', type=int, default=16, help='Batch size for inference')
    ap.add_argument('--out_dir', type=str, default='.', help='Output directory for CSVs')
    return ap.parse_args()


def to_word_level_ids(tokenizer, texts: List[List[str]], token_preds: List[List[int]]):
    """Map token-level ids → word-level ids (first subword for each word)."""
    word_level = []
    tokenized = tokenizer(texts, is_split_into_words=True, truncation=True, padding=True)
    for i in range(len(texts)):
        word_ids = tokenized.word_ids(batch_index=i)
        tok_preds = token_preds[i]
        labels = []
        seen = set()
        for ti, wi in enumerate(word_ids):
            if wi is None or wi in seen:
                continue
            labels.append(int(tok_preds[ti]))
            seen.add(wi)
        word_level.append(labels)
    return word_level


def reconstruct_sentence(words: List[str], word_label_ids: List[int], id2label: dict) -> str:
    """Rebuild grouped sentence using BIO labels (B/I/O) and '__' joiner."""
    labs = [id2label.get(int(x), 'O') for x in word_label_ids]
    out, i = [], 0
    while i < len(words):
        lab = labs[i] if i < len(labs) else 'O'
        if lab == 'B':
            group = [words[i]]
            i += 1
            while i < len(words):
                lab_i = labs[i] if i < len(labs) else 'O'
                if lab_i == 'I':
                    group.append(words[i])
                    i += 1
                else:
                    break
            out.append('__'.join(group))
        else:
            out.append(words[i])
            i += 1
    return ' '.join(out)


def predict_for(df: pd.DataFrame, tokenizer, model, batch_size: int) -> List[str]:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    # id2label from config or default mapping
    if hasattr(model.config, 'id2label') and isinstance(model.config.id2label, dict) and model.config.id2label:
        id2label = {int(k): v for k, v in model.config.id2label.items()}
    else:
        id2label = {0: 'B', 1: 'I', 2: 'O'}

    sentences = df['Input Sentence'].astype(str).tolist()
    texts = [s.strip().split() for s in sentences]

    preds_token_level: List[List[int]] = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch_words = texts[start:start+batch_size]
            enc = tokenizer(batch_words, is_split_into_words=True, return_tensors='pt',
                            truncation=True, padding=True)
            inputs = {k: v.to(device) for k, v in enc.items() if k in ('input_ids', 'attention_mask', 'token_type_ids')}
            logits = model(**inputs).logits  # [B, T, C]
            pred_ids = torch.argmax(logits, dim=-1).cpu().tolist()
            preds_token_level.extend(pred_ids)

    word_level = to_word_level_ids(tokenizer, texts, preds_token_level)
    outputs: List[str] = []
    for words, wlabs in zip(texts, word_level):
        outputs.append(reconstruct_sentence(words, wlabs, id2label))
    return outputs


def exact_match(gold: List[str], pred: List[str]) -> float:
    total = len(gold)
    correct = sum(1 for g, p in zip(gold, pred) if str(p) == str(g))
    return (correct / total * 100.0) if total else 0.0


def run_split(name: str, path: str, tokenizer, model, out_dir: str, batch_size: int):
    if not os.path.exists(path):
        print(f"[{name}] File not found: {path}")
        return
    df = pd.read_csv(path)
    if 'Input Sentence' not in df.columns:
        print(f"[{name}] Missing 'Input Sentence' column in {path}")
        return

    preds = predict_for(df, tokenizer, model, batch_size)
    out_df = pd.DataFrame({
        'Input Sentence': df['Input Sentence'],
        'Output Sentence': preds,
    })
    out_path = os.path.join(out_dir, f'predictions_{name}.csv')
    out_df.to_csv(out_path, index=False, encoding='utf-8')
    print(f"[{name}] Wrote {len(out_df)} rows → {out_path}")

    if 'Output Sentence' in df.columns:
        em = exact_match(df['Output Sentence'].astype(str).tolist(), preds)
        print(f"[{name}] Exact Match: {em:.4f}%")
    else:
        print(f"[{name}] Gold 'Output Sentence' not found. Skipping EM computation.")


def main():
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    model = AutoModelForTokenClassification.from_pretrained(args.checkpoint)

    os.makedirs(args.out_dir, exist_ok=True)
    run_split('dev', args.dev, tokenizer, model, args.out_dir, args.batch_size)
    run_split('test', args.test, tokenizer, model, args.out_dir, args.batch_size)


if __name__ == '__main__':
    main()
