#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, sys, csv, json
REQ_COL_OUT = "Output Sentence"
def read_rows_require_out(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        rdr = csv.DictReader(f)
        cols = rdr.fieldnames or []
        if REQ_COL_OUT not in cols:
            raise ValueError(f"{os.path.basename(path)} must contain column '{REQ_COL_OUT}'. Found: {cols}")
        return list(rdr)
def main(input_dir, output_dir):
    ref_dir = os.path.join(input_dir, "ref")
    res_dir = os.path.join(input_dir, "res")
    os.makedirs(output_dir, exist_ok=True)
    ref_file = None
    for name in ["dev_gold.csv", "test_gold.csv"]:
        p = os.path.join(ref_dir, name)
        if os.path.exists(p):
            ref_file = p
            break
    if ref_file is None:
        raise FileNotFoundError("Missing reference: dev_gold.csv or test_gold.csv in input/ref/.")
    pred_file = os.path.join(res_dir, "predictions.csv")
    if not os.path.exists(pred_file):
        raise FileNotFoundError("Missing input/res/predictions.csv")
    gold = read_rows_require_out(ref_file)
    pred = read_rows_require_out(pred_file)
    if len(gold) != len(pred):
        raise ValueError(f"Row count mismatch: gold={len(gold)}, predictions={len(pred)}. They must be equal.")
    total, correct = len(gold), 0
    for g, p in zip(gold, pred):
        if p.get(REQ_COL_OUT, "") == g.get(REQ_COL_OUT, ""):
            correct += 1
    acc = (correct / total) * 100.0 if total else 0.0
    with open(os.path.join(output_dir, "scores.txt"), "w", encoding="utf-8") as f:
        f.write(f"exact_match: {acc:.6f}\n")
    with open(os.path.join(output_dir, "scores.json"), "w", encoding="utf-8") as f:
        json.dump({"exact_match": acc}, f, ensure_ascii=False)
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python score.py <input_dir> <output_dir>")
        sys.exit(2)
    main(sys.argv[1], sys.argv[2])
