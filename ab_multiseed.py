"""多随机种子重复 A/B 对比，消除单次划分的偶然性"""
import json
import numpy as np
import torch
import ab_text_compare as M

d = torch.load("ab_cmp_stats.pt", map_location="cpu", weights_only=False)
A, B, y = d["A_with_text"], d["B_no_text"], d["labels"]

SEEDS = [42, 0, 1, 2, 3, 7, 13, 21, 100, 2024]

rows = []
for s in SEEDS:
    cfg = M.ABConfig()
    cfg.seed = s
    cfg.epochs = 200
    mA, _ = M.train_one_group("A", A, y, cfg, verbose=False)
    mB, _ = M.train_one_group("B", B, y, cfg, verbose=False)
    rows.append((s, mA, mB))
    print(f"seed={s:5d} | A acc={mA['accuracy']:.4f} auc={mA['auc']:.4f} f1={mA['f1']:.4f}"
          f" | B acc={mB['accuracy']:.4f} auc={mB['auc']:.4f} f1={mB['f1']:.4f}")

print("\n" + "=" * 74)
print(f"多随机种子平均 ({len(SEEDS)} seeds, 均值±标准差)")
print("=" * 74)
print(f"{'指标':<12}{'A 有文字':>18}{'B 无文字':>18}{'Δ(A-B)':>14}")
print("-" * 74)
summary = {}
for k in ["accuracy", "precision", "recall", "f1", "auc"]:
    a = np.array([r[1][k] for r in rows])
    b = np.array([r[2][k] for r in rows])
    summary[k] = {"A_mean": a.mean(), "A_std": a.std(),
                  "B_mean": b.mean(), "B_std": b.std(),
                  "delta": (a - b).mean()}
    print(f"{k:<12}{a.mean():>10.4f}±{a.std():.4f}{b.mean():>10.4f}±{b.std():.4f}"
          f"{(a-b).mean():>+14.4f}")
print("-" * 74)

# 胜负统计
wins = sum(1 for r in rows if r[1]["auc"] > r[2]["auc"])
print(f"\nAUC 上 A 优于 B 的次数: {wins}/{len(SEEDS)}")
acc_wins = sum(1 for r in rows if r[1]["accuracy"] > r[2]["accuracy"])
print(f"Accuracy 上 A 优于 B 的次数: {acc_wins}/{len(SEEDS)}")

json.dump({"per_seed": [{"seed": r[0], "A": r[1], "B": r[2]} for r in rows],
           "summary": summary,
           "auc_A_wins": wins, "acc_A_wins": acc_wins, "n_seeds": len(SEEDS)},
          open("ab_cmp_multiseed.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=2)
print("\n[保存] ab_cmp_multiseed.json")
