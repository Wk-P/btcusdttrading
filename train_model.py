"""
训练 + 回测脚本：读取 label_features.py 生成的 data/labeled/labeled_features.parquet，
训练 lightgbm 二分类模型预测未来 N 分钟涨跌，并按事件合约的固定赔率结构回测期望收益。

赔付结构（二元期权）：押对赢回 WIN_PAYOUT * 本金，押错输掉 100% 本金。
按此赔率，模型胜率需要 > 1 / (1 + WIN_PAYOUT) 才能有正期望，默认 80% 赔率对应盈亏平衡点 55.56%。

用法:
    python train_model.py --labeled-path data/labeled/labeled_features.parquet
"""
import argparse
import os
from datetime import datetime, timezone

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

WIN_PAYOUT = 0.8  # 押对赢回的比例，押错本金全亏
MODELS_DIR = "models"

FEATURE_COLS = [
    "obi_mean", "obi_last", "obi_min", "obi_max",
    "trade_count", "volume", "taker_buy_volume", "taker_buy_ratio",
    "momentum_30s", "volatility_30s",
]


def time_split(df: pd.DataFrame, test_ratio: float):
    """按时间顺序切分，禁止随机打乱——否则测试集会看到训练集的未来信息。"""
    split_idx = int(len(df) * (1 - test_ratio))
    return df.iloc[:split_idx], df.iloc[split_idx:]


def backtest(y_true: np.ndarray, proba_up: np.ndarray, threshold: float):
    """只在模型置信度足够高时才下注，按二元期权赔率结算期望收益。"""
    long_mask = proba_up >= threshold
    short_mask = proba_up <= (1 - threshold)
    bet_mask = long_mask | short_mask
    n_bets = bet_mask.sum()
    if n_bets == 0:
        return {"n_bets": 0, "win_rate": None, "total_return": 0.0, "avg_return_per_bet": None}

    predicted_up = long_mask[bet_mask]
    actual_up = y_true[bet_mask].astype(bool)
    correct = predicted_up == actual_up

    pnl = np.where(correct, WIN_PAYOUT, -1.0)
    return {
        "n_bets": int(n_bets),
        "coverage": n_bets / len(y_true),
        "win_rate": correct.mean(),
        "total_return": pnl.sum(),
        "avg_return_per_bet": pnl.mean(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labeled-path", default="data/labeled/labeled_features.parquet")
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--thresholds", type=float, nargs="+",
                         default=[0.55, 0.60, 0.65, 0.70, 0.75])
    parser.add_argument("--horizon-min", type=int, default=10,
                         help="仅用于给保存的模型文件命名，和 label_features.py 的 --horizon-min 对应")
    parser.add_argument("--no-save", action="store_true", help="只评估，不保存模型")
    args = parser.parse_args()

    df = pd.read_parquet(args.labeled_path).sort_values("timestamp_ms").reset_index(drop=True)
    print(f"Loaded {len(df)} labeled rows.")

    train_df, test_df = time_split(df, args.test_ratio)
    print(f"Train: {len(train_df)} rows | Test: {len(test_df)} rows "
          f"(split at {train_df['timestamp_ms'].max()} -> {test_df['timestamp_ms'].min()})")

    X_train, y_train = train_df[FEATURE_COLS], train_df["label"]
    X_test, y_test = test_df[FEATURE_COLS], test_df["label"]

    model = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=50,
    )
    model.fit(X_train, y_train)

    proba_up = model.predict_proba(X_test)[:, 1]
    preds = (proba_up >= 0.5).astype(int)

    print(f"\nAccuracy @0.5: {accuracy_score(y_test, preds):.4f}")
    print(f"AUC: {roc_auc_score(y_test, proba_up):.4f}")
    print(f"Test set up-ratio (baseline): {y_test.mean():.4f}")

    breakeven = 1 / (1 + WIN_PAYOUT)
    print(f"\nBreakeven win rate at {WIN_PAYOUT:.0%} payout: {breakeven:.4f}")
    print("\nThreshold sweep (only bet when confidence is high enough):")
    print(f"{'thresh':>8} {'n_bets':>8} {'coverage':>10} {'win_rate':>10} {'total_ret':>10} {'avg_ret':>10}")
    for t in args.thresholds:
        r = backtest(y_test.to_numpy(), proba_up, t)
        if r["n_bets"] == 0:
            print(f"{t:>8.2f} {'0':>8} {'-':>10} {'-':>10} {'-':>10} {'-':>10}")
            continue
        print(f"{t:>8.2f} {r['n_bets']:>8} {r['coverage']:>10.4f} {r['win_rate']:>10.4f} "
              f"{r['total_return']:>10.2f} {r['avg_return_per_bet']:>10.4f}")

    importances = pd.Series(model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
    print("\nFeature importance:")
    print(importances.to_string())

    if not args.no_save:
        os.makedirs(MODELS_DIR, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        model_path = os.path.join(MODELS_DIR, f"lgbm_{args.horizon_min}min_{stamp}.txt")
        model.booster_.save_model(model_path)
        print(f"\nModel saved to {model_path}")
        print("Load it later with: lightgbm.Booster(model_file=<path>)")


if __name__ == "__main__":
    main()
