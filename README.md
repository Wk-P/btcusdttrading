# btcusdttrading

BTCUSDT 事件合约（10分钟级别二元期权）方向预测。项目分三个阶段：数据采集（现在跑着）、离线打标签、训练+回测。目前还在第一阶段。

## 整体流程

```
main.py (常驻服务)
  Binance WebSocket (trade + depth10@100ms)
    -> 生产者协程：只管收数据，塞进内存队列
    -> 消费者协程：按秒聚合成特征
         data/trades/*.parquet    逐笔成交，原始不聚合
         data/features/*.parquet  秒级特征（OBI、动量、波动率、taker买卖比例...）
       按小时滚动文件，攒够500行才flush一次

label_features.py (离线，攒够数据后手动跑一次)
  读 data/features/*.parquet
  按 --horizon-min 给每一行打"未来N分钟涨/跌"标签
  -> data/labeled/labeled_features.parquet

train_model.py (离线，标签生成后手动跑)
  按时间顺序切分训练/测试集（不随机打乱，避免未来信息泄漏）
  训练 lightgbm 二分类模型
  按事件合约赔率（赢80% / 输100%本金，盈亏平衡胜率55.56%）做阈值扫描回测
  -> 打印 accuracy / AUC / 各置信度阈值下的胜率、覆盖率、总收益
```

## 现在的状态

- `btcusdttrading.service` 已经作为 systemd 服务在后台跑，持续采集数据
- 计划采集到 **2026-09-02** 左右，尽量覆盖不同行情状态（涨/跌/震荡）
- `label_features.py`、`train_model.py` 已经写好，等数据攒够了直接跑

## 接下来该做什么

**现在到 2026-09-02 之间：不需要做任何事**，让采集服务持续跑。可以偶尔检查一下服务状态：

```bash
systemctl status btcusdttrading      # 服务是否存活
journalctl -u btcusdttrading -f      # 实时日志，看有没有频繁断线重连
```

**2026-09-02（或数据攒够之后）：跑这两步**

```bash
cd /home/ecs-user/projects/github/btcusdttrading
python label_features.py --horizon-min 10   # 第一步：打标签
python train_model.py                        # 第二步：训练 + 回测
```

**怎么看结果：**

`train_model.py` 最后会打印一张阈值扫描表：

| thresh | n_bets | coverage | win_rate | total_ret | avg_ret |
|--------|--------|----------|----------|-----------|---------|

- 关注 `win_rate` 是否明显超过 **55.56%**（当前赔率下的盈亏平衡点）
- 同时看 `n_bets`/`coverage`：胜率高但下注次数太少（比如只有几十次）说明样本不够、不能直接信
- 如果所有阈值下 `win_rate` 都在 55.56% 附近徘徊，说明现有特征预测力不够，需要回来加特征、调整聚合窗口，或者换预测周期，然后重新采集/训练

## 环境

依赖用 [uv](https://docs.astral.sh/uv/) 管理：

```bash
uv sync
```

## 部署

`deploy/btcusdttrading.service` 是采集服务的 systemd unit 文件，已经安装在 `/etc/systemd/system/` 并 enable。改了 `main.py` 之后需要重启服务：

```bash
sudo systemctl restart btcusdttrading
```
