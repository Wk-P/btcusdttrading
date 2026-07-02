import asyncio
import collections
import logging
import os
import signal
import time
from datetime import datetime, timezone

import orjson
import pyarrow as pa
import pyarrow.parquet as pq
import websockets

SYMBOL = "btcusdt"
BINANCE_WS_URL = f"wss://stream.binance.com:9443/stream?streams={SYMBOL}@trade/{SYMBOL}@depth10@100ms"

DATA_DIR = "data"
TRADES_DIR = os.path.join(DATA_DIR, "trades")
FEATURES_DIR = os.path.join(DATA_DIR, "features")
os.makedirs(TRADES_DIR, exist_ok=True)
os.makedirs(FEATURES_DIR, exist_ok=True)

MOMENTUM_WINDOW_SEC = 30  # 用于动量/波动率的回看窗口

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("collector")

TRADES_SCHEMA = pa.schema([
    ("timestamp_ms", pa.int64()),
    ("price", pa.float64()),
    ("qty", pa.float64()),
    ("is_taker_buy", pa.bool_()),
])

FEATURES_SCHEMA = pa.schema([
    ("timestamp_ms", pa.int64()),
    ("price", pa.float64()),
    ("obi_mean", pa.float64()),
    ("obi_last", pa.float64()),
    ("obi_min", pa.float64()),
    ("obi_max", pa.float64()),
    ("trade_count", pa.int32()),
    ("volume", pa.float64()),
    ("taker_buy_volume", pa.float64()),
    ("taker_buy_ratio", pa.float64()),
    ("momentum_30s", pa.float64()),
    ("volatility_30s", pa.float64()),
])


def calculate_obi(bids, asks):
    """计算盘口不平衡度 (基于前10档挂单量)"""
    vol_bid = sum(float(b[1]) for b in bids)
    vol_ask = sum(float(a[1]) for a in asks)
    if vol_bid + vol_ask == 0:
        return 0.0
    return (vol_bid - vol_ask) / (vol_bid + vol_ask)


class HourlyParquetWriter:
    """按UTC小时滚动写parquet文件，定期flush一批行。

    ParquetWriter 只有在 close() 时才写入 footer，写入中的文件是无法被读取的
    残缺文件。按小时(而非按天)轮转，是为了让数据尽快"落定"变得可读，
    并缩小进程异常退出时可能丢失/损坏的窗口。
    """

    def __init__(self, directory: str, prefix: str, schema: pa.Schema, flush_rows: int = 500):
        self.directory = directory
        self.prefix = prefix
        self.schema = schema
        self.flush_rows = flush_rows
        self._writer = None
        self._current_key = None
        self._buffer = []

    def _path_for(self, key: str) -> str:
        return os.path.join(self.directory, f"{self.prefix}_{key}.parquet")

    def _rotate_if_needed(self):
        key = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H")
        if key != self._current_key:
            self._flush()
            if self._writer is not None:
                self._writer.close()
            self._current_key = key
            self._writer = pq.ParquetWriter(self._path_for(key), self.schema)
            log.info(f"{self.prefix}: rotated to new file for hour {key}")

    def add(self, row: dict):
        self._rotate_if_needed()
        self._buffer.append(row)
        if len(self._buffer) >= self.flush_rows:
            self._flush()

    def _flush(self):
        if not self._buffer or self._writer is None:
            return
        table = pa.Table.from_pylist(self._buffer, schema=self.schema)
        self._writer.write_table(table)
        self._buffer.clear()

    def close(self):
        self._flush()
        if self._writer is not None:
            self._writer.close()


async def market_data_producer(queue: asyncio.Queue):
    """生产者：从交易所接收数据并推入内存队列（断线自动重连）"""
    backoff = 1
    reconnect_count = 0
    last_disconnect_at = None
    while True:
        try:
            log.info("Producer connecting to Binance...")
            async with websockets.connect(BINANCE_WS_URL) as ws:
                if last_disconnect_at is not None:
                    gap_sec = time.time() - last_disconnect_at
                    log.warning(f"Producer reconnected after {gap_sec:.1f}s gap (reconnect #{reconnect_count})")
                else:
                    log.info("Producer connected.")
                backoff = 1
                while True:
                    msg = await ws.recv()
                    data = orjson.loads(msg)
                    await queue.put(data)
        except Exception as e:
            reconnect_count += 1
            last_disconnect_at = time.time()
            log.error(f"Producer connection error: {e}. Reconnecting in {backoff}s... (total reconnects: {reconnect_count})")
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 60)


async def feature_engine_consumer(queue: asyncio.Queue):
    """消费者：从内存队列取出数据，全量落盘trade，按1秒聚合落盘特征"""
    log.info("Consumer started. Waiting for data...")

    trades_writer = HourlyParquetWriter(TRADES_DIR, "trades", TRADES_SCHEMA)
    features_writer = HourlyParquetWriter(FEATURES_DIR, "features", FEATURES_SCHEMA)

    current_price = None
    price_history = collections.deque()  # (ts_ms, price)，用于动量/波动率

    bucket_second = None
    bucket_obi = []
    bucket_trade_count = 0
    bucket_volume = 0.0
    bucket_taker_buy_volume = 0.0

    def flush_bucket(ts_ms: int):
        nonlocal bucket_obi, bucket_trade_count, bucket_volume, bucket_taker_buy_volume
        if not bucket_obi or current_price is None:
            bucket_obi = []
            bucket_trade_count = 0
            bucket_volume = 0.0
            bucket_taker_buy_volume = 0.0
            return

        while price_history and ts_ms - price_history[0][0] > MOMENTUM_WINDOW_SEC * 1000:
            price_history.popleft()

        if len(price_history) >= 2:
            prices = [p for _, p in price_history]
            momentum = (prices[-1] - prices[0]) / prices[0] if prices[0] else 0.0
            mean_p = sum(prices) / len(prices)
            volatility = (sum((p - mean_p) ** 2 for p in prices) / len(prices)) ** 0.5
        else:
            momentum = 0.0
            volatility = 0.0

        features_writer.add({
            "timestamp_ms": ts_ms,
            "price": current_price,
            "obi_mean": sum(bucket_obi) / len(bucket_obi),
            "obi_last": bucket_obi[-1],
            "obi_min": min(bucket_obi),
            "obi_max": max(bucket_obi),
            "trade_count": bucket_trade_count,
            "volume": bucket_volume,
            "taker_buy_volume": bucket_taker_buy_volume,
            "taker_buy_ratio": (bucket_taker_buy_volume / bucket_volume) if bucket_volume > 0 else 0.0,
            "momentum_30s": momentum,
            "volatility_30s": volatility,
        })

        bucket_obi = []
        bucket_trade_count = 0
        bucket_volume = 0.0
        bucket_taker_buy_volume = 0.0

    try:
        while True:
            payload = await queue.get()
            stream = payload.get("stream", "")
            data = payload.get("data", {})
            ts_ms = int(time.time() * 1000)
            sec = ts_ms // 1000

            if bucket_second is None:
                bucket_second = sec
            elif sec != bucket_second:
                flush_bucket(bucket_second * 1000)
                bucket_second = sec

            if "trade" in stream:
                price = float(data.get("p", 0))
                qty = float(data.get("q", 0))
                is_taker_buy = not data.get("m", False)  # m=True 表示买方是maker，即卖方是taker

                current_price = price
                price_history.append((ts_ms, price))

                trades_writer.add({
                    "timestamp_ms": ts_ms,
                    "price": price,
                    "qty": qty,
                    "is_taker_buy": is_taker_buy,
                })

                bucket_trade_count += 1
                bucket_volume += qty
                if is_taker_buy:
                    bucket_taker_buy_volume += qty

            elif "depth" in stream:
                bids = data.get("bids", [])
                asks = data.get("asks", [])
                bucket_obi.append(calculate_obi(bids, asks))

            queue.task_done()
    except asyncio.CancelledError:
        if bucket_second is not None:
            flush_bucket(bucket_second * 1000)
        trades_writer.close()
        features_writer.close()
        log.info("Consumer shut down cleanly, writers closed.")
        raise


async def main():
    message_queue = asyncio.Queue()
    producer_task = asyncio.create_task(market_data_producer(message_queue))
    consumer_task = asyncio.create_task(feature_engine_consumer(message_queue))

    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, stop_event.set)

    await stop_event.wait()
    log.info("Shutdown signal received, stopping...")
    producer_task.cancel()
    consumer_task.cancel()
    await asyncio.gather(producer_task, consumer_task, return_exceptions=True)


if __name__ == "__main__":
    import sys

    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log.info("Bot stopped by user.")
