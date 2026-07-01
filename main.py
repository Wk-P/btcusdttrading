import asyncio
import os
import websockets
import orjson
import csv
import time
from datetime import datetime

# Binance WebSocket 地址
BINANCE_WS_URL = "wss://stream.binance.com:9443/stream?streams=btcusdt@trade/btcusdt@depth10@100ms"

def calculate_obi(bids, asks):
    """计算盘口不平衡度 (基于前10档挂单量)"""
    vol_bid = sum(float(b[1]) for b in bids)
    vol_ask = sum(float(a[1]) for a in asks)
    
    if vol_bid + vol_ask == 0:
        return 0
    return (vol_bid - vol_ask) / (vol_bid + vol_ask)

async def market_data_producer(queue: asyncio.Queue):
    """生产者：从交易所接收数据并推入内存队列（断线自动重连）"""
    backoff = 1
    while True:
        try:
            print(f"[{datetime.now()}] Producer connecting to Binance...")
            async with websockets.connect(BINANCE_WS_URL) as ws:
                print(f"[{datetime.now()}] Producer connected.")
                backoff = 1
                while True:
                    msg = await ws.recv()
                    data = orjson.loads(msg)
                    # 无阻塞地将数据推入队列
                    await queue.put(data)
        except Exception as e:
            print(f"[{datetime.now()}] Producer Connection Error: {e}. Reconnecting in {backoff}s...")
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 60)

async def feature_engine_consumer(queue: asyncio.Queue):
    """消费者：从内存队列取出数据并计算特征"""
    print("Consumer Started. Waiting for data...")
    current_price = None
    
    file_exists = os.path.exists("btc_tick_features.csv")
    with open("btc_tick_features.csv", "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp_ms", "datetime", "price", "obi"])

        while True:
            payload = await queue.get()
            stream = payload.get('stream', '')
            data = payload.get('data', {})
            
            # 1. 更新最新成交价
            if 'trade' in stream:
                current_price = float(data.get('p', 0))
                
            # 2. 计算盘口 OBI 特征
            elif 'depth' in stream:
                bids = data.get('bids', [])
                asks = data.get('asks', [])
                obi = calculate_obi(bids, asks)
                
                if current_price:
                    ts = int(time.time() * 1000)
                    dt = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]

                    writer.writerow([ts, dt, current_price, obi])
                    f.flush()
            # 标记任务完成
            queue.task_done()

async def main():
    # 创建一个内存队列，充当轻量级的 "Kafka"
    message_queue = asyncio.Queue()
    
    # 并发运行生产者和消费者
    await asyncio.gather(
        market_data_producer(message_queue),
        feature_engine_consumer(message_queue)
    )

if __name__ == "__main__":
    # Windows 下有时需要指定 EventLoopPolicy 以避免一些底层 socket 报错
    import sys
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        
    while True:
        try:
            asyncio.run(main())
        except KeyboardInterrupt:
            print("\nBot stopped by user.")
            break
        except Exception as e:
            print(f"[{datetime.now()}] Fatal error, restarting in 5s: {e}")
            time.sleep(5)