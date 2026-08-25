#!/usr/bin/env python3
"""Download multi-year daily cryptocurrency candles from Coinbase Exchange."""

import argparse
import csv
import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen


API_URL = "https://api.exchange.coinbase.com/products/{product}/candles"


def iso_timestamp(value):
    return value.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def fetch_candles(product, start, end):
    """Coinbase returns at most 300 candles per request, so fetch in chunks."""
    candles_by_time = {}
    chunk_start = start
    while chunk_start < end:
        chunk_end = min(chunk_start + timedelta(days=299), end)
        query = urlencode(
            {"granularity": 86400, "start": iso_timestamp(chunk_start), "end": iso_timestamp(chunk_end)}
        )
        request = Request(
            f"{API_URL.format(product=product)}?{query}",
            headers={"User-Agent": "crypto-regression-student-project/1.0"},
        )
        with urlopen(request, timeout=30) as response:
            payload = json.load(response)
        if not isinstance(payload, list):
            raise RuntimeError(f"Unexpected API response: {payload}")
        for candle in payload:
            candles_by_time[candle[0]] = candle
        print(f"Downloaded through {chunk_end.date()} ({len(candles_by_time)} rows)")
        chunk_start = chunk_end + timedelta(days=1)
        time.sleep(0.25)
    return [candles_by_time[key] for key in sorted(candles_by_time)]


def main():
    parser = argparse.ArgumentParser(description="Download daily crypto candles for regression.")
    parser.add_argument("--product", default="BTC-USD", help="Coinbase product, e.g. BTC-USD or ETH-USD")
    parser.add_argument("--years", type=int, default=8, help="How many years of daily data to download")
    parser.add_argument("--output", default="btc_usd_daily.csv", help="CSV output path")
    args = parser.parse_args()
    if args.years < 1:
        raise ValueError("--years must be at least 1")

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=365 * args.years)
    candles = fetch_candles(args.product.upper(), start, end)
    output = Path(args.output)
    with output.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["timestamp", "date", "low", "high", "open", "close", "volume"])
        for timestamp, low, high, open_price, close, volume in candles:
            date = datetime.fromtimestamp(timestamp, tz=timezone.utc).date().isoformat()
            writer.writerow([timestamp, date, low, high, open_price, close, volume])
    print(f"Saved {len(candles)} daily rows to {output.resolve()}")


if __name__ == "__main__":
    main()
