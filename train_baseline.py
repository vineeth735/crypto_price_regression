#!/usr/bin/env python3
"""Train a small leakage-safe regression baseline using only Python's standard library."""

import argparse
import csv
import math


def solve_linear_system(matrix, vector):
    """Solve Ax=b with Gaussian elimination and partial pivoting."""
    size = len(vector)
    augmented = [matrix[i][:] + [vector[i]] for i in range(size)]
    for pivot_column in range(size):
        pivot_row = max(range(pivot_column, size), key=lambda row: abs(augmented[row][pivot_column]))
        if abs(augmented[pivot_row][pivot_column]) < 1e-12:
            raise ValueError("Features are not independent enough to fit this model.")
        augmented[pivot_column], augmented[pivot_row] = augmented[pivot_row], augmented[pivot_column]
        pivot = augmented[pivot_column][pivot_column]
        augmented[pivot_column] = [value / pivot for value in augmented[pivot_column]]
        for row in range(size):
            if row == pivot_column:
                continue
            factor = augmented[row][pivot_column]
            augmented[row] = [value - factor * reference for value, reference in zip(augmented[row], augmented[pivot_column])]
    return [row[-1] for row in augmented]


def metrics(actual, predicted):
    errors = [observed - estimate for observed, estimate in zip(actual, predicted)]
    mae = sum(abs(error) for error in errors) / len(errors)
    rmse = math.sqrt(sum(error * error for error in errors) / len(errors))
    average = sum(actual) / len(actual)
    total = sum((value - average) ** 2 for value in actual)
    r2 = 1 - sum(error * error for error in errors) / total if total else 0.0
    return mae, rmse, r2


def main():
    parser = argparse.ArgumentParser(description="Fit next-day closing-price regression.")
    parser.add_argument("--data", default="btc_usd_daily.csv", help="CSV created by download_data.py")
    args = parser.parse_args()

    with open(args.data, newline="") as handle:
        rows = list(csv.DictReader(handle))
    closes = [float(row["close"]) for row in rows]
    volumes = [float(row["volume"]) for row in rows]

    # Today's close is predicted using information available at the end of yesterday.
    samples = []
    for index in range(14, len(rows)):
        samples.append(([closes[index - 1], closes[index - 7], closes[index - 14], volumes[index - 1]], closes[index], rows[index]["date"]))
    split = int(len(samples) * 0.8)
    train, test = samples[:split], samples[split:]
    if len(train) < 20 or not test:
        raise ValueError("Need more data. Download at least one year of daily prices.")

    means = [sum(features[column] for features, _, _ in train) / len(train) for column in range(4)]
    scales = [math.sqrt(sum((features[column] - means[column]) ** 2 for features, _, _ in train) / len(train)) or 1 for column in range(4)]
    def normalized(features):
        return [1.0] + [(features[column] - means[column]) / scales[column] for column in range(4)]

    design = [normalized(features) for features, _, _ in train]
    targets = [target for _, target, _ in train]
    width = len(design[0])
    gram = [[sum(row[i] * row[j] for row in design) for j in range(width)] for i in range(width)]
    right_side = [sum(row[i] * target for row, target in zip(design, targets)) for i in range(width)]
    coefficients = solve_linear_system(gram, right_side)

    actual = [target for _, target, _ in test]
    predicted = [sum(weight * value for weight, value in zip(coefficients, normalized(features))) for features, _, _ in test]
    mae, rmse, r2 = metrics(actual, predicted)
    print(f"Rows loaded: {len(rows)}")
    print(f"Training samples: {len(train)} | Test samples: {len(test)}")
    print(f"Test date range: {test[0][2]} to {test[-1][2]}")
    print(f"MAE:  ${mae:,.2f}")
    print(f"RMSE: ${rmse:,.2f}")
    print(f"R-squared: {r2:.4f}")
    print(f"Last test prediction: ${predicted[-1]:,.2f} | Actual: ${actual[-1]:,.2f}")


if __name__ == "__main__":
    main()
