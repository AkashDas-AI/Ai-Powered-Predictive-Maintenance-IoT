"""
Finite Streaming Demonstration Runner for AI-Predictive-Maintenance-IoT.

Orchestrates the virtual telemetry simulator, bounded sliding-window buffer,
and streaming inference service, producing formatted tabular output and
rigorous latency metrics.
"""

from pathlib import Path
import sys
from typing import Any, Dict, List, Optional
import numpy as np

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.streaming.simulator import TelemetrySimulator
from src.streaming.buffer import SlidingWindowBuffer
from src.streaming.inference import StreamingInferenceService


def run_streaming_demo(
    num_records: int = 25,
    window_size: int = 10,
    random_state: int = 42,
    shuffle: bool = False,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Executes a finite streaming inference demonstration.

    Steps:
        1. Initializes TelemetrySimulator from AI4I benchmark data.
        2. Maintains SlidingWindowBuffer for recent sensor readings.
        3. Invokes StreamingInferenceService for low-latency scoring.
        4. Calculates rolling window diagnostics (mean, min, max, std).
        5. Computes high-resolution latency statistics (P50, P95, mean, min, max).

    Args:
        num_records: Number of simulated records to process (default: 25).
        window_size: Capacity of the sliding-window buffer (default: 10).
        random_state: Seed for reproducible record selection.
        shuffle: If True, shuffles records; if False, replays in sequential order.
        verbose: If True, prints formatted console logs.

    Returns:
        Dict[str, Any]: Summary containing processed records, latency metrics, and alert counts.
    """
    simulator = TelemetrySimulator(random_state=random_state, shuffle=shuffle)
    buffer = SlidingWindowBuffer(window_size=window_size)
    service = StreamingInferenceService()

    latencies_ms: List[float] = []
    alert_counts: Dict[str, int] = {}
    inference_results: List[Dict[str, Any]] = []

    if verbose:
        print("\n" + "=" * 90)
        print("  IoT Telemetry Streaming & Low-Latency Inference Demonstration (Phase 6)")
        print("=" * 90)
        print(f"Total Stream Samples: {num_records} | Buffer Capacity: {window_size} | "
              f"Decision Threshold: {service.threshold:.4f}")
        print("Simulation Mode: Replaying AI4I benchmark records (Software simulation only)")
        print("-" * 90)
        header = (
            f"{'Seq':<5} | {'Type':<4} | {'Temp Diff [K]':<13} | {'Speed [rpm]':<11} | "
            f"{'Torque [Nm]':<11} | {'P(Fail)':<8} | {'Alert':<8} | {'HI (%)':<7} | {'Latency':<9}"
        )
        print(header)
        print("-" * 90)

    for record in simulator.stream(max_records=num_records):
        # 1. Ingest into bounded buffer
        buffer.append(record)

        # 2. Score via streaming inference service
        result = service.predict_record(record)
        inference_results.append(result)

        lat = result["inference_latency_ms"]
        latencies_ms.append(lat)

        alert = result["alert_state"]
        alert_counts[alert] = alert_counts.get(alert, 0) + 1

        if verbose:
            # Derive instantaneous physical diffs for display
            temp_diff = record["Process temperature [K]"] - record["Air temperature [K]"]
            prob = result["failure_probability"]
            prob_str = f"{prob:.4f}" if prob is not None else "N/A"
            hi = result["demonstration_health_index"]
            hi_str = f"{hi:.1f}" if hi is not None else "N/A"

            row_str = (
                f"{record['sequence_id']:<5} | "
                f"{record['Type']:<4} | "
                f"{temp_diff:<13.2f} | "
                f"{record['Rotational speed [rpm]']:<11} | "
                f"{record['Torque [Nm]']:<11.1f} | "
                f"{prob_str:<8} | "
                f"{alert:<8} | "
                f"{hi_str:<7} | "
                f"{lat:.2f} ms"
            )
            print(row_str)

    # 3. Rolling diagnostics from final buffer window
    rolling_diagnostics = buffer.compute_rolling_stats()

    # 4. Latency performance statistics
    lat_arr = np.array(latencies_ms)
    p50_latency = float(np.percentile(lat_arr, 50))
    p95_latency = float(np.percentile(lat_arr, 95))
    mean_latency = float(np.mean(lat_arr))
    min_latency = float(np.min(lat_arr))
    max_latency = float(np.max(lat_arr))

    summary = {
        "samples_processed": len(latencies_ms),
        "buffer_final_size": len(buffer),
        "buffer_capacity": buffer.capacity,
        "alert_distribution": alert_counts,
        "rolling_diagnostics_final_window": rolling_diagnostics,
        "latency_statistics_ms": {
            "count": len(lat_arr),
            "median_p50": round(p50_latency, 3),
            "p95": round(p95_latency, 3),
            "mean": round(mean_latency, 3),
            "min": round(min_latency, 3),
            "max": round(max_latency, 3),
        },
        "first_result_sample": inference_results[0] if inference_results else {},
    }

    if verbose:
        print("-" * 90)
        print("Final Buffer Rolling Diagnostics (Last 10 Records - For Operator Display Only):")
        for col, stats in rolling_diagnostics.items():
            print(f"  - {col}: Mean={stats['mean']}, Min={stats['min']}, Max={stats['max']}, Std={stats['std']}")

        print("-" * 90)
        print("Performance Measurement (Local Inference Latency):")
        print(f"  - Sample Count: {len(lat_arr)}")
        print(f"  - Median (P50) Latency: {p50_latency:.3f} ms")
        print(f"  - 95th Percentile (P95): {p95_latency:.3f} ms")
        print(f"  - Mean Latency:         {mean_latency:.3f} ms (Min: {min_latency:.3f} ms, Max: {max_latency:.3f} ms)")
        print(f"  - Alert Distribution:   {alert_counts}")
        print("=" * 90 + "\n")

    return summary


if __name__ == "__main__":
    run_streaming_demo(num_records=25, window_size=10, shuffle=False)

