#!/usr/bin/env python3
"""
Extract summary statistics from benchmark results JSONL file.
Usage: python extract_stats.py <jsonl_file>
"""

import json
import sys
from pathlib import Path
from collections import defaultdict
import statistics


def extract_stats(jsonl_file):
    """Extract and display summary statistics from JSONL file."""
    
    if not Path(jsonl_file).exists():
        print(f"Error: File '{jsonl_file}' not found")
        sys.exit(1)
    
    records = []
    with open(jsonl_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping line {line_num}: {e}", file=sys.stderr)
    
    if not records:
        print("No valid records found in file")
        sys.exit(1)
    
    # Extract numeric fields
    numeric_fields = {
        'ttft_ms': [],
        'tps': [],
        'tpot_ms': [],
        'itl_ms': [],
        'prompt_tokens': [],
        'output_tokens': [],
        'total_duration_ms': [],
        'generation_ms': [],
        'vram_used_gb': [],
        'vram_free_gb': [],
        'vram_total_gb': [],
        'mbu_pct': [],
        'model_size_gb': [],
        'gpu_temp_c': [],
        'gpu_power_w': [],
        'gpu_utilization': [],
    }
    
    # Extract categorical fields
    categories = defaultdict(int)
    models = defaultdict(int)
    gpu_names = defaultdict(int)
    errors = defaultdict(int)
    
    for record in records:
        # Collect numeric values
        for field in numeric_fields:
            if field in record and record[field] is not None:
                numeric_fields[field].append(record[field])
        
        # Collect categorical data
        if 'category' in record:
            categories[record['category']] += 1
        if 'model' in record:
            models[record['model']] += 1
        if 'gpu_name' in record:
            gpu_names[record['gpu_name']] += 1
        if 'error' in record and record['error'] is not None:
            errors[record['error']] += 1
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"BENCHMARK SUMMARY STATISTICS")
    print(f"{'='*70}")
    print(f"\nTotal records: {len(records)}")
    
    print(f"\n{'-'*70}")
    print("CATEGORICAL BREAKDOWN")
    print(f"{'-'*70}")
    
    if categories:
        print("\nCategories:")
        for cat, count in sorted(categories.items()):
            print(f"  {cat}: {count}")
    
    if models:
        print("\nModels:")
        for model, count in sorted(models.items()):
            print(f"  {model}: {count}")
    
    if gpu_names:
        print("\nGPUs:")
        for gpu, count in sorted(gpu_names.items()):
            print(f"  {gpu}: {count}")
    
    if errors:
        print("\nErrors:")
        for error, count in sorted(errors.items()):
            print(f"  {error}: {count}")
    
    print(f"\n{'-'*70}")
    print("NUMERIC STATISTICS")
    print(f"{'-'*70}\n")
    
    # Calculate and display statistics for each numeric field
    for field in sorted(numeric_fields.keys()):
        values = numeric_fields[field]
        if not values:
            continue
        
        min_val = min(values)
        max_val = max(values)
        mean_val = statistics.mean(values)
        
        if len(values) > 1:
            median_val = statistics.median(values)
            stdev_val = statistics.stdev(values)
            print(f"{field:20s}: count={len(values):3d} | "
                  f"min={min_val:10.2f} | max={max_val:10.2f} | "
                  f"mean={mean_val:10.2f} | median={median_val:10.2f} | "
                  f"stdev={stdev_val:10.2f}")
        else:
            print(f"{field:20s}: count={len(values):3d} | "
                  f"value={mean_val:10.2f}")
    
    print(f"\n{'='*70}\n")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python extract_stats.py <jsonl_file>")
        sys.exit(1)
    
    extract_stats(sys.argv[1])
