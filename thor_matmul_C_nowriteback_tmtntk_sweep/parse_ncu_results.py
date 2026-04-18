#!/usr/bin/env python3

import re
import subprocess
import json
import csv
import io
import sys
from datetime import datetime
from math import ceil
from pathlib import Path

RESULT_FIELDNAMES = [
    'tile_m',
    'tile_n',
    'tile_k',
    'M',
    'N',
    'K',
    'm_cycles',
    'n_cycles',
    'ddr_read_bytes_theoretical',
    'ddr_write_bytes_theoretical',
    'm_ddr_read_bytes_actual',
    'n_ddr_read_bytes_actual',
    'm_ddr_write_bytes_actual',
    'n_ddr_write_bytes_actual',
    'filename',
    'experiment'
]


def parse_numeric(value):
    """Best-effort parsing for numeric CSV fields exported by ncu."""
    if value is None:
        return None

    text = str(value).strip().replace(',', '')
    if not text:
        return None

    try:
        return float(text)
    except ValueError:
        return None


def extract_tile_dims_from_text(text):
    if not text:
        return {}

    match = re.search(r'tm[_-]?(\d+)[^\d]+tn[_-]?(\d+)[^\d]+tk[_-]?(\d+)', text)
    if match:
        return {
            'tile_m': int(match.group(1)),
            'tile_n': int(match.group(2)),
            'tile_k': int(match.group(3))
        }

    dims = {}
    tm_match = re.search(r'tm[_-]?(\d+)', text)
    tn_match = re.search(r'tn[_-]?(\d+)', text)
    tk_match = re.search(r'tk[_-]?(\d+)', text)
    if tm_match:
        dims['tile_m'] = int(tm_match.group(1))
    if tn_match:
        dims['tile_n'] = int(tn_match.group(1))
    if tk_match:
        dims['tile_k'] = int(tk_match.group(1))
    return dims


def extract_problem_dims_from_text(text):
    if not text:
        return {}

    match = re.search(r'M[_-]?(\d+)[^\d]+N[_-]?(\d+)[^\d]+K[_-]?(\d+)', text)
    if match:
        return {
            'M': int(match.group(1)),
            'N': int(match.group(2)),
            'K': int(match.group(3))
        }

    dims = {}
    m_match = re.search(r'M[_-]?(\d+)', text)
    n_match = re.search(r'N[_-]?(\d+)', text)
    k_match = re.search(r'(^|[\\/])K[_-]?(\d+)(?=$|[\\/])', text)
    if m_match:
        dims['M'] = int(m_match.group(1))
    if n_match:
        dims['N'] = int(n_match.group(1))
    if k_match:
        dims['K'] = int(k_match.group(2))
    return dims


def detect_kernel_role(kernel_name):
    name = (kernel_name or '').lower()
    if 'matmul_kernel_m' in name or 'kernel_m' in name:
        return 'm'
    if 'matmul_kernel_n' in name or 'kernel_n' in name:
        return 'n'
    return None


def lookup_metric_value(row, metric_names):
    for metric_name in metric_names:
        value = parse_numeric(row.get(metric_name, ''))
        if value is not None:
            return value
    return None


def build_experiment_name(tile_key, exp_dir_name):
    return f"{tile_key}_{exp_dir_name}" if tile_key else exp_dir_name


def get_experiment_output_dir(output_root, tile_key, exp_dir_name):
    if tile_key:
        return output_root / tile_key / exp_dir_name
    return output_root / exp_dir_name


def write_results_csv(output_csv, results):
    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDNAMES)
        writer.writeheader()
        for record in results:
            row = {field: record.get(field, '') for field in RESULT_FIELDNAMES}
            writer.writerow(row)


class TeeWriter:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def setup_logging(output_dir):
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = logs_dir / f"parse_ncu_results_{timestamp}.log"
    log_file = open(log_path, 'w')
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = TeeWriter(original_stdout, log_file)
    sys.stderr = TeeWriter(original_stderr, log_file)
    return log_path, log_file, original_stdout, original_stderr


def teardown_logging(log_file, original_stdout, original_stderr):
    sys.stdout = original_stdout
    sys.stderr = original_stderr
    log_file.close()


def parse_ncu_report(ncu_rep_file):
    """
    Parse a single ncu-rep file and extract key performance metrics for matmul kernels.
    Returns both matmul_kernel_m and matmul_kernel_n metrics, plus actual DDR read/write data.
    
    Returns:
        dict: Contains m_cycles, n_cycles, ddr_read_bytes_actual, ddr_write_bytes_actual
    """
    try:
        # Run ncu command to export the report data as CSV (raw page)
        result = subprocess.run(
            ['ncu', '--import', str(ncu_rep_file), '--csv', '--page', 'raw'],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.returncode != 0:
            print(f"Error: ncu command failed")
            return None
        
        # Parse CSV using csv.DictReader
        reader = csv.DictReader(io.StringIO(result.stdout))
        metrics = {
            'm_cycles': 0,
            'n_cycles': 0,
            'm_ddr_read_bytes_actual': 0,
            'm_ddr_write_bytes_actual': 0,
            'n_ddr_read_bytes_actual': 0,
            'n_ddr_write_bytes_actual': 0
        }
        
        for row in reader:
            try:
                kernel_name = row.get('Kernel Name', '').strip()
                
                # Only process matmul kernel rows
                if 'matmul_kernel' not in kernel_name.lower():
                    continue
                
                kernel_role = detect_kernel_role(kernel_name)
                if kernel_role is None:
                    continue
                
                # Extract Elapsed Cycles metric
                cycles = lookup_metric_value(row, [
                    'sm__cycles_elapsed.sum',
                    'sm__cycles_elapsed.avg',
                    'gpc__cycles_elapsed.max',
                ])
                if cycles:
                    if kernel_role == 'm':
                        metrics['m_cycles'] = cycles
                    else:
                        metrics['n_cycles'] = cycles
                
                # Extract actual DDR read bytes: lts__d_sectors_fill_sysmem.sum * 32
                read_bytes = lookup_metric_value(row, [
                    'dram__bytes_read.sum',
                    'dram__bytes.sum.per_second',
                ])
                if read_bytes is None:
                    read_sectors = lookup_metric_value(row, [
                        'lts__d_sectors_fill_sysmem.sum',
                        'dram__sectors_read.sum',
                    ])
                    if read_sectors is not None:
                        read_bytes = read_sectors * 32

                if read_bytes is not None:
                    if kernel_role == 'm':
                        metrics['m_ddr_read_bytes_actual'] = read_bytes
                    else:
                        metrics['n_ddr_read_bytes_actual'] = read_bytes
                
                # Extract actual DDR write bytes: lts__t_sectors_aperture_sysmem_op_write.sum * 32
                write_bytes = lookup_metric_value(row, [
                    'dram__bytes_write.sum',
                ])
                if write_bytes is None:
                    write_sectors = lookup_metric_value(row, [
                        'lts__t_sectors_aperture_sysmem_op_write.sum',
                        'dram__sectors_write.sum',
                    ])
                    if write_sectors is not None:
                        write_bytes = write_sectors * 32

                if write_bytes is not None:
                    if kernel_role == 'm':
                        metrics['m_ddr_write_bytes_actual'] = write_bytes
                    else:
                        metrics['n_ddr_write_bytes_actual'] = write_bytes
                
            except Exception as e:
                pass
        
        return metrics if any([metrics['m_cycles'], metrics['n_cycles']]) else None
    
    except subprocess.TimeoutExpired:
        print(f"Timeout reading {ncu_rep_file}")
        return None
    except Exception as e:
        print(f"Error parsing {ncu_rep_file}: {e}")
        return None


def extract_dimensions(file_path, tile_key=None):
    """Extract tile-m, tile-n, tile-k, M, N, K from filename/path."""
    path = Path(file_path)
    search_texts = [path.name, str(path)]
    if tile_key:
        search_texts.append(tile_key)

    dims = {
        'tile_m': None,
        'tile_n': None,
        'tile_k': None,
        'M': None,
        'N': None,
        'K': None
    }

    for text in search_texts:
        dims.update({k: v for k, v in extract_tile_dims_from_text(text).items() if v is not None})
        dims.update({k: v for k, v in extract_problem_dims_from_text(text).items() if v is not None})

    if all(dims[key] is not None for key in ['M', 'N', 'K']):
        return dims
    return None


def process_experiment_dir(exp_dir, output_dir, tile_key):
    """Process a single experiment directory (e.g., tm_256_tn_256_tk_64/K_768)."""
    exp_name = build_experiment_name(tile_key, exp_dir.name)
    exp_output_dir = get_experiment_output_dir(output_dir, tile_key, exp_dir.name)
    exp_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*80}")
    print(f"Processing experiment: {exp_name}")
    print(f"Parsed output directory: {exp_output_dir}")
    print(f"{'='*80}\n")
    
    # Collect all ncu-rep files in this experiment directory.
    # Use recursive scan so a changed per-K layout can still be parsed.
    ncu_files = sorted(exp_dir.rglob("*.ncu-rep"))
    
    if not ncu_files:
        print(f"No .ncu-rep files found in {exp_dir}")
        return None
    
    print(f"Found {len(ncu_files)} ncu-rep files\n")
    
    results = []
    
    for i, ncu_file in enumerate(ncu_files, 1):
        print(f"[{i}/{len(ncu_files)}] Processing: {ncu_file.name}")
        
        # Extract dimensions
        dims = extract_dimensions(ncu_file, tile_key=tile_key)
        if not dims:
            print(f"  Failed to extract dimensions from filename")
            continue
        
        # Parse the ncu report
        metrics = parse_ncu_report(str(ncu_file))
        if not metrics:
            print(f"  No matmul kernel metrics found")
            continue
        
        # Calculate theoretical DDR traffic (assuming float16 = 2 bytes per element)
        # Matrix multiply: C = A @ B
        # A: M x K, B: K x N
        # Total read = M*K + K*N (all in float16, 2 bytes each)
        # No-writeback: each block writes 1 fp32 checksum instead of a full C tile
        m, n, k = dims['M'], dims['N'], dims['K']
        tile_m = dims.get('tile_m') or 256
        tile_n = dims.get('tile_n') or 256
        bytes_per_element = 2  # float16
        
        metrics['ddr_read_bytes_theoretical'] = (m * k + k * n) * bytes_per_element
        metrics['ddr_write_bytes_theoretical'] = ceil(m / tile_m) * ceil(n / tile_n) * 4
        
        # Combine results
        record = {**dims, **metrics}
        record['filename'] = str(ncu_file.relative_to(exp_dir))
        record['experiment'] = exp_name
        results.append(record)
        
        print(f"  tm={record['tile_m']}, tn={record['tile_n']}, tk={record['tile_k']}, M={dims['M']}, N={dims['N']}, K={dims['K']}")
        if metrics.get('m_cycles'):
            print(f"    matmul_kernel_m Cycles: {metrics['m_cycles']:,.0f}")
        if metrics.get('n_cycles'):
            print(f"    matmul_kernel_n Cycles: {metrics['n_cycles']:,.0f}")
        print(f"    DDR Read  (theoretical): {metrics['ddr_read_bytes_theoretical']:,.0f} bytes ({metrics['ddr_read_bytes_theoretical']/1e6:.2f} MB)")
        print(f"    DDR Read  (m_actual):     {metrics['m_ddr_read_bytes_actual']:,.0f} bytes ({metrics['m_ddr_read_bytes_actual']/1e6:.2f} MB)")
        print(f"    DDR Read  (n_actual):     {metrics['n_ddr_read_bytes_actual']:,.0f} bytes ({metrics['n_ddr_read_bytes_actual']/1e6:.2f} MB)")
        print(f"    DDR Write (theoretical): {metrics['ddr_write_bytes_theoretical']:,.0f} bytes ({metrics['ddr_write_bytes_theoretical']/1e6:.2f} MB)")
        print(f"    DDR Write (m_actual):     {metrics['m_ddr_write_bytes_actual']:,.0f} bytes ({metrics['m_ddr_write_bytes_actual']/1e6:.2f} MB)")
        print(f"    DDR Write (n_actual):     {metrics['n_ddr_write_bytes_actual']:,.0f} bytes ({metrics['n_ddr_write_bytes_actual']/1e6:.2f} MB)")
        print()
    
    if not results:
        print("No results to save for this experiment")
        return None
    
    # Sort results by problem size and file path for stable output.
    results.sort(key=lambda x: (x['K'], x['M'], x['N'], x['filename']))
    
    # Save results to CSV for this experiment.
    # Mirror the ncu_results directory layout under parsed_results.
    output_csv = exp_output_dir / "ncu_results.csv"
    write_results_csv(output_csv, results)
    
    print(f"\nResults saved to: {output_csv}")
    
    # Print summary table
    print(f"\nSummary for {exp_name} (Theoretical vs Actual DDR Data):")
    print(f"{'tm':<6} {'tn':<6} {'tk':<6} {'M':<5} {'N':<5} {'M-Cy':<10} {'N-Cy':<10} {'R-Theo':<10} {'R-M-Act':<10} {'R-N-Act':<10} {'W-Theo':<10} {'W-M-Act':<10} {'W-N-Act':<10}")
    print("-" * 124)
    for record in results:
        tile_m = record.get('tile_m', '')
        tile_n = record.get('tile_n', '')
        tile_k = record.get('tile_k', '')
        m = record['M']
        n = record['N']
        m_cycles = record.get('m_cycles', 0) or 0
        n_cycles = record.get('n_cycles', 0) or 0
        read_theo_mb = (record.get('ddr_read_bytes_theoretical', 0) or 0) / 1e6
        read_m_act_mb = (record.get('m_ddr_read_bytes_actual', 0) or 0) / 1e6
        read_n_act_mb = (record.get('n_ddr_read_bytes_actual', 0) or 0) / 1e6
        write_theo_mb = (record.get('ddr_write_bytes_theoretical', 0) or 0) / 1e6
        write_m_act_mb = (record.get('m_ddr_write_bytes_actual', 0) or 0) / 1e6
        write_n_act_mb = (record.get('n_ddr_write_bytes_actual', 0) or 0) / 1e6
        print(f"{tile_m:<6} {tile_n:<6} {tile_k:<6} {m:<5} {n:<5} {m_cycles:<10,.0f} {n_cycles:<10,.0f} {read_theo_mb:<10.2f} {read_m_act_mb:<10.2f} {read_n_act_mb:<10.2f} {write_theo_mb:<10.2f} {write_m_act_mb:<10.2f} {write_n_act_mb:<10.2f}")
    
    # Save as JSON too
    output_json = exp_output_dir / "ncu_results.json"
    with open(output_json, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Also saved to: {output_json}")
    
    return results


def main():
    script_dir = Path(__file__).parent
    ncu_results_dir = script_dir / "ncu_results"
    
    if not ncu_results_dir.exists():
        print(f"Error: {ncu_results_dir} not found")
        return
    
    # Create output directory for parsed results
    output_dir = script_dir / "parsed_results"
    output_dir.mkdir(exist_ok=True)
    log_path, log_file, original_stdout, original_stderr = setup_logging(output_dir)
    print(f"Output directory: {output_dir}\n")
    print(f"Log file: {log_path}\n")
    
    # Find all tile-setting subdirectories and their K_xxx subdirectories.
    def extract_k_value(path):
        try:
            return int(path.name.split('_')[1])
        except:
            return 0

    tile_dirs = sorted(
        [d for d in ncu_results_dir.iterdir() if d.is_dir() and (d.name.startswith('tm_') or d.name.startswith('tk_'))]
    )

    if tile_dirs:
        exp_items = []
        for tile_dir in tile_dirs:
            k_dirs = sorted(
                [d for d in tile_dir.iterdir() if d.is_dir() and d.name.startswith('K_')],
                key=extract_k_value
            )
            for k_dir in k_dirs:
                exp_items.append((tile_dir.name, k_dir))
    else:
        # Backward compatibility with the old flat K_xxx layout.
        exp_items = [(None, d) for d in sorted(
            [d for d in ncu_results_dir.iterdir() if d.is_dir() and d.name.startswith('K_')],
            key=extract_k_value
        )]

    if not exp_items:
        print(f"No experiment directories found in {ncu_results_dir}")
        return

    if tile_dirs:
        tile_keys = sorted({tile_key for tile_key, _ in exp_items if tile_key is not None})
        k_values = sorted({extract_k_value(d) for _, d in exp_items})
        print(f"Found tile settings: {tile_keys}")
        print(f"Found {len(exp_items)} experiment directories with K values: {k_values}\n")
    else:
        k_values = [extract_k_value(d) for _, d in exp_items]
        print(f"Found {len(exp_items)} experiment directories with K values: {k_values}\n")
    
    all_results = []
    
    # Process each experiment directory
    for tile_key, exp_dir in exp_items:
        results = process_experiment_dir(exp_dir, output_dir, tile_key)
        if results:
            all_results.extend(results)
    
    if not all_results:
        print("\nNo results collected from any experiment")
        return
    
    # Save combined results to CSV
    print(f"\n{'='*80}")
    print("Saving combined results...")
    print(f"{'='*80}\n")
    
    output_csv = output_dir / "ncu_results_combined.csv"
    write_results_csv(output_csv, all_results)
    
    print(f"Combined results saved to: {output_csv}")
    
    # Save combined results as JSON
    output_json = output_dir / "ncu_results_combined.json"
    with open(output_json, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"Combined results also saved to: {output_json}")
    print(f"\nTotal: Processed {len(all_results)} results from {len(exp_items)} experiments")
    teardown_logging(log_file, original_stdout, original_stderr)


if __name__ == "__main__":
    main()
