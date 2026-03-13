#!/usr/bin/env python3

import re
import subprocess
import json
import csv
import io
from pathlib import Path

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
                
                # Only process no-writeback matmul kernel rows
                if 'matmul_kernel' not in kernel_name.lower():
                    continue
                
                # Distinguish between kernel_m and kernel_n variants
                is_kernel_m = 'matmul_kernel_m' in kernel_name.lower()
                is_kernel_n = 'matmul_kernel_n' in kernel_name.lower()
                
                # Extract Elapsed Cycles metric
                elapsed_cycles_str = row.get('sm__cycles_elapsed.sum', '').strip()
                if elapsed_cycles_str and elapsed_cycles_str != '0':
                    try:
                        cycles = float(elapsed_cycles_str)
                        if is_kernel_m:
                            metrics['m_cycles'] = cycles
                        elif is_kernel_n:
                            metrics['n_cycles'] = cycles
                    except:
                        pass
                
                # Extract actual DDR read bytes: lts__d_sectors_fill_sysmem.sum * 32
                read_sectors_str = row.get('lts__d_sectors_fill_sysmem.sum', '').strip()
                if read_sectors_str:
                    try:
                        read_sectors = float(read_sectors_str)
                        if is_kernel_m:
                            metrics['m_ddr_read_bytes_actual'] = read_sectors * 32
                        elif is_kernel_n:
                            metrics['n_ddr_read_bytes_actual'] = read_sectors * 32
                    except:
                        pass
                
                # Extract actual DDR write bytes: lts__t_sectors_aperture_sysmem_op_write.sum * 32
                write_sectors_str = row.get('lts__t_sectors_aperture_sysmem_op_write.sum', '').strip()
                if write_sectors_str:
                    try:
                        write_sectors = float(write_sectors_str)
                        if is_kernel_m:
                            metrics['m_ddr_write_bytes_actual'] = write_sectors * 32
                        elif is_kernel_n:
                            metrics['n_ddr_write_bytes_actual'] = write_sectors * 32
                    except:
                        pass
                
            except Exception as e:
                pass
        
        return metrics if any([metrics['m_cycles'], metrics['n_cycles']]) else None
    
    except subprocess.TimeoutExpired:
        print(f"Timeout reading {ncu_rep_file}")
        return None
    except Exception as e:
        print(f"Error parsing {ncu_rep_file}: {e}")
        return None


def extract_dimensions(filename):
    """Extract M, N, K dimensions from filename."""
    # Format: MatMulPerf_M256_N256_K768.ncu-rep
    match = re.search(r'M(\d+)_N(\d+)_K(\d+)', filename)
    if match:
        return {
            'M': int(match.group(1)),
            'N': int(match.group(2)),
            'K': int(match.group(3))
        }
    return None


def process_experiment_dir(exp_dir, output_dir):
    """Process a single experiment directory (e.g., K_768)"""
    exp_name = exp_dir.name
    print(f"\n{'='*80}")
    print(f"Processing experiment: {exp_name}")
    print(f"{'='*80}\n")
    
    # Collect all ncu-rep files in this experiment directory
    ncu_files = sorted(exp_dir.glob("*.ncu-rep"))
    
    if not ncu_files:
        print(f"No .ncu-rep files found in {exp_dir}")
        return None
    
    print(f"Found {len(ncu_files)} ncu-rep files\n")
    
    results = []
    
    for i, ncu_file in enumerate(ncu_files, 1):
        print(f"[{i}/{len(ncu_files)}] Processing: {ncu_file.name}")
        
        # Extract dimensions
        dims = extract_dimensions(ncu_file.name)
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
        # No-writeback: C is NOT written back; only 1 float32 checksum per block
        m, n, k = dims['M'], dims['N'], dims['K']
        bytes_per_element = 2  # float16
        
        metrics['ddr_read_bytes_theoretical'] = (m * k + k * n) * bytes_per_element
        # Theoretical write is negligible: 1 float32 per block
        # For tm=tn=256: num_blocks = ceil(m/256)*ceil(n/256), each writes 4 bytes
        from math import ceil
        tm, tn = 256, 256
        num_blocks = ceil(m / tm) * ceil(n / tn)
        metrics['ddr_write_bytes_theoretical'] = num_blocks * 4  # 4 bytes per float32 checksum
        
        # Combine results
        record = {**dims, **metrics}
        record['filename'] = ncu_file.name
        record['experiment'] = exp_name
        results.append(record)
        
        print(f"  M={dims['M']}, N={dims['N']}, K={dims['K']}")
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
    
    # Sort results by M dimension
    results.sort(key=lambda x: x['M'])
    
    # Save results to CSV for this experiment
    output_csv = output_dir / f"ncu_results_{exp_name}.csv"
    with open(output_csv, 'w', newline='') as f:
        fieldnames = ['M', 'N', 'K', 'm_cycles', 'n_cycles', 'ddr_read_bytes_theoretical', 'ddr_write_bytes_theoretical', 'm_ddr_read_bytes_actual', 'n_ddr_read_bytes_actual', 'm_ddr_write_bytes_actual', 'n_ddr_write_bytes_actual', 'filename', 'experiment']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in results:
            # Handle missing fields
            for field in fieldnames:
                if field not in record:
                    record[field] = ''
            writer.writerow(record)
    
    print(f"\nResults saved to: {output_csv}")
    
    # Print summary table
    print(f"\nSummary for {exp_name} (Theoretical vs Actual DDR Data):")
    print(f"{'M':<5} {'N':<5} {'M-Cy':<10} {'N-Cy':<10} {'R-Theo':<10} {'R-M-Act':<10} {'R-N-Act':<10} {'W-Theo':<10} {'W-M-Act':<10} {'W-N-Act':<10}")
    print("-" * 105)
    for record in results:
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
        print(f"{m:<5} {n:<5} {m_cycles:<10,.0f} {n_cycles:<10,.0f} {read_theo_mb:<10.2f} {read_m_act_mb:<10.2f} {read_n_act_mb:<10.2f} {write_theo_mb:<10.2f} {write_m_act_mb:<10.2f} {write_n_act_mb:<10.2f}")
    
    # Save as JSON too
    output_json = output_dir / f"ncu_results_{exp_name}.json"
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
    print(f"Output directory: {output_dir}\n")
    
    # Find all experiment subdirectories (K_xxx) and sort by K value
    exp_dirs = [d for d in ncu_results_dir.iterdir() if d.is_dir() and d.name.startswith('K_')]
    
    # Sort by K value numerically
    def extract_k_value(path):
        try:
            return int(path.name.split('_')[1])
        except:
            return 0
    
    exp_dirs = sorted(exp_dirs, key=extract_k_value)
    
    if not exp_dirs:
        print(f"No experiment directories found in {ncu_results_dir}")
        return
    
    k_values = [extract_k_value(d) for d in exp_dirs]
    print(f"Found {len(exp_dirs)} experiment directories with K values: {k_values}\n")
    
    all_results = []
    
    # Process each experiment directory
    for exp_dir in exp_dirs:
        results = process_experiment_dir(exp_dir, output_dir)
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
    with open(output_csv, 'w', newline='') as f:
        fieldnames = ['M', 'N', 'K', 'm_cycles', 'n_cycles', 'ddr_read_bytes_theoretical', 'ddr_write_bytes_theoretical', 'm_ddr_read_bytes_actual', 'n_ddr_read_bytes_actual', 'm_ddr_write_bytes_actual', 'n_ddr_write_bytes_actual', 'filename', 'experiment']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in all_results:
            # Handle missing fields
            for field in fieldnames:
                if field not in record:
                    record[field] = ''
            writer.writerow(record)
    
    print(f"Combined results saved to: {output_csv}")
    
    # Save combined results as JSON
    output_json = output_dir / "ncu_results_combined.json"
    with open(output_json, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"Combined results also saved to: {output_json}")
    print(f"\nTotal: Processed {len(all_results)} results from {len(exp_dirs)} experiments")


if __name__ == "__main__":
    main()
