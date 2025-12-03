import csv
import os
import re
import sys

def read_csv(filepath):
    """Reads a CSV file and returns its content as a list of lists of integers."""
    if not os.path.exists(filepath):
        return None
    with open(filepath, 'r', newline='') as f:
        reader = csv.reader(f)
        try:
            return [[int(cell) for cell in row] for row in reader if row]
        except ValueError as e:
            print(f"Error reading {filepath}: Could not convert data to integer. {e}", file=sys.stderr)
            return None

def compare_csv_data(data1, data2):
    """Compares two lists of lists of integers."""
    if len(data1) != len(data2):
        return False
    for i in range(len(data1)):
        if len(data1[i]) != len(data2[i]):
            return False
        if data1[i] != data2[i]:
            return False
    return True

def find_matching_answer_file(generated_filename, ans_dir):
    """
    Finds a matching reference file in the answer directory based on rotation params.
    Example: For 'run_0001_clean_waf15_r45_b98.csv', it looks for a file
    in ans_dir that contains '_r45_b98.csv'.
    """
    # Extract the parameter part of the filename, e.g., "_r45_b98"
    match = re.search(r'_r\d+_b\d+', generated_filename)
    if not match:
        return None
    
    param_part = match.group(0) # e.g., _r45_b98
    
    # Search for a file in the answer directory with this parameter part
    for ans_filename in os.listdir(ans_dir):
        if ans_filename.endswith(f"{param_part}.csv"):
            return os.path.join(ans_dir, ans_filename)
            
    return None

def main():
    """Main validation function."""
    output_dir = 'outputs'
    ans_dir = 'ans'

    if not os.path.isdir(output_dir):
        print(f"Error: Output directory '{output_dir}' not found.", file=sys.stderr)
        print("Please run the C++ program first to generate output files.", file=sys.stderr)
        sys.exit(1)
        
    if not os.path.isdir(ans_dir):
        print(f"Error: Answer directory '{ans_dir}' not found.", file=sys.stderr)
        sys.exit(1)

    generated_files = [f for f in os.listdir(output_dir) if f.endswith('.csv')]

    if not generated_files:
        print(f"No CSV files found in '{output_dir}'.")
        sys.exit(0)

    success_count = 0
    fail_count = 0
    missing_ans_count = 0

    print(f"Starting validation for {len(generated_files)} generated files...")

    for gen_filename in sorted(generated_files):
        gen_filepath = os.path.join(output_dir, gen_filename)
        
        print(f"\n--- Validating {gen_filename} ---")

        # Find the corresponding answer file
        ans_filepath = find_matching_answer_file(gen_filename, ans_dir)

        if not ans_filepath:
            print(f"Result: ❗️ SKIPPED - No corresponding answer file found in '{ans_dir}'.")
            missing_ans_count += 1
            continue

        print(f"Found matching answer file: {os.path.basename(ans_filepath)}")

        # Read data from both files
        generated_data = read_csv(gen_filepath)
        answer_data = read_csv(ans_filepath)

        if generated_data is None or answer_data is None:
            print(f"Result: ❌ FAILED - Could not read or parse one of the files.")
            fail_count += 1
            continue
        
        # Compare the data
        if compare_csv_data(generated_data, answer_data):
            print("Result: ✅ PASSED - Files match perfectly.")
            success_count += 1
        else:
            print("Result: ❌ FAILED - File contents do not match.")
            fail_count += 1

    # Final summary
    print("\n================== Validation Summary ==================")
    print(f"Total files processed: {len(generated_files)}")
    print(f"✅ Successful matches:  {success_count}")
    print(f"❌ Failed matches:      {fail_count}")
    print(f"❗️ Skipped (no answer): {missing_ans_count}")
    print("======================================================")

    if fail_count > 0:
        sys.exit(1)

if __name__ == '__main__':
    main()

