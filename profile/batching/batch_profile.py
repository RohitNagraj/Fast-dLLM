import subprocess
import json
import os

# Get the path to generate.py (assuming it's in the parent directory)
generate_script = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'llada', 'generate.py')
out_file = 'out.txt'

for i in range(1, 17):
    # Run the subprocess and redirect output to out.txt
    with open(out_file, 'w') as f:
        subprocess.run(['python', generate_script, f'--repeat={i}', f'--block={16}'], stdout=f, stderr=subprocess.STDOUT)

    # Read out.txt and drop the first 100 lines
    with open(out_file, 'r') as f:
        lines = f.readlines()

    # Drop the first 100 lines
    json_lines = lines[100:]

    # Parse each line as JSON and collect all values
    all_values = {}
    count = 0

    for line in json_lines:
        line = line.strip()
        if line:  # Skip empty lines
            try:
                data = json.loads(line)
                count += 1
                # Add values to running totals
                for key, value in data.items():
                    if key not in all_values:
                        all_values[key] = 0
                    all_values[key] += value
            except json.JSONDecodeError:
                continue

    # Calculate averages
    if count > 0:
        avg_values = {key: total / count for key, total in all_values.items()}
    else:
        avg_values = {}

    # Print as JSONL (one line JSON)
    print(json.dumps(avg_values))
