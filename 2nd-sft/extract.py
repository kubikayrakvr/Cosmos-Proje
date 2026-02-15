import json

# Define input and output file names
input_file = 'ultimate_sft_shuffled.jsonl'
output_file = 'gen_A_data.jsonl'

def extract_gen_a_data(input_path, output_path):
    count = 0
    try:
        with open(input_path, 'r', encoding='utf-8') as infile, \
             open(output_path, 'w', encoding='utf-8') as outfile:
            
            for line in infile:
                if not line.strip():
                    continue
                
                try:
                    data = json.loads(line)
                    # Check if the 'id' field starts with 'gen_A'
                    if data.get('id', '').startswith('gen_A'):
                        outfile.write(json.dumps(data, ensure_ascii=False) + '\n')
                        count += 1
                except json.JSONDecodeError:
                    print(f"Skipping invalid JSON line: {line[:50]}...")
                    continue

        print(f"Extraction complete. Found {count} records tagged with 'gen_A'.")
        print(f"Data saved to: {output_path}")

    except FileNotFoundError:
        print(f"Error: The file '{input_path}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    extract_gen_a_data(input_file, output_file)