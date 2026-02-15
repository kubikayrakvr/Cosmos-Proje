import json

# Input and output file names
input_file = 'gen_A_data.jsonl'
output_file = 'gen_A_data_reformatted.jsonl'

def process_file(input_path, output_path):
    count = 0
    
    try:
        with open(input_path, 'r', encoding='utf-8') as infile, \
             open(output_path, 'w', encoding='utf-8') as outfile:
            
            for line in infile:
                if not line.strip():
                    continue
                
                try:
                    data = json.loads(line)
                    
                    # Reformat the answer field if it exists
                    if 'answer' in data:
                        # Replace only the specific separator
                        data['answer'] = data['answer'].replace('\n\n###', '\n####')
                    
                    # Write the modified data to the new file
                    outfile.write(json.dumps(data, ensure_ascii=False) + '\n')
                    count += 1
                    
                except json.JSONDecodeError:
                    print(f"Skipping invalid JSON line: {line[:50]}...")
                    continue

        print(f"Processing complete.")
        print(f"Total records processed: {count}")
        print(f"Reformatted data saved to: {output_path}")

    except FileNotFoundError:
        print(f"Error: The file '{input_path}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    process_file(input_file, output_file)