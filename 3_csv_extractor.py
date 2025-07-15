import re
from datetime import datetime

input_path = 'rad.txt.rtf'
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
output_path = f'radio_data_extracted_{timestamp}.csv'

with open(input_path, 'r') as infile, open(output_path, 'w') as outfile:
    for line in infile:
        # Match lines that start with a number and comma (CSV data)
        if re.match(r'^\d+,', line.strip()):
            outfile.write(line)

print(f"Extracted CSV data to {output_path}") 