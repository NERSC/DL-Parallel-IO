# Extracts the output of TimeLogger from the final output from running benchmark scripts on SLURM and writes the extracted information to a file
# Also creates a csv file with the necessary information
# Usage: python write_time_logger_messages.py <file_path_to_the_output_from_slurm> <file_path_to_the_output_from_this_script>

import sys
import re

input_file_path = sys.argv[1]
output_file_path = sys.argv[2]
time_logger_output_regex = re.compile(r'\'TIME LOGGER OUTPUT.*For Excel:[^\']+\'')
for_excel_regex = re.compile(r'For Excel:[^\']+\'')

input_file = open(input_file_path, "r")
output_file = open(output_file_path, "w")
output_file_csv = open(output_file_path+".csv", "w")
print >> output_file_csv, "Epoch, Rank, Action Name, Action Description, Start Time, End Time, Time Taken"

for line in input_file:
    time_logger_outputs = time_logger_output_regex.findall(line)
    for time_logger_output in time_logger_outputs:
        print >> output_file, time_logger_output
        for_excel_outputs = for_excel_regex.findall(time_logger_output)
        for for_excel_output in for_excel_outputs:
            print >> output_file_csv, for_excel_output.split(':')[1].strip().replace('\'', '')
