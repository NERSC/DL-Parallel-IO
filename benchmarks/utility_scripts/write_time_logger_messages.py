import sys
import re

input_file_path = sys.argv[1]
output_file_path = sys.argv[2]
time_logger_output_regex = re.compile(r'\'TIME LOGGER OUTPUT.*Time Taken:.*\'')

input_file = open(input_file_path, "r")
output_file = open(output_file_path, "w")

for line in input_file:
    time_logger_outputs = time_logger_output_regex.findall(line)
    for time_logger_output in time_logger_outputs:
        print >> output_file,time_logger_output

