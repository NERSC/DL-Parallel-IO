import sys
import csv

import numpy as np
import matplotlib.pyplot as plt

from pprint import pprint

def merge_and_add_common_intervals(interval_list_per_epoch):
    merged_interval_list = []
    for interval_list_per_epoch_per_call in interval_list_per_epoch:
        if len(merged_interval_list) == 0:
            merged_interval_list.append(interval_list_per_epoch_per_call)
        else:
            start1 = merged_interval_list[len(merged_interval_list) - 1][0]
            end1 = merged_interval_list[len(merged_interval_list) - 1][1]
            start2 = interval_list_per_epoch_per_call[0]
            end2 = interval_list_per_epoch_per_call[1]
            if start2 > end1 or end2 < start1:
                merged_interval_list.append(interval_list_per_epoch_per_call)
            else:
                if(start2 < start1):
                    merged_interval_list[len(merged_interval_list)-1][0] = start2
                if(end2 > end1):
                    merged_interval_list[len(merged_interval_list)-1][1] = end2

    total_time = 0.0
    for merged_interval_element in merged_interval_list:
        total_time += (merged_interval_element[1] - merged_interval_element[0])
    return total_time

def calculate_time_from_interval(interval_list):
    total_read_time_from_intervals = np.array([0.0, 0.0, 0.0, 0.0])
    epoch = 0
    for interval_list_per_epoch in interval_list:
        total_read_time_from_intervals[epoch] += merge_and_add_common_intervals(interval_list_per_epoch)
        epoch = epoch + 1

    return total_read_time_from_intervals

number_of_criteria = int(sys.argv[1])  # e.g. If you want to plot 64, 128, 256; this value should be 3

result_file_path_list = []
number_of_nodes_list = []

for arg_indices in xrange(2, (number_of_criteria+1)*2, 2):
    result_file_path_list.append(sys.argv[arg_indices])
    number_of_nodes_list.append(sys.argv[arg_indices + 1])

rows = ['Read', 'Training Iteration']

data_scaling = [] * len(rows)

data_read_count_scaling = []

for index in xrange(0, len(rows)):
    data_scaling.append([])

for data_scaling_element in data_scaling:
    for count in xrange(number_of_criteria):
        data_scaling_element.append(0.0)

for index in xrange(0, len(result_file_path_list)):
    result_file_path = result_file_path_list[index]
    number_of_nodes = number_of_nodes_list[index]
    with open(result_file_path) as csv_file:
        data = []
        csv.register_dialect('TrimmerDialect', quotechar='"', skipinitialspace=True, quoting=csv.QUOTE_NONE,
                             lineterminator='\n', strict=True)
        dict_reader = csv.DictReader(csv_file, dialect='TrimmerDialect')

        total_read_time = np.array([0.0, 0.0, 0.0, 0.0])  # indices are epoch numbers
        total_training_iteration_time = np.array([0.0, 0.0, 0.0, 0.0])

        interval_list = []
        for i in xrange(0, 4):
            interval_list.append([])

        read_count = 0

        for row in dict_reader:
            epoch_number = 0 if int(row['Epoch']) == -1 else int(row['Epoch'])
            if row['Action Description'] == 'Time to Read Single Image':
                total_read_time[epoch_number] += float(row['Time Taken'])
                interval_list[epoch_number].append([float(row['Start Time']), float(row['End Time'])])
                read_count = read_count + 1
            if row['Action Description'] == 'Training Iteration':
                total_training_iteration_time[epoch_number] += float(row['Time Taken'])

        total_read_time_from_interval = calculate_time_from_interval(interval_list)

        data = [total_read_time_from_interval/int(number_of_nodes), total_training_iteration_time/int(number_of_nodes)]
        data_read_count_scaling.append(read_count)

        for data_index in xrange(0, len(data)):
            data_scaling[data_index][index] += sum(data[data_index])

        columns = ['Asynchronous Read', 'Epoch 1', 'Epoch 2', 'Epoch 3']

        # Get some pastel shades for the colors
        colors = plt.cm.BuPu(np.linspace(0, 0.5, len(rows)))
        n_rows = len(data)

        index = np.arange(len(columns)) + 0.3
        bar_width = 0.4

        # Initialize the vertical-offset for the stacked bar chart.
        y_offset = np.zeros(len(columns))

        # Plot bars and create text labels for the table
        cell_text = []
        for row in range(n_rows):
            plt.bar(index, data[row], bar_width, bottom=y_offset, color=colors[row])
            y_offset = y_offset + data[row]

        data_table = []
        for row in xrange(0, len(data)):
            data_table_row = []
            for column in xrange(0, len(data[0])):
                data_table_row.append(str("{0:.2f} ({1:.2f} %)").format(data[row][column],
                                                                        data[row][column] /
                                                                        ((total_training_iteration_time[column] +
                                                                          total_read_time_from_interval[column]) /
                                                                         int(number_of_nodes)) * 100))
            data_table.append(data_table_row)

        # Add a table at the bottom of the axes
        the_table = plt.table(cellText=data_table,
                              rowLabels=rows,
                              rowColours=colors,
                              colLabels=columns,
                              loc='bottom')

        # Adjust layout to make room for the table:
        plt.subplots_adjust(left=0.2, bottom=0.2)

        plt.ylabel("Average Time Taken Per Node (seconds)")
        plt.xticks([])
        plt.title("{} Nodes".format(number_of_nodes))

        #plt.show()
        plt.savefig(result_file_path.split('.')[0]+"_plot.png")
        plt.clf()

columns = []
for number_of_node in number_of_nodes_list:
    columns.append(number_of_node)

data_table = []
for row in xrange(0, len(data_scaling)):
    data_table_row = []
    for column in xrange(0, len(data_scaling[0])):
        total_column_criteria_time = 0.0
        for data_scaling_element in data_scaling:
            total_column_criteria_time += data_scaling_element[column]

        data_table_row.append(str("{0:.2f} ({1:.2f} %)").format(data_scaling[row][column],
                                                                data_scaling[row][column] /
                                                                total_column_criteria_time * 100))
    data_table.append(data_table_row)

# Get some pastel shades for the colors
colors = plt.cm.BuPu(np.linspace(0, 0.5, len(rows)))
n_rows = len(data_scaling)

index = np.arange(len(columns)) + 0.3
bar_width = 0.4

# Initialize the vertical-offset for the stacked bar chart.
y_offset = np.zeros(len(columns))

# Plot bars and create text labels for the table
cell_text = []
for row in range(n_rows):
    plt.bar(index, data_scaling[row], bar_width, bottom=y_offset, color=colors[row])
    y_offset = y_offset + data_scaling[row]

# Add a table at the bottom of the axes
the_table = plt.table(cellText=data_table,
                      rowLabels=rows,
                      rowColours=colors,
                      colLabels=columns,
                      loc='bottom')

# Adjust layout to make room for the table:
plt.subplots_adjust(left=0.2, bottom=0.2)

plt.ylabel("Average Time Taken Per Node (seconds)")
plt.xticks([])
plt.title("Scale Out 3 Epochs")

# plt.show()
plt.savefig(result_file_path.split('.')[0].split('_')[0] + "_scaling_plot.png")

plt.clf()

rows = ['Read Time', 'Ideal Read Time']

# Get some pastel shades for the colors
colors = plt.cm.BuPu(np.linspace(0, 0.5, len(rows)))

bar_width = 0.4

index = 0
ploting_index = 0.3

data_ideal_scaling = []
data_ideal_scaling.append(data_scaling[0][0])

for read_value in data_scaling[0]:
    if index > 0:
        data_ideal_scaling.append(data_ideal_scaling[index - 1] / (int(number_of_nodes_list[index])/int(number_of_nodes_list[index-1])))

    plt.bar(ploting_index, read_value, bar_width, color=colors[0])
    index = index + 1
    ploting_index = ploting_index + 1

xdata = np.arange(len(columns)) + 0.5

plt.plot(data_ideal_scaling, color=colors[1], linestyle='-', marker='o', xdata=xdata)

data_table = []

data_table_row = []
for column in xrange(0, len(data_scaling[0])):
    total_column_criteria_time = 0.0
    for data_scaling_element in data_scaling:
        total_column_criteria_time += data_scaling_element[column]

    data_table_row.append(str("{0:.2f} ({1:.2f}X)").format(data_scaling[0][column],
                                                            data_ideal_scaling[column] /
                                                            data_scaling[0][column]))
data_table.append(data_table_row)

data_table.append(data_ideal_scaling)

# Add a table at the bottom of the axes
the_table = plt.table(cellText=data_table,
                      rowLabels=rows,
                      rowColours=colors,
                      colLabels=columns,
                      loc='bottom')

# Adjust layout to make room for the table:
plt.subplots_adjust(left=0.2, bottom=0.2)

plt.ylabel("Average Time Taken Per Node (seconds)")
plt.xticks([])
plt.title("Scale Out 3 Epochs Read Time")

# plt.show()
plt.savefig(result_file_path.split('.')[0].split('_')[0] + "_read_scaling_plot.png")

plt.clf()

# Plot Read Count

rows = ['Read Count', 'Read Time', 'Read Bandwidth', 'Ideal Case']

colors = plt.cm.BuPu(np.linspace(0, 0.5, len(rows)))

bar_width = 0.4

index = 0
ploting_index = 0.3

# Keep it for bandwidth plotting
data_ideal_scaling = []
data_ideal_scaling.append(data_read_count_scaling[0])

for read_value in data_read_count_scaling:
    if index > 0:
        data_ideal_scaling.append(data_ideal_scaling[index - 1] * 2)

    plt.bar(ploting_index, read_value, bar_width, color=colors[0])
    index = index + 1
    ploting_index = ploting_index + 1

data_table = []

data_table_row = []
for column in xrange(0, len(data_read_count_scaling)):
    data_table_row.append(str("{0:.2f}").format(data_read_count_scaling[column]))
data_table.append(data_table_row)

data_table.append(data_scaling[0])

data_table.append(data_scaling[0])

data_table.append(data_ideal_scaling)

# Add a table at the bottom of the axes
the_table = plt.table(cellText=data_table,
                      rowLabels=rows,
                      rowColours=colors,
                      colLabels=columns,
                      loc='bottom')

# Adjust layout to make room for the table:
plt.subplots_adjust(left=0.2, bottom=0.2)

plt.ylabel("Bandwidth (MB/s)")
plt.xticks([])
plt.title("Scale Out 3 Epochs Read Bandwidth")

# plt.show()
plt.savefig(result_file_path.split('.')[0].split('_')[0] + "_read_count_scaling_plot.png")

plt.clf()

# Plot Read Count

rows = ['Read Count', 'Read Time', 'Read Bandwidth', 'Ideal Case']

colors = plt.cm.BuPu(np.linspace(0, 0.5, len(rows)))

bar_width = 0.4

index = 0
ploting_index = 0.3

# Keep it for bandwidth plotting
data_ideal_read_bandwidth_scaling = []

size_of_training_file = 17694752

data_read_bandwidth_scaling = []

for read_value in data_read_count_scaling:
    bandwidth = ((read_value * size_of_training_file) / data_scaling[0][index]) / 1000000000

    if index == 0:
        data_ideal_read_bandwidth_scaling.append(bandwidth)
    if index > 0:
        data_ideal_read_bandwidth_scaling.append(data_ideal_read_bandwidth_scaling[index - 1] * (int(number_of_nodes_list[index])/int(number_of_nodes_list[index-1])))

    data_read_bandwidth_scaling.append("{0:.3f} ({1:.2f}X)".format(bandwidth, bandwidth /
                                                                data_ideal_read_bandwidth_scaling[index]))

    plt.bar(ploting_index, bandwidth, bar_width, color=colors[0])
    index = index + 1
    ploting_index = ploting_index + 1

xdata = np.arange(len(columns)) + 0.5

plt.plot(data_ideal_read_bandwidth_scaling, color=colors[3], linestyle='-', marker='o', xdata=xdata)

data_table = []

data_table_row = []
for column in xrange(0, len(data_read_count_scaling)):
    data_table_row.append(str("{}").format(data_read_count_scaling[column]))

data_table.append(data_table_row)
data_table.append(data_scaling[0])
data_table.append(data_read_bandwidth_scaling)
data_table.append(data_ideal_read_bandwidth_scaling)

# Add a table at the bottom of the axes
the_table = plt.table(cellText=data_table,
                      rowLabels=rows,
                      colLabels=columns,
                      rowColours=[colors[0], colors[0], colors[0], colors[3]],
                      loc='bottom')

# Adjust layout to make room for the table:
plt.subplots_adjust(left=0.2, bottom=0.2)

plt.ylabel("Bandwidth (GB/s)")
plt.xticks([])
plt.title("Scale Out 3 Epochs Read Bandwidth")

# plt.show()
plt.savefig(result_file_path.split('.')[0].split('_')[0] + "_read_bandwidth_scaling_plot.png")
