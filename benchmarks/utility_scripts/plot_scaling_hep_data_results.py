import sys
import csv

import numpy as np
import matplotlib.pyplot as plt

from pprint import pprint

number_of_criteria = int(sys.argv[1])  # e.g. If you want to plot 64, 128, 256; this value should be 3

result_file_path_list = []
number_of_nodes_list = []

for arg_indices in xrange(2, (number_of_criteria+1)*2, 2):
    result_file_path_list.append(sys.argv[arg_indices])
    number_of_nodes_list.append(sys.argv[arg_indices + 1])

rows = ['Read', 'Load File', 'Training Iteration']

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

        total_read_time = np.array([0.0, 0.0, 0.0, 0.0, 0.0])  # indices are epoch numbers
        total_load_file_time = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        total_training_iteration_time = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

        read_count = 0
        for row in dict_reader:
            epoch_number = 0 if int(row['Epoch']) == -1 else int(row['Epoch'])
            if row['Action Description'] == 'HDF5 File Read':
                total_read_time[epoch_number] += float(row['Time Taken'])
                read_count = read_count + 1
            if row['Action Description'] == 'Load File':
                total_load_file_time[epoch_number] += float(row['Time Taken'])
            if row['Action Description'] == 'Training Iteration':
                total_training_iteration_time[epoch_number] += float(row['Time Taken'])

        data = [total_read_time/int(number_of_nodes), (total_load_file_time - total_read_time)/int(number_of_nodes),
                (total_training_iteration_time - total_load_file_time)/int(number_of_nodes)]
        data_read_count_scaling.append(read_count)

        for data_index in xrange(0, len(data)):
            data_scaling[data_index][index] += sum(data[data_index])

        columns = ['Epoch 1', 'Epoch 2', 'Epoch 3', 'Epoch 4', 'Epoch 5']

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
                                                                        (total_training_iteration_time[column] /
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
        plt.savefig(result_file_path.split('.csv')[0]+"_plot.png")
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
plt.title("Scale Out 5 Epochs")

# plt.show()
plt.savefig(result_file_path.split("_" + sys.argv[len(sys.argv)-1])[0] + "_scaling_plot.png")

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
plt.title("Scale Out 5 Epochs Read Time")

# plt.show()
plt.savefig(result_file_path.split("_" + sys.argv[len(sys.argv)-1])[0] + "_read_scaling_plot.png")

plt.clf()

# Plot Read Count

rows = ['Read Count', 'Read Time', 'Read Bandwidth', 'Ideal Case']

colors = plt.cm.BuPu(np.linspace(0, 0.5, len(rows)))

bar_width = 0.4

index = 0
ploting_index = 0.3

# Keep it for bandwidth plotting
data_ideal_read_bandwidth_scaling = []

number_of_training_file = 1024
size_of_training_file = 426897408 + 2836 * 4
size_of_validation_file = 56598528 + 376 * 4

data_read_bandwidth_scaling = []

for read_value in data_read_count_scaling:
    number_of_validation_file = 0
    if read_value > number_of_training_file:
        number_of_validation_file = read_value - number_of_training_file
    elif read_value == 0:
        number_of_training_file = 0

    bandwidth = ((number_of_training_file * size_of_training_file + number_of_validation_file *
                 size_of_validation_file) / data_scaling[0][index]) / 1000000000

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
plt.savefig(result_file_path.split("_" + sys.argv[len(sys.argv)-1])[0] + "_read_bandwidth_scaling_plot.png")
