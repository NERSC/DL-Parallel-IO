import sys
import csv

import numpy as np
import matplotlib.pyplot as plt

result_file_path = sys.argv[1]
number_of_nodes = sys.argv[2]

with open(result_file_path) as csv_file:
    data = []
    csv.register_dialect('TrimmerDialect', quotechar='"', skipinitialspace=True, quoting=csv.QUOTE_NONE,
                         lineterminator='\n', strict=True)
    dict_reader = csv.DictReader(csv_file, dialect='TrimmerDialect')

    total_read_time = np.array([0.0, 0.0, 0.0, 0.0]) # indices are epoch numbers
    total_training_iteration_time = np.array([0.0, 0.0, 0.0, 0.0])
    for row in dict_reader:
        epoch_number = 0 if int(row['Epoch']) == -1 else int(row['Epoch'])
        if row['Action Description'] == 'Parallel Read Images with 4 Threads':
            total_read_time[epoch_number] += float(row['Time Taken'])
        if row['Action Description'] == 'Training Iteration':
            total_training_iteration_time[epoch_number] += float(row['Time Taken'])

    data = [total_read_time/int(number_of_nodes), total_training_iteration_time/int(number_of_nodes)]
    columns = ('Before Training', 'Epoch 1', 'Epoch 2', 'Epoch 3')
    rows = ['Read', 'Training Iteration']

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
                                                                    total_read_time[column]) /
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

    plt.show()
