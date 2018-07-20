from operator import itemgetter
import matplotlib.pyplot as plt
import sys
import csv
import numpy as np
from pprint import pprint

def merge_and_add_common_intervals(data):
    begin = get_elements(1, data)
    end = get_elements(2, data)
    interval_list = [begin, end]
    merged_interval_list = []
    sorted_interval_list = sorted(interval_list, key=lambda interval: interval[0])
    # check_if_sorted(interval_list_per_epoch)
    # check_if_sorted(sorted_interval_list)
    for interval_list_per_epoch_per_call in sorted_interval_list:
        if len(merged_interval_list) == 0:
            merged_interval_list.append(interval_list_per_epoch_per_call)
        else:
            start1 = merged_interval_list[len(merged_interval_list) - 1][0]
            end1 = merged_interval_list[len(merged_interval_list) - 1][1]
            start2 = interval_list_per_epoch_per_call[0]
            end2 = interval_list_per_epoch_per_call[1]
            if start2 > end1:
                merged_interval_list.append(interval_list_per_epoch_per_call)
            else:
                if(start2 < start1):
                    merged_interval_list[len(merged_interval_list)-1][0] = start2
                if(end2 > end1):
                    merged_interval_list[len(merged_interval_list)-1][1] = end2

    total_time = 0.0

    fake_actions = []
    for merged_interval_element in merged_interval_list:
        fake_actions.append("Fake")
        total_time += (merged_interval_element[1] - merged_interval_element[0])

    return total_time, zip(fake_actions, get_elements(0, merged_interval_list), get_elements(1, merged_interval_list))

def get_elements(k, list_of_tuples):
    """Return k-th element of each tuple in* list_of_tuples*."""
    return list(map(itemgetter(k), list_of_tuples))

def plot_hrange(events, out_file_name=""):
    d = dict(event=get_elements(0, events),
             begin=get_elements(1, events),
             end=get_elements(2, events),
             out_file_name=out_file_name)
    return plot_hrange_on_vectors(**d)

def plot_hrange_on_vectors(event, begin, end, out_file_name):
    hrange = xrange(int(min(begin)), int(max(end)), 1000)
    vrange = range(len(begin))
    widths = [b - e for b, e in zip(begin, end)]
    plt.barh(vrange, widths, left=begin)
    #plt.xticks(hrange)
    plt.yticks([])
    plt.savefig(out_file_name)

if __name__ == "__main__":
    result_file_path = sys.argv[1]
    number_of_nodes = sys.argv[2]
    output_image = sys.argv[3]
    with open(result_file_path) as csv_file:
        csv.register_dialect('TrimmerDialect', quotechar='"', skipinitialspace=True, quoting=csv.QUOTE_NONE,
                             lineterminator='\n', strict=True)
        dict_reader = csv.DictReader(csv_file, dialect='TrimmerDialect')

        read_text = 'Time to Read Single Image'
        training_iteration_text = 'Training Iteration'
        action_names = []
        start_times = []
        end_times = []
        count_read = 0
        count_training_iteration = 0
        for row in dict_reader:
            if row['Rank'] != '0':
                continue
            if row['Action Description'] == read_text \
                    or row['Action Description'] == training_iteration_text:
                if row['Action Description'] == read_text:
                    count_read += 1
                if row['Action Description'] == training_iteration_text:
                    count_training_iteration += 1

                action_names.append(row['Action Description'])
                start_times.append(float(row['Start Time']))
                end_times.append(float(row['End Time']))

    minimum_start_time = min(start_times)
    maximum_end_time = max(end_times)

    data = zip(action_names, start_times, end_times)

    _, data = merge_and_add_common_intervals(data)

    columns = ('Read', 'Training Iteration')

    # Get some pastel shades for the colors
    colors = plt.cm.BuPu(np.linspace(0, 0.5, len(columns)))

    index = np.arange(len(columns)) + 0.3
    bar_width = 0.4

    '''# Initialize the vertical-offset for the stacked bar chart.
    y_offset_read = np.zeros(len(count_read))
    y_offset_training_iteration = np.zeros(len(count_training_iteration))

    data = [action_names, start_times, end_times]

    n_columns = len(data)
    n_rows = len(action_names)
    y_offset = minimum_start_time
    # Plot bars and create text labels for the table
    for row in range(n_rows):
        if action_names[row] == read_text:
            y_offset = y_offset + start_times[row]
            plt.bar(index, data[0][row], bar_width, bottom=y_offset, color=colors[0])

    # Adjust layout to make room for the table:
    plt.subplots_adjust(left=0.2, bottom=0.2)

    plt.ylabel("Time line")
    plt.xticks(columns)
    plt.title("Time line for {} nodes run".format(number_of_nodes))

    plt.show()'''

    plot_hrange(data, output_image)
