import sys
import time


class TimeLogger:
    def __init__(self, rank, action_description, epoch_num=-1, write_to_file=False):
        # Initialize members
        self.start_time = 0
        self.end_time = 0
        self.rank = rank
        self.action_name = ""
        self.action_description = action_description
        self.epoch_num = epoch_num
        self.write_to_file = write_to_file
        self.file_ptr = 0

    def __del__(self):
        if self.write_to_file == True and self.file_ptr > 0:
            self.file_ptr.close()

    def set_rank(self, rank):
        self.rank = rank

    def set_action_description(self, action_description):
        self.action_description = action_description

    def set_epoch_num(self, epoch_num):
        self.epoch_num = epoch_num

    def start_timer(self, rank=-1, action_description=""):
        if rank != -1:
            self.rank = rank
        if action_description != "":
            self.action_description = action_description

        self.action_name = self.get_caller_name()
        self.start_time = time.time()
        self.end_time = 0

    def end_timer(self):
        self.action_name = self.get_caller_name()
        self.end_time = time.time()
        if self.write_to_file == True:
            self.write_log()
        else:
            self.print_log()

    def print_log(self):
        time_taken = self.end_time - self.start_time
        if time_taken < 0:
            print("TIME LOGGER OUTPUT", "ERROR", "End time is less than start time")
        print("TIME LOGGER OUTPUT", "Epoch: %d"%(self.epoch_num), "Rank: %d"%(self.rank), self.action_name, self.action_description,
              "Start Time: %g"%(self.start_time), "End Time: %g"%(self.end_time), "Time Taken: %g"%(time_taken),
              "For Excel:{}, {}, {}, {}, {}, {}, {}".format(self.epoch_num, self.rank, self.action_name, self.action_description,
                                                         self.start_time, self.end_time, time_taken))

    def write_log(self):
        filename = str(self.rank)
        if self.rank == -1:
            filename = "all_ranks"
        self.file_ptr = open("time_logger_output/" + filename, "a")

        if self.file_ptr == 0:
            print("No log file created")
            return

        time_taken = self.end_time - self.start_time
        if time_taken < 0:
            self.file_ptr.write("\'TIME LOGGER OUTPUT\', \'ERROR\', \'End time is less than start time\'")
        self.file_ptr.write("\'TIME LOGGER OUTPUT\'," + "\'Epoch: " + str(self.epoch_num) + "\', \'Rank: " +
            str(self.rank) + "\',\'" + self.action_name + "\',\'" + "\',\'" + self.action_description + "\'," +
            "\'Start Time: " + str(self.start_time) + "\',\'End Time: " + str(self.end_time) +
            "\', \'Time Taken: " + str(time_taken) + "\',\'For Excel:" + str(self.epoch_num) + "," + str(self.rank) +
            "," + self.action_name + "," + self.action_description + "," + str(self.start_time) + "," +
            str(self.end_time) + "," + str(time_taken) + "\'")

    def get_caller_name(self):
        return sys._getframe(2).f_code.co_name
