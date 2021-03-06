import os
import sys
import time

import tensorflow as tf
import threading


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
        self.process_id = 0
        self.current_thread = ""

    def __del__(self):
        if self.write_to_file and self.file_ptr > 0:
            self.file_ptr.close()

    def set_rank(self, rank):
        self.rank = rank

    def set_action_description(self, action_description):
        self.action_description = action_description

    def set_epoch_num(self, epoch_num):
        self.epoch_num = epoch_num

    def set_current_thread(self, current_thread):
        self.current_thread = current_thread

    def set_process_id(self, process_id):
        self.process_id = process_id

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

        # Normally end_timer() is called from the same process and thread
        self.process_id = os.getpid()
        self.current_thread = str(threading.current_thread())

        if self.write_to_file:
            self.write_log()
        else:
            self.print_log()

    def print_log(self):
        time_taken = self.end_time - self.start_time
        if time_taken < 0:
            print("TIME LOGGER OUTPUT", "ERROR", "End time is less than start time")
        print("TIME LOGGER OUTPUT", "Epoch: %d"%(self.epoch_num), "Rank: %d"%(self.rank), self.action_name, self.action_description,
              "Start Time: %g"%(self.start_time), "End Time: %g"%(self.end_time), "Time Taken: %g"%(time_taken),
              "For Excel:{}, {}, {}, {}, {}, {}, {}, {}, {}".format(self.epoch_num, self.rank, self.action_name, self.action_description,
                                                         self.start_time, self.end_time, time_taken, self.process_id, self.current_thread))

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
            str(self.end_time) + "," + str(time_taken) + "," + str(self.process_id) + "," + self.current_thread + "\'")

    def get_caller_name(self):
        return sys._getframe(2).f_code.co_name

# Listener class for logging checkpointing events


class TimeLoggerCheckpointSaverListener(tf.train.CheckpointSaverListener):
    def __init__(self, rank=-1, write_to_file=False, process_id=0, current_thread=""):
        self.rank = rank
        self.write_to_file = write_to_file
        self.process_id = process_id
        self.current_thread = current_thread

    def begin(self):
        self.session_logger = TimeLogger(self.rank, "Session Listener", -1, self.write_to_file)
        self.session_logger.start_timer()
        self.session_logger.set_process_id(self.process_id)
        self.session_logger.set_current_thread(self.current_thread)

        self.checkpoint_logger = TimeLogger(self.rank, "Checkpoint Listener", -1, self.write_to_file)

    # TODO: For now, considering global_step_value as the epoch number in TimeLogger. Need to fix later.
    def before_save(self, session, global_step_value):
        self.checkpoint_logger.set_epoch_num(global_step_value)
        self.checkpoint_logger.start_timer()

    def after_save(self, session, global_step_value):
        self.checkpoint_logger.end_timer()

    def end(self, session, global_step_value):
        self.session_logger.end_timer()
