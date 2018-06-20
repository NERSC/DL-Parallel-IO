import sys
import time


class TimeLogger:
    def __init__(self, rank, action_description):
        # Initialize members
        self.start_time = 0
        self.end_time = 0
        self.rank = rank
        self.action_name = ""
        self.action_description = action_description

    def set_rank(self, rank):
        self.rank = rank

    def set_action_description(self, action_description):
        self.action_description = action_description

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
        self.print_log()

    def print_log(self):
        time_taken = self.end_time - self.start_time
        if time_taken < 0:
            print("TIME LOGGER OUTPUT", "ERROR", "End time is less than start time")
        print("TIME LOGGER OUTPUT", "Rank %d"%(self.rank), self.action_name, self.action_description,
              "Start Time: %g"%(self.start_time), "End Time: %g"%(self.end_time), "Time Taken: %g"%(time_taken))

    def get_caller_name(self):
        return sys._getframe(2).f_code.co_name
