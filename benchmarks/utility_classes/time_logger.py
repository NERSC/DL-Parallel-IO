import time


class TimeLogger:
    def __init__(self, rank, action):
        # Initialize members
        self.start_time = 0
        self.end_time = 0
        self.rank = rank
        self.action = action

    def start_timer(self, rank=-1, action=""):
        if action != "":
            self.action = action
        if rank != -1:
            self.rank = rank

        self.start_time = time.time()
        self.end_time = 0

    def end_timer(self):
        self.end_time = time.time()
        self.print_log()

    def print_log(self):
        time_taken = self.end_time - self.start_time
        if time_taken < 0:
            print("TIME LOGGER OUTPUT", "ERROR", "End time is less than start time")
        print("TIME LOGGER OUTPUT", "Rank %d"%(self.rank), self.action, "Start Time: %g, End Time: %g Time Taken: %g"%(self.start_time, self.end_time, time_taken))
