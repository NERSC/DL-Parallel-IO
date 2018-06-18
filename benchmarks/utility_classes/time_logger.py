import time

class time_logger:
    def __init__(self):
        # Initialize members
        self.start_time = 0
        self.end_time = 0

    def start_timer(self):
        self.start_time = time.time()

    def end_timer(self):
        self.end_time = time.time()

    def print_log(self, message):
        time_taken = self.end_time - self.start_time
        print("TIME LOGGER OUTPUT", message, "Time Taken: %g"%(time_taken))
