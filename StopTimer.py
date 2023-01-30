import time


class StopTimer:
    def __init__(self):

        self.t_start = 0
        self.t_stop = 0
        self.t_duration = 0

    def start(self):
        self.t_start = time.time()
        return self.t_start

    def stop(self):
        self.t_stop = time.time()
        return self.t_stop

    def duration(self):
        self.t_duration = round(self.t_stop - self.t_start, 3)
        return self.t_duration

    def reset(self):
        self.t_start = 0
        self.t_stop = 0
        self.t_duration = 0


#
# x = StopTimer()
#
# x.start()
# time.sleep(1)
#
# x.stop()
#
# print(x.duration())
# x.reset()
