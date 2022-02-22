import datetime


class Event:
    """
    Represent an ICME event object with start and stop time
    """

    def __init__(self, begin, end):
        self.begin = begin
        self.end = end

    def duration(self):
        """
        Compute the duration of this event

        :return:self.end - self.begin
        """
        return self.end - self.begin

    def __eq__(self, other):
        """
        Return True if other overlaps self during a 65% of its duration
        :param other: an other event
        :return: overlap(self, other) > 0.65 * self.duration
        """
        duration = self.duration()
        return overlap(self, other) > 0.65 * duration

    def __str__(self):
        return "{} ---> {}".format(self.begin, self.end)


def overlap(event1, event2):
    """
    Return the time overlapping between two events as a timedelta

    :param event1: first event
    :param event2: second event
    :return: max(event1.begin, event2.begin) - min(event1.end, event2.end)
    """
    delta1 = min(event1.end, event2.end)
    delta2 = max(event1.begin, event2.begin)
    return max(delta1 - delta2,
               datetime.timedelta(0))


