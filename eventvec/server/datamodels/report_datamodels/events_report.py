class EventsReport:
    def __init__(self):
        self._event_1 = None
        self._event_2 = None
        self._relationship_distribution = {}
        self._similarity = None

    def to_dict(self):
        return {
            'event_1': self._event_1.to_dict(),
            'event_2': self._event_2.to_dict(),
            'relationship_distribution': sorted(self._relationship_distribution.items(), key=lambda x: x[1]),
            'similarity': self._similarity,
        }

    def __repr__(self):
        return str(self.to_dict())

    @staticmethod
    def create(event_1, event_2, relationship_distribution, similarity):
        event_report = EventsReport()
        event_report._event_1 = event_1
        event_report._event_2 = event_2
        event_report._relationship_distribution = relationship_distribution
        event_report._similarity = similarity
        return event_report
