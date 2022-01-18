import json


class Report:
    def __init__(self):
        self._event_reports = []

    def add_events_report(self, event_report):
        self._event_reports.append(event_report)

    def to_dict(self):
        return {
            'event_reports': [i.to_dict() for i in self._event_reports],
        }

    def __repr__(self):
        return str(self.to_dict())

    def to_file(self, file_name):
        with open(file_name, 'wt') as f:
            json.dump(self.to_dict(), f, indent=4)
