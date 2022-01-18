class EventRelationship:
    def __init__(self):
        self._event_1 = None
        self._event_2 = None
        self._relationship_distribution = {}
        self._relationships = {}

    def to_dict(self):
        return {
            "event_1": self._event_1.to_dict(),
            "event_2": self._event_2.to_dict(),
            "relationship_distribution": self._relationship_distribution,
            "relationships": self._relationships,
        }

    def __repr__(self) -> str:
        return str(self.to_dict())

    def normalize_distribution(self):
        total = float(sum(self._relationships.values()))
        self._relationship_distribution = {k: v/total for k, v in self._relationships.items()}

    def relationship_distribution(self):
        return self._relationship_distribution

    def relationships(self):
        return self._relationships

    def add_relationship_type(self, relationship_type, relationship_score):
        self._relationships[relationship_type] = relationship_score

    def event_1(self):
        return self._event_1
        
    def event_2(self):
        return self._event_2

    @staticmethod
    def create(event_1, event_2):
        event_relationship = EventRelationship()
        event_relationship._event_1 = event_1
        event_relationship._event_2 = event_2
        return event_relationship
