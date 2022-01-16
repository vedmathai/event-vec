class EventRelationship:
    def __init__(self):
        self._event_1 = None
        self._event_2 = None
        self._relationship = None
        self._relationship_score = 0

    def to_dict(self):
        return {
            "event_1": self._event_1.to_dict(),
            "event_2": self._event_2.to_dict(),
            "relationship": self._relationship,
            "relationship_score": self._relationship_score
        }

    def __repr__(self) -> str:
        return str(self.to_dict())

    def relationship(self):
        return self._relationship

    def relationship_score(self):
        return self._relationship_score

    def event_1(self):
        return self._event_1
        
    def event_2(self):
        return self._event_2

    @staticmethod
    def create(event_1, event_2, relationship, relationship_score):
        event_relationship = EventRelationship()
        event_relationship._event_1 = event_1
        event_relationship._event_2 = event_2
        event_relationship._relationship = relationship
        event_relationship._relationship_score = relationship_score
        return event_relationship
