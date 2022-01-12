
class EventDetector():
    def detect(self, psentence):
        event_roots = []
        for word in psentence:
            if word.pos() == "VERB":
                event_roots.append(word)
        return event_roots
