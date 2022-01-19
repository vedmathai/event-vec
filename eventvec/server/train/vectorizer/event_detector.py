
class EventDetector():
    def detect(self, psentence):
        verb_roots = self._detect_verbs(psentence)
        date_roots = self._detect_dates(psentence)
        return {
            "verb_roots": verb_roots,
            "date_roots": date_roots,
        }
        
    def _detect_verbs(self, psentence):
        verb_roots = []
        for word in psentence:
            if word.pos() == "VERB":
                verb_roots.append(word)
        return verb_roots

    def _detect_dates(self, psentence):
        date_roots = []
        for word in psentence:
            if word.ent_type() == 'DATE' and word.dep() == 'pobj':
                date_roots.append(word)
        return date_roots
