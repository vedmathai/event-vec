class ClimateDatum():
    def __init__(self):
        self._label = None
        self._msg_id_parent = None
        self._msg_id_child = None
        self._submission_id = None
        self._body_parent = None
        self._body_parent_credence_roots = None
        self._body_parent_credence_roots_reason = None
        self._body_child = None
        self._body_child_credence_roots = None
        self._body_child_credence_roots_reason = None
        self._submission_text = None
        self._subreddit = None
        self._author_parent = None
        self._exact_time = None
        self._author_child = None
        self._datetime = None
        self._agreement_fraction = None
        self._individual_kappa = None
        self._parent_topic = None
        self._child_topic = None

    def label(self):
        return self._label

    def msg_id_parent(self):
        return self._msg_id_parent
    
    def msg_id_child(self):
        return self._msg_id_child
    
    def submission_id(self):
        return self._submission_id
    
    def body_parent(self):
        return self._body_parent
    
    def body_parent_credence_roots(self):
        return self._body_parent_credence_roots

    def body_parent_credence_roots_reason(self):
        return self._body_parent_credence_roots_reason
    
    def body_child(self):
        return self._body_child
    
    def body_child_credence_roots(self):
        return self._body_child_credence_roots
    
    def body_child_credence_roots_reason(self):
        return self._body_child_credence_roots_reason
    
    def submission_text(self):
        return self._submission_text

    def subreddit(self):
        return self._subreddit
    
    def author_parent(self):
        return self._author_parent
    
    def exact_time(self):
        return self._exact_time
    
    def author_child(self):
        return self._author_child
    
    def datetime(self):
        return self._datetime
    
    def agreement_fraction(self):
        return self._agreement_fraction
    
    def individual_kappa(self):
        return self._individual_kappa
    
    def parent_topic(self):
        return self._parent_topic
    
    def child_topic(self):
        return self._child_topic
    
    def set_label(self, label):
        self._label = label

    def set_msg_id_parent(self, msg_id_parent):
        self._msg_id_parent = msg_id_parent

    def set_msg_id_child(self, msg_id_child):
        self._msg_id_child = msg_id_child
    
    def set_submission_id(self, submission_id):
        self._submission_id = submission_id

    def set_body_parent(self, body_parent):
        self._body_parent = body_parent

    def set_body_parent_credence_roots(self, body_parent_credence_roots):
        self._body_parent_credence_roots = body_parent_credence_roots

    def set_body_parent_credence_roots_reason(self, body_parent_credence_roots_reason):
        self._body_parent_credence_roots_reason = body_parent_credence_roots_reason

    def set_body_child(self, body_child):
        self._body_child = body_child

    def set_body_child_credence_roots(self, body_child_credence_roots):
        self._body_child_credence_roots = body_child_credence_roots

    def set_body_child_credence_roots_reason(self, body_child_credence_roots_reason):
        self._body_child_credence_roots_reason = body_child_credence_roots_reason
    
    def set_submission_text(self, submission_text):
        self._submission_text = submission_text
    
    def set_subreddit(self, subreddit):
        self._subreddit = subreddit

    def set_author_parent(self, author_parent):
        self._author_parent = author_parent

    def set_exact_time(self, exact_time):
        self._exact_time = exact_time

    def set_author_child(self, author_child):
        self._author_child = author_child

    def set_datetime(self, datetime):
        self._datetime = datetime

    def set_agreement_fraction(self, agreement_fraction):
        self._agreement_fraction = agreement_fraction

    def set_individual_kappa(self, individual_kappa):
        self._individual_kappa = individual_kappa

    def set_parent_topic(self, parent_topic):
        self._parent_topic = parent_topic
    
    def set_child_topic(self, child_topic):
        self._child_topic = child_topic
    
    def to_dict(self):
        return {
            'label': self.label(),
            'msg_id_parent': self.msg_id_parent(),
            'msg_id_child': self.msg_id_child(),
            'submission_id': self.submission_id(),
            'body_parent': self.body_parent(),
            'body_parent_credence_roots': self.body_parent_credence_roots(),
            'body_parent_credence_roots_reason': self.body_parent_credence_roots_reason(),
            'body_child': self.body_child(),
            'body_child_credence_roots': self.body_child_credence_roots(),
            'body_child_credence_roots_reason': self.body_child_credence_roots_reason(),
            'submission_text': self.submission_text(),
            'subreddit': self.subreddit(),
            'author_parent': self.author_parent(),
            'exact_time': self.exact_time(),
            'author_child': self.author_child(),
            'datetime': self.datetime(),
            'agreement_fraction': self.agreement_fraction(),
            'individual_kappa': self.individual_kappa(),
            'parent_topic': self.parent_topic(),
            'child_topic': self.child_topic()
        }
    
    @staticmethod
    def from_dict(d):
        datum = ClimateDatum()
        datum.set_label(d['label'])
        datum.set_msg_id_parent(d['msg_id_parent'])
        datum.set_msg_id_child(d['msg_id_child'])
        datum.set_submission_id(d['submission_id'])
        datum.set_body_parent(d['body_parent'])
        datum.set_body_parent_credence_roots(d['body_parent_credence_roots'])
        datum.set_body_parent_credence_roots_reason(d['body_parent_credence_roots_reason'])
        datum.set_body_child(d['body_child'])
        datum.set_body_child_credence_roots(d['body_child_credence_roots'])
        datum.set_body_child_credence_roots_reason(d['body_child_credence_roots_reason'])
        datum.set_submission_text(d['submission_text'])
        datum.set_subreddit(d['subreddit'])
        datum.set_author_parent(d['author_parent'])
        datum.set_exact_time(d['exact_time'])
        datum.set_author_child(d['author_child'])
        datum.set_datetime(d['datetime'])
        datum.set_agreement_fraction(d['agreement_fraction'])
        datum.set_individual_kappa(d['individual_kappa'])
        datum.set_parent_topic(d['parent_topic'])
        datum.set_child_topic(d['child_topic'])
        return datum

    def copy(self):
        return ClimateDatum.from_dict(self.to_dict())
