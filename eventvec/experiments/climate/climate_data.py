import csv


from eventvec.experiments.climate.climate_datum import ClimateDatum

class ClimateData:
    def __init__(self):
        self._data = []
        self._tagged_data = []

    def data(self):
        return self._data
    
    def append_datum(self, datum):
        self._data.append(datum)

    def tagged_data(self):
        return self._tagged_data
    
    def append_tagged(self, datum):
        self._tagged_data.append(datum)

    def read_data(self, data_location):
        with open(data_location, 'r') as f:
            reader = csv.reader(f)
            for r in reader:
                climate_datum = ClimateDatum()
                climate_datum.set_label(r[0])
                climate_datum.set_msg_id_parent(r[1])
                climate_datum.set_msg_id_child(r[2])
                climate_datum.set_submission_id(r[3])
                climate_datum.set_body_parent(r[4])
                climate_datum.set_body_child(r[5])
                climate_datum.set_submission_text(r[6])
                climate_datum.set_subreddit(r[7])
                climate_datum.set_author_parent(r[8])
                climate_datum.set_exact_time(r[9])
                climate_datum.set_author_child(r[10])
                climate_datum.set_datetime(r[11])
                climate_datum.set_agreement_fraction(r[12])
                climate_datum.set_individual_kappa(r[13])
                climate_datum.set_parent_topic(r[14])
                climate_datum.set_child_topic(r[15])
                self._data.append(climate_datum)
        return self._data
    
    def write_data(self, write_location):
        with open(write_location, 'w') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow([
                'label',
                'msg_id_parent',
                'msg_id_child',
                'submission_id',
                'body_parent',
                'body_parent_credence_roots',
                'body_parent_credence_roots_reason',
                'body_child',
                'body_child_credence_roots',
                'body_child_credence_roots_reason',
                'submission_text',
                'subreddit',
                'author_parent',
                'exact_time',
                'author_child',
                'datetime',
                'agreement_fraction',
                'individual_kappa',
                'parent_topic',
                'child_topic'
            ])
            for datum in self.tagged_data():
                writer.writerow([
                    datum.label(),
                    datum.msg_id_parent(),
                    datum.msg_id_child(),
                    datum.submission_id(),
                    datum.body_parent(),
                    datum.body_parent_credence_roots(),
                    datum.body_parent_credence_roots_reason(),
                    datum.body_child(),
                    datum.body_child_credence_roots(),
                    datum.body_child_credence_roots_reason(),
                    datum.submission_text(),
                    datum.subreddit(),
                    datum.author_parent(),
                    datum.exact_time(),
                    datum.author_child(),
                    datum.datetime(),
                    datum.agreement_fraction(),
                    datum.individual_kappa(),
                    datum.parent_topic(),
                    datum.child_topic()
                ])
