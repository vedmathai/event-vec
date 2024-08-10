# Event Vec
## What it is
This project aims to vectorize event mentions in text. Earlier implementations use a version of the skip-gram model to vectorized event mentions. This project aims to incorporate temporal information in to the process of event vectorization.

## Initial Design
We learn the event vectors by feeding through a network that takes the vectorized versions of the parts of two event mention as inputs and outputs the temporal relationship between them. An additional output is the confidence of distribution.

## Parts of the code
### Data Ingestor
This modular component can be swapped out for other components that read from different types of sources such as Wikipedia or ebooks.

### Coreference resolution
This modular component can be swapped out for other components that resolve coreferences within the same text.

### Event extractor
This component, based on dependency parses and other hand-created rules, will extract the verb, subject, object etc for a given event mentions.

### Date extractor
This component based on Spacy's entity recognizer and hand-created rules will extract date mentions in the text and attempt to normalize them

### Event relationship extractor
Using Spacy's dependency parser, the relationships between events or events and dates are extracted. This is learned from TIMEBANK1.2.

### Vectorizer
The network as described in the intial design.
### Utils
The component that is able to learn the relationships between entity and dates from the TIMEBANK data.

## Testing
The vectors will be tested against the tasks of
* event similarity
* event cloze
* event year prediction

And the implementations of Modi and Granroth-clark for event vectorization.

Also added will be an ablation study that is able to find the impact of
* coreference resolution
* relationship extractor and the different methods for it.

## Downstream usage
For this to be useful, one can
* train with their own data
* feed events in a structured format and obtain event vectors
* feed sentences and obtain events in an structured format and their vectors.

## Demo
To quickly see the project and what it is trying to achieve. There will be a demo which can take two events and output the similarity. Take an event and output the similar events from the trained data.

# Running on Jade
PYTHONPATH=REMOTE_CODE_FOLDER_PATH/event-vec ENV=JADE RUNCONFIGID=41 python3.8 REMOTE_CODE_FOLDER_PATH/event-vec/eventvec/server/entry_points/main.py

# Running Locally
REQUEST_ID=1234 PROJECT_ID=event-vec RUNCONFIGID=21 ENV=JADE PYTHONPATH=jade_front/event-vec/code/event-vec python jade_front/event-vec/code/event-vec/eventvec/server/entry_points/main.py
