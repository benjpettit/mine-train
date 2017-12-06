# mine-train
interactive text classifier

## Background

This is a demonstration of a simple active learning system, which incrementally trains a machine learning model for classifying text, based on feedback from the user. I consider the case of a small corpus with a few hundred thousand documents at most, which can be loaded in memory in their raw and vectorized forms (e.g. ~500Mb).

The front end requests a batch of, say, 5 documents, which are initially picked at random. The user rates these documents one at a time as "liked" or "disliked". The back end incrementally fits a logistic regression model using TF-IDF bag-of-words features and SGD optimization. When the front end requests a new batch, the documents sent to the user to rate are picked according to the model's latest estimation. The model can either _explore_ the documents it is least certain about, or _exploit_ its understanding of the user's preferences by showing the documents the user is most likely to prefer.

Active learning is an interesting and evolving field of machine learning. This interactive demonstration, using a relatively basic machine learning approach, helps highlight the opportunities and common pitfalls of trying to model the user's judgement as quickly as possible, by prompting the user for feedback on one item at a time.

## Installation

I recommend installing the application in a Python 3.5 virtual environment.
```
pip install -r requirements.txt
pip install -U .
```

## Input data

The documents to be classified should be in text files, where each line is one document formatted as JSON that contains the fields "title" and "abstract". This just happened to be the format of the dataset I first used. I have plans to abstract the file reader to make it easier to swap file formats.

## Running the application

`python app/main.py $INPUT_DIRECTORY [$PORT]`

The application will attempt to read all files in the directory specified by $INPUT_DIRECTORY. If the optional argument $PORT is not specified, the application will run on port 5000.

## API

GET /nextItems
Returns an array of JSON objects for the user to score, each with fields `id`, `title`, `description`.

POST /label
Content-Type: application/json
Example body: `{"id":"8f19b755-0bd1-327c-86fd-ede1a0f0e05d","liked":true}`
Body fields: `id` (String), `liked` (Boolean)
Send the user's feedback regarding whether or not the user liked the document with the given id. Returns a message.
