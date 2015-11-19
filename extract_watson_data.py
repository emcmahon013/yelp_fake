import requests
import json
import pandas as pd

def watson_request(text):
    base_url = "https://gateway.watsonplatform.net/tone-analyzer-experimental/api/v1/tone"

    r = requests.post(base_url, data = text, headers = {"Content-Type" : "text/plain"}, 
                      auth=("84a3bf89-c5f3-45b6-9ee2-5c6ee1d66fa5", "t95US4RGaQnX"))

    response = json.loads(r.text)
    return response

def create_row(response, text_id):
    row = {}
    row["Filename"] = text_id
    for tone in response['children']:
        tone_id = tone['id']
        for features in tone['children']:
            characteristic_id = tone_id + "_" + features['id']
            row[characteristic_id + "_" + "word_count"] = features['word_count']
            row[characteristic_id + "_" + "normalized_score"] = features['normalized_score']
            row[characteristic_id + "_" + "raw_score"] = features['raw_score']
    return row

watson_data_negative = []
import os
for subdir, dirs, files in os.walk('./op_spam_v1.4/negative_polarity/'):
    for file in files:
        text = open(str(subdir) + "/" + str(file)).read()
        response = watson_request(text)
        row = create_row(response, file)
        watson_data_negative.append(row)



watson_data_positive = []
import os
for subdir, dirs, files in os.walk('./op_spam_v1.4/positive_polarity/'):
    for file in files:
        text = open(str(subdir) + "/" + str(file)).read()
        response = watson_request(text)
        row = create_row(response, file)
        watson_data_positive.append(row)


watson_df_negative = pd.DataFrame(watson_data_negative)
watson_df_positive = pd.DataFrame(watson_data_positive)
