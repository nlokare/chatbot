import csv
import json

def build_training_doc(input_filename, output_filename):
  training_doc = {
    'intents': []
  }
  with open(input_filename, 'rb') as f:
    reader = csv.DictReader(f)
    tag_patterns = {}
    for row in reader:
      tag = row['Tag']
      pattern = row['Pattern']
      if tag == '':
        break
      if tag_patterns.get(tag):
        tag_patterns[tag].append(pattern)
      else:
        tag_patterns[tag] = [pattern]
  f.close()
  for key, value in tag_patterns.iteritems():
    training_doc['intents'].append({
        'tag': key,
        'patterns': value
      })

  with open(output_filename, 'wb') as out:
    out.write(json.dumps(training_doc))
  out.close()
  return

def build_responses_doc(input_filename, output_filename):
  responses_doc = {
    'Unknown': 'Sorry, I don\'t have a suitable answer for your question. \
    Can you please try rewording your question?'
  }
  with open(input_filename, 'rb') as f:
    reader = csv.reader(f)
    headers = next(reader)[0:]
    for row in reader:
      for idx, val in enumerate(headers):
        response = unicode(row[idx], errors='replace')
        if responses_doc.get(val) is not None:
          responses_doc[val].append(response)
        else:
          responses_doc[val] = [response]

  with open(output_filename, 'wb') as out:
    out.write(json.dumps(responses_doc))
  out.close()
  return

# build_responses_doc('responses_raw.csv', 'responses.json')
# build_training_doc('training_data_raw.csv', 'intents.json')
