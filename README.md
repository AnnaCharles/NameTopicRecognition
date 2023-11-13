# NameTopicRecognition  
This repository allows you to apply name entity recognition and topic detection on the parliamentary data.

# Demo 

## Name entity recognition
Run the name entity recognition on the data file.The file should contain a column 'contenu' with the text, a column 'intervenant_nom' and a column 'intervenant_fonction' with respectively the name of the speaker and its fonction.
You can chose between the algorithm spacy and stanza. Stanza takes longer than Spacy but it is more efficient to recognize names.

```bash
python __main__.py ner stanza data_file
```

## Name entity recognition statitics
From the results of the NER, create a file with 4 columns data_file+algo+'_statistiques': 
- 'last' : last token of the NE that we count
-'longest' : longest NE contaning this last token
- 'personnes' : all the NE having this last token
- 'count' : sum of number of occurrences of the NE in personnes

```bash
python __main__.py frequencies_ner stanza data_file
```
Then to visualize the results in a histogram you can use xsv :
```bash
xsv map 'val("last_name")' field data_file_stanza_statistiques.csv | xsv sort -s count -N | xsv hist --label longest --value count
```
## To get the enunciates of the questions for each 'sous section'
To then do the topic detection on the data file, you should first do this :
```bash
python __main__.py enunciate data_file demo
```
demo is the name template, you should use the same for the topic detection. 
This creates a file with the list of the 'sous section' and the enunciate of the question for each 'sous section'.

## Topic recognition results and visualization
Create a file with for each line of the data file its associated topic, a file with the list of the 'sous section' associated to a topic, and a visualisation of the topics. You can specify if it is not the first time using this data file with this command with '--first_time', and the execution will be faster. You can specify if you want to visualize only the questions with "--questions_only". You can specify if you want to use the labels of the topics defined with the dataset nosdeputes-questions_orales_220620-230723.csv with the parameter '--use_labels'. 

```bash
 python __main__.py topics_recognition data_file demo --questions_only 
```

## List of the topics for each question/sous section
From the results of the command topics_recognition, create a file with for each sous section, its associated topics.
```bash
 python __main__.py topics_question demo 
```
