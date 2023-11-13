import argparse
import logging
from sentence_transformers import SentenceTransformer
import json 
import numpy as np
import os
import NameEntityRecognition.NER_spacy as spacyobject 
import NameEntityRecognition.NER_stanza as stanzaobject 
import NameEntityRecognition.NER_statistiques as statistiques
import enonce_de_la_question
import TopicRecognition.topic_recognition as recognition
import TopicRecognition.topic_par_soussection as par_ssection
import TopicRecognition.questions_par_topic as par_topic

from datetime import date

#logger = logging.getLogger("pimmi")
parser = argparse.ArgumentParser()
# config_path = os.path.join(os.path.dirname(__file__), "config.yml")
# config_dict = prm.load_config_file(config_path)
# for param, value in config_dict.items():
#     if type(value) == dict and "help" in value:
#         help_message = value["help"]
#         default = value["value"]
#         argument_type = type(value["value"])
#     else:
#         help_message = argparse.SUPPRESS
#         argument_type = type(value)
#         default = value
    
#     parser.add_argument(
#         "--{}".format(param.replace("_", "-")),
#         type=argument_type,
#         default=default,
#         help=help_message
#     )

def load_cli_parameters():
    subparsers = parser.add_subparsers(title="commands")

# NER command
    ner_parser = subparsers.add_parser('ner', help="Create a file with 4 columns data_file+'NER_'+algo+'_results': \n"
                                       "- 'text' : text of the data file \n -'size' : its number of tokens \n"
                                       "- 'ner' : the list of the name entities recognized by the algorithm in the text \n"
                                       "- 'description' : description of each name entity")
    ner_parser.add_argument('algo', type=str, metavar='algo',choices=['spacy','stanza'], help="Chose the algorithm spacy or stanza to do the name entity recognition."
                            " Stanza takes longer than Spacy but it is more efficient to recognize names.")
    ner_parser.add_argument('data_file', type=str, help="The file containing the data. The file should contain a column 'contenu' with the text, "
                            ", a column 'intervenant_nom' and a column 'intervenant_fonction' with respectively the name of the speaker and its fonction.")
    ner_parser.set_defaults(func=ner)

# frequencies NER command
    frequencies_ner_parser = subparsers.add_parser('frequencies_ner', help="From the results of the NER, create a file with 4 columns data_file+algo+'_statistiques': \n"
                                                    "- 'last' : last token of the NE that we count \n -'longest' : longest NE contaning this last token  \n"
                                                 "- 'personnes' : all the NE having this last token \n"
                                                 "- 'count' : sum of number of occurrences of the NE in personnes")
    
    frequencies_ner_parser.add_argument('algo', type=str, metavar='algo',choices=['spacy','stanza'], help="The algorithm spacy or stanza used to do the name entity recognition")
    frequencies_ner_parser.add_argument('data_file', type=str, help="The file containing the data. The file should contain a column 'contenu' with the text, "
                            ", a column 'intervenant_nom' and a column 'intervenant_fonction' with respectively the name of the speaker and its fonction.")
    frequencies_ner_parser.set_defaults(func=frequencies_ner)

# enunciates of the questions command
    enunciate_parser = subparsers.add_parser('enunciate', help="Create a file with the list of the 'sous section' and the enunciate of the question for each 'sous section'")
    enunciate_parser.add_argument('data_file', type=str, help="The file containing the data. The file should contain a column 'contenu' with the text, "
                            ", a column 'intervenant_nom' and a column 'intervenant_fonction' with respectively the name of the speaker and its fonction.")
    enunciate_parser.add_argument('name_template', type=str, help="Name used for the name template of every output files concerning this data file")
    enunciate_parser.set_defaults(func=enunciate)

#topic recognition results and visualisation command
    topics_parser = subparsers.add_parser('topics_recognition', help="Create a file with for each line of the data file its associated topic,"
                                   " a file with the list of sous section associated to a topic, and a visualisation of the topics")
    topics_parser.add_argument('data_file', type=str, help="The file containing the data.")
    topics_parser.add_argument('--first_time', action='store_false', help="By default, at True, it means it is the first time using the dataset\n"
                               "False : the dataset has already been used with the command topics_recognition, gain of time")
    topics_parser.add_argument('name_template', type=str, help="Name used for the name template of every output files concerning this data file ")
    topics_parser.add_argument('--questions_only', action='store_true',  help="By default at False, print all the points in the visualisation \n"
                               "True : the visualisaton only print the points corresponding to the enunciate of the questions") 
#si le user ne met pas --questions_only : sa valeur est à False et s'il met --questions-only: sa valeur est à True
# idem pour use_labels
    topics_parser.add_argument('--use_labels',action='store_true', help="By default at False, do not use any label for the topics \n"
                               "True : Use the labels defined for the document nosdeputes to qualify the topics")
    topics_parser.set_defaults(func=topics)

# topics for each question command
    topics_question_parser=subparsers.add_parser('topics_question', help="From the results of the command topics_recognition, create a file with for each sous section , its associated topics")
    topics_question_parser.add_argument('name_template', type=str, help="Name used for the name template of every output files concerning the data used with the command topics_recognition")
  
    topics_question_parser.set_defaults(func=topics_question)


    cli_parameters = vars(parser.parse_args())

    return cli_parameters

def ner(algo, data_file) :
    if algo=="spacy" :
        spacyobject.ner(data_file)
    if algo=="stanza" :
        stanzaobject.ner(data_file)

def frequencies_ner(algo,data_file):
    statistiques.stats(algo,data_file)

def enunciate(data_file,name_template):
    enonce_de_la_question.enunciate(data_file,name_template)

def topics(data_file, first_time,name_template,questions_only, use_labels):
    titles=[]
    phrases_completes=[]
    if first_time :
        results=recognition.get_docs(data_file,name_template)
        docs = results[0]
        titles= results[1]
        phrases_completes=results[2]

    else :
        results=recognition.get_docs_filtres(name_template)
        docs = results[0]
        phrases_completes=results[2]

    embedding_model = SentenceTransformer("dangvantuan/sentence-camembert-large")
    embeddings=[]
    print(len(docs))
    print(first_time)

    if not first_time :
        embeddings = embedding_model.encode(docs, show_progress_bar=True)
        print(len(embeddings))
        print(len(embeddings.tolist()))
        with open(name_template+"_embeddings.json","w") as f :
            json.dump(embeddings.tolist(), f)
    else :
        with open(name_template+"_embeddings.json") as f :
            embeddings = np.array(json.load(f))
    


    topic_model = recognition.set_model_parameters(embedding_model)
    topic_model.fit(docs, embeddings)

    print(topic_model.generate_topic_labels(nr_words=5, topic_prefix=True, word_length=None, separator='  --  '))
    print(topic_model.get_topic_info())

    labels=None
    if use_labels:
        labels=recognition.labels
        par_topic.questions_par_topic(phrases_completes,labels,topic_model.topics_,name_template,use_labels)
    else :
        par_topic.questions_par_topic(phrases_completes,topic_model.get_topic_info()["Name"],topic_model.topics_,name_template,use_labels)

    recognition.store_results(name_template,topic_model,docs,labels)
    
    if questions_only:
        recognition.visualisation(True,False, phrases_completes,None, name_template, topic_model,embeddings,labels)
    else :
        recognition.visualisation(False,False, None,titles, name_template, topic_model,embeddings,labels)

def topics_question(name_template):
    par_ssection.classification(name_template)


def main():
    cli_params = load_cli_parameters()
    # if cli_params["silent"]:
    #     logger.setLevel(level=logging.ERROR)
    if "func" in cli_params:
        #config_path = cli_params.pop("config_path", None)
        #check_custom_config(config_path)

        command = cli_params.pop("func")
        command(**cli_params)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()