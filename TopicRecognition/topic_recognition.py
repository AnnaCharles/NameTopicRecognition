from sentence_transformers import SentenceTransformer
import umap
from hdbscan import HDBSCAN
from bertopic.vectorizers import ClassTfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from bertopic import BERTopic
from nltk.corpus import stopwords
import nltk
from sentence_transformers import SentenceTransformer
import csv
import casanova
import spacy 
from tqdm import tqdm
from datetime import date
import sys
import numpy as np 
import json 
import TopicRecognition.topics_visualisation as tv
from collections import Counter
import TopicRecognition.questions_par_topic as qt

def get_stopwords():
    nltk.download('stopwords')
    stoplist = stopwords.words("french")
    ADDITIONAL_STOPWORDS = ["plus", "chaque", "tout", "tous", "toutes", "toute", "leur", "leurs", "comme", "afin", "pendant", "lorsque","tant",'monsieur',"madame",'ministre',"france"]
    stop=[]
    with open("TopicRecognition/stopwords_twitter_fr.csv") as f :
        reader = csv.DictReader(f)
        for row in reader :
            stop.append(row['stopwords_fr'])
        stoplist = stopwords.words("french")
        ADDITIONAL_STOPWORDS = ["question","questions","réponses","réponses","plus", "chaque", "tout", "tous", "toutes", "toute", "leur", "leurs", "comme", "afin", "pendant", "lorsque","tant",'monsieur',"madame",'ministre',"france"]
        

        stop=list(set(stoplist+stop+ADDITIONAL_STOPWORDS))

    return stop



def get_docs(DATAFILE,name_template):
    nlp = spacy.load("fr_core_news_sm")
    docs=[]
    titles=[]
    phrases_completes=[]
    with open(DATAFILE) as f, open(name_template+"_filtered_file.csv", 'w') as of :
        reader = csv.DictReader(f)
        enricher = csv.DictWriter(of, fieldnames=reader.fieldnames+["contenu_entier"])
        enricher.writeheader() 
        for i,row in enumerate(reader) :
            cell= row['contenu']
            if row["intervenant_fonction"] != "présidente" and row["intervenant_nom"] != "":
                doc=nlp(cell)
                if len(doc) > 10 :
                    entier=cell
                    phrases_completes.append(cell)
                    while len(doc) >=300 :
                        cell=cell[:cell.rfind(".")]
                        doc=nlp(cell)
                    docs.append(cell)
                    titles.append(row['soussection'])
                    enricher.writerow({"seance_titre":row['seance_titre'],"date":row['date'],"heure":row['heure'],"timestamp":row['timestamp'],'soussection':row['soussection'],
                                       'intervenant_nom':row['intervenant_nom'],'intervenant_fonction':row['intervenant_fonction'],
                                       'intervenant_groupe':row['intervenant_groupe'],'nbmots':row['nbmots'],"contenu":cell,'id':row['id'],"contenu_entier":entier})
        print(f"Dataset includes {len(docs)} docs.")
        return docs,titles,phrases_completes
    
# results=get_docs("../nosdeputes-questions_orales_220620-230723.csv")
# docs = results[0]
# titles_list= results[1]
# phrases_completes=results[2]

def get_docs_filtres(name_template):
    docs=[]
    titles=[]
    phrases_completes=[]
    with open(name_template+"_filtered_file.csv") as f :
        reader = csv.DictReader(f)
        for row in tqdm(reader, total= 11245 ):
            docs.append(row['contenu'])
            titles.append(row['soussection'])
            phrases_completes.append(row['contenu_entier'])
        print(f"Dataset includes {len(docs)} docs.")
        return docs,titles, phrases_completes

#results=get_docs_filtres()
#docs = results[0]
#titles_list= results[1]
#phrases_completes=results[2]


def set_model_parameters(embedding_model):

    # Step 1 - Extract embeddings
    embedding_model = embedding_model

    # Step 2 - Reduce dimensionality
    umap_model = umap.UMAP(angular_rp_forest=True, metric='cosine', n_components=9, n_neighbors=30, min_dist=0.1,random_state=42)

    # Step 3 - Cluster reduced embeddings
    hdbscan_model = HDBSCAN(min_cluster_size=20, min_samples=3, prediction_data=True, metric='euclidean', cluster_selection_method='eom')

    # Step 4 - Tokenize topics
    stoplist=get_stopwords()
    vectorizer_model = CountVectorizer(stop_words=stoplist, ngram_range=(1, 3))

    # Step 5 - Create topic representation
    ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True, bm25_weighting=True)

    # Topic model
    return BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        ctfidf_model=ctfidf_model,
        n_gram_range=(1,3),
        nr_topics='auto'
    )


#embedding_model = SentenceTransformer("dangvantuan/sentence-camembert-large")
#embeddings = embedding_model.encode(docs, show_progress_bar=True)

# with open("data/embeddings_nosdeputes.json","w") as f :
#     json.dump(embeddings.tolist(), f)

# embeddings=[]
# with open("TopicRecognition/data/embeddings_nosdeputes.json") as f :
#     embeddings = np.array(json.load(f))

# topic_model = set_model_parameters(embedding_model)
# topic_model.fit(docs, embeddings)
# print(topic_model.generate_topic_labels(nr_words=5, topic_prefix=True, word_length=None, separator='  --  '))
# print(topic_model.get_topic_info())

labels={
            
            0:"Nul",
            1:"Forces de l'ordre et insécurité",
            2:"Complots",
            3:"Nul",
            4:"Immigration",
            5:"Réchauffement climatique",
            6:"Taxation",
            7:"Inflation",
            8:"Nul",
            9:"Réforme des retraites",
            10:"Guerre Ukraine-Russie, Azerbaïdjan-Arménie",
            11:"Laïcité à l'école",
            12:"Nul",
            13:"Nul",
            14:"Médias",
            15:"Précarité etudiante",
            16:"Vaccination et prévention",
            17:"Incendies, catastrophes naturelles",
            18:"Dettes publiques",
            19:"Répression des femmes iraniennes ",
            20:"Enseignement",
            21:"Culture",
            22:"Procédures administratives",
            23:"Pêcheurs",
            24:"Armement et spatial",
            25:"Service national universel",
            26:"Compétitions sportives internationales",
            27:"Vaccins",
            28:"Rassemblement National",
            29:"Nul",
            30:"Harcélement",
            31:"Nul",
            32:"Inflation",
            33:"Bioéthique",
            34:"Nul",
            35:"Union européenne",
            36:"Protéger les français",
            37:"Outre-mer",
            38:"Nul",
            39:"Nul",
            40:"Nul",
            41:"Uber",
            42:"Paradis fiscaux",
            43:"Esclavage",
            -1:"Nul"
        }
def visualisation(only_questions, use_json,phrases,titles, name_template, topic_model,embeddings,labels):
    
    if not only_questions :
        fig=None
        if labels :
            topic_model.set_topic_labels(labels)
            reduced_embeddings = umap.UMAP(n_neighbors=10, n_components=2, min_dist=0.0, metric='cosine',random_state=42).fit_transform(embeddings)
            fig =topic_model.visualize_documents(titles, reduced_embeddings=reduced_embeddings,topics=[1,2,4,5,6,7,9,10,11,14,15,16,17,18,18,19,20,21,22,23,24,25,26,27,28,30,32,33,35,36,37,41,42,43],custom_labels=True, sample=0.2)
        else :
            reduced_embeddings = umap.UMAP(n_neighbors=10, n_components=2, min_dist=0.0, metric='cosine',random_state=42).fit_transform(embeddings)
            fig =topic_model.visualize_documents(titles, reduced_embeddings=reduced_embeddings, sample=0.2)
        fig.write_html(name_template+"_visualisation_all.html")
    if only_questions :
        embeddings_quest=[]
        titles_quest=[]
        topics_per_doc=[]
        count_questions=Counter()
        if not use_json :
            with open(name_template+"_enonce_questions.csv") as f, open(name_template+"_embeddings.json","w") as wf:

                reader = csv.DictReader(f)
                for row in reader :
                    for i,phrase in enumerate(phrases) :
                        if labels !=None :
                            if row['question'] == phrase and labels[topic_model.topics_[i]]!="Nul":
                                count_questions[topic_model.topics_[i]] +=1
                                embeddings_quest.append(embeddings[i])
                                titles_quest.append(row['soussection'])
                                topics_per_doc.append(topic_model.topics_[i])
                                break
                        else :
                            if row['question'] == phrase :
                                count_questions[topic_model.topics_[i]] +=1
                                embeddings_quest.append(embeddings[i])
                                titles_quest.append(row['soussection'])
                                topics_per_doc.append(topic_model.topics_[i])
                                break
                for topics,count in count_questions.items():
                    if count==1 :
                        idx =topics_per_doc.index(topics)
                        del(topics_per_doc[idx])
                        del(embeddings_quest[idx])
                        del(titles_quest[idx])
                json.dump(np.array(embeddings_quest).tolist(), wf)
            
        #faire un json qui prend l'embedding et le topic correspondant
        else :
            with open(name_template+"_embeddings.json") as f:
                embeddings_quest = np.array(json.load(f))
            with open(name_template+"_enonce_questions.json") as rf :
                reader = csv.DictReader(rf)
                for row in reader :
                    titles_quest.append(row['soussection'])

        
        reduced_embeddings = umap.UMAP(n_neighbors=10, n_components=2, min_dist=0.0, metric='cosine',random_state=42).fit_transform(np.array(embeddings_quest))
        fig=tv.topics_visualisation(titles_quest,topics_per_doc,topics_names_list=topic_model.get_topics(),custom_labels_=labels,reduced_embeddings=reduced_embeddings)
        fig.write_html(name_template+"_visualisation_filtre.html")



# visualisation(True,False,phrases_completes,None, "../enonce_des_questions_orales.csv")

# # visualisation avec tous les points
# visualisation(False,False,phrases_completes,titles_list, "../enonce_des_questions_orales.csv")


# qt.questions_par_topic(phrases_completes,labels,topic_model.topics_)

# today = str(date.today())

# topic_model.save(f'TopicRecognition/data/models/{today}_bertopic_nosdeputes.model')

#PREDICTIONS = 'TopicRecognition/bertopic_results_questions_orales.csv'

def store_results(name_template,topic_model,docs,labels):
    if labels !=None :
        topic_model.set_topic_labels(labels)
        results = topic_model.get_document_info(docs=docs)
        results.to_csv()
        with open(name_template+"_filtered_file.csv") as f, open(name_template+"_bertopic_results.csv", 'w') as of:
            enricher = casanova.enricher(f, of, add=[ "topic", "name", "custom label","top_n_words", "probability", "representative_document"])
            for i, row in enumerate(enricher):
                enricher.writerow(row=row, add=[ results["Topic"][i], results["Name"][i],topic_model.custom_labels_[results["Topic"][i]+topic_model._outliers], results["Top_n_words"][i], results["Probability"][i], results["Representative_document"][i]])
    else :
        results = topic_model.get_document_info(docs=docs)
        results.to_csv()
        with open(name_template+"_filtered_file.csv") as f, open(name_template+"_bertopic_results.csv", 'w') as of:
            enricher = casanova.enricher(f, of, add=[ "topic", "name","top_n_words", "probability", "representative_document"])
            for i, row in enumerate(enricher):
                enricher.writerow(row=row, add=[ results["Topic"][i], results["Name"][i], results["Top_n_words"][i], results["Probability"][i], results["Representative_document"][i]])

#store_results(PREDICTIONS)
