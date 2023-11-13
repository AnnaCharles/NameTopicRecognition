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
import json 

def get_docs(DATAFILE):
    nlp = spacy.load("fr_core_news_sm")
    docs=[]
    titles=[]
    count=0
    with open(DATAFILE) as f, open("filtered_file_sample.csv", 'w') as of :
        reader = csv.DictReader(f)
        enricher = csv.DictWriter(of, fieldnames=reader.fieldnames)
        enricher.writeheader() 
        for i,row in enumerate(reader) :
            cell= row['contenu']
            if row["intervenant_fonction"] != "présidente" and row["intervenant_nom"] != "":
                doc=nlp(cell)
                if len(doc) > 10 :
                    while len(doc) >=300 :
                        cell=cell[:cell.rfind(".")]
                        doc=nlp(cell)
                    
                    docs.append(cell)
                    #titles.append(row['soussection'])
                    titles.append(count)
                    count+=1
                    enricher.writerow(row)
        print(f"Dataset includes {len(docs)} docs.")
        return docs,titles

docs = get_docs("sample_20questions.csv")[0]
titles= get_docs("sample_20questions.csv")[1]
with open("data/titles/titles_sample.csv","w") as f :
    enricher = csv.DictWriter(f, fieldnames=['titles'])
    enricher.writerow({'titles':titles})



nltk.download('stopwords')
stoplist = stopwords.words("french")
ADDITIONAL_STOPWORDS = ["plus", "chaque", "tout", "tous", "toutes", "toute", "leur", "leurs", "comme", "afin", "pendant", "lorsque","tant"]

def set_model_parameters(embedding_model):

    # Step 1 - Extract embeddings
    embedding_model = embedding_model

    # Step 2 - Reduce dimensionality
    umap_model = umap.UMAP(angular_rp_forest=True, metric='cosine', n_components=9, n_neighbors=30, min_dist=0.1)

    # Step 3 - Cluster reduced embeddings
    hdbscan_model = HDBSCAN(min_cluster_size=13, min_samples=3, prediction_data=True, metric='euclidean', cluster_selection_method='eom')

    # Step 4 - Tokenize topics
    stoplist.extend(ADDITIONAL_STOPWORDS)
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


embedding_model = SentenceTransformer("dangvantuan/sentence-camembert-large")
embeddings = embedding_model.encode(docs, show_progress_bar=True)

with open("data/embeddings_sample.json","w") as f :
    json.dump(embeddings.tolist(), f)

topic_model = set_model_parameters(embedding_model)
topic_model.fit(docs, embeddings)
print(topic_model.generate_topic_labels(nr_words=5, topic_prefix=True, word_length=None, separator='  --  '))
print(topic_model.get_topic_info())

reduced_embeddings = umap.UMAP(n_neighbors=10, n_components=2, min_dist=0.0, metric='cosine').fit_transform(embeddings)
fig =topic_model.visualize_documents(titles, reduced_embeddings=reduced_embeddings)
fig.write_html("data/fig/bertopic_sample_questions.html")
today = str(date.today())

topic_model.save(f'data/models/{today}_bertopic.model',save_embedding_model=True)

PREDICTIONS_FILE = 'bertopic_topics_sample.csv'

results = topic_model.get_document_info(docs=docs)
results.to_csv()
with open("filtered_file_sample.csv") as f, open(PREDICTIONS_FILE, 'w') as of:
    enricher = casanova.enricher(f, of, add=[ "topic", "name", "top_n_words", "probability", "representative_document"])
    for i, row in enumerate(enricher):
        enricher.writerow(row=row, add=[ results["Topic"][i], results["Name"][i], results["Top_n_words"][i], results["Probability"][i], results["Representative_document"][i]])

