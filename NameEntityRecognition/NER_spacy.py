import spacy 
import csv 
from tqdm import tqdm



def ner(data_file):
    nlp = spacy.load("fr_core_news_sm")
    with open(data_file,'r') as file :
        with open(data_file+"NER_spacy_results.csv","w") as wfile :
            fieldnames=['text','size','ner','description']
            reader=csv.DictReader(file)
            writer=csv.DictWriter(wfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in tqdm(reader,total=27000) :
                if row['intervenant_fonction'] != "présidente" and row['intervenant_nom'] != "":
                    doc=nlp(row['contenu'])
                    names=""
                    desc=""
                    if doc.ents :
                        for ent in doc.ents :
                            names+=ent.text +", \n"
                            desc+=ent.label_ +", \n"
                        writer.writerow({
                            "text":row['contenu'],
                            "size":len(doc),
                            "ner": names,
                            "description": desc
                        })
                    else :
                        writer.writerow({
                                "text":row['contenu'],
                                "size":len(doc),
                                "ner": "No name entity found",
                                "description": ""
                            })