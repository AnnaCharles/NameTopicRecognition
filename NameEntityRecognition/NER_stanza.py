import stanza
import csv 
from tqdm import tqdm


def ner(data_file):

    stanza.download("fr")
    nlp = stanza.Pipeline('fr') # initialize English neural pipeline

    with open(data_file,'r') as file :
        with open(data_file+"NER_stanza_results.csv","w") as wfile :
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
                        for ent in doc.entities :
                            names+=ent.text +", \n"
                            desc+=ent.type +", \n"
                        writer.writerow({
                            "text":row['contenu'],
                            "size":doc.num_tokens,
                            "ner": names,
                            "description": desc
                        })
                    else :
                        writer.writerow({
                                "text":row['contenu'],
                                "size":doc.num_tokens,
                                "ner": "No name entity found",
                                "description": ""
                            })