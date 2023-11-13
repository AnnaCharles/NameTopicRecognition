import csv
from tqdm import tqdm
from collections import Counter,defaultdict

def stats(algo,data_file) :
    if algo=="spacy":
        filename=data_file+"NER_spacy_results.csv"
    if algo=="stanza":
        filename=data_file+"NER_stanza_results.csv"
    
    association=defaultdict(list)
    longest=defaultdict(str)
    with open(filename) as file :   
        reader=csv.DictReader(file)
        for row in tqdm(reader,total=17000) :
            ner=row['ner'].split(",")
            for idx, desc in enumerate(row['description'].split(",")):
                if "PER" in desc :    
                    pers= ner[idx].replace("\n", "").strip().lower().split(" ")
                    last_pers=pers[len(pers)-1]

                    association[last_pers].append(ner[idx].replace("\n", "").strip().lower())
                    if len(ner[idx].replace("\n", "").strip().lower().split(" ")) >= len(longest[last_pers].split(" ")) :
                        longest[last_pers]=ner[idx].replace("\n", "").strip().lower()

    with open(data_file+"_"+algo+"_statistiques.csv",'w') as result :
        fieldnames=['last','longest','personnes','count']
        writer=csv.DictWriter(result, fieldnames=fieldnames)
        writer.writeheader()  
        for tok in association.keys() :
            if len(association[tok])>1 and len(longest[tok])>2:

                writer.writerow({"last":tok, "longest":longest[tok],"personnes":association[tok], 'count':len(association[tok]), })
                            
