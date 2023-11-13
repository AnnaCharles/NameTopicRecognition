import csv
from collections import defaultdict, Counter
from tqdm import tqdm

def classification(name_template) :
    ssections_topics=defaultdict(Counter)

    with open(name_template+"_bertopic_results.csv") as f :
        reader = csv.DictReader(f)
        for row in tqdm(reader, total= 11245): 
            soussection = row['soussection']
            topic = row['name']
        
            ssections_topics[soussection][topic]+=1

    with open(name_template+"_topics_par_question.csv","w") as f :
        writer = csv.DictWriter(f, fieldnames=["soussection","topics","occurence"])
        writer.writeheader() 
        for ssection in tqdm(ssections_topics.keys()):
            topics=""
            occurence=""
            for topic in ssections_topics[ssection].keys() :
                topics+=topic + "\n"
                occurence+=str(ssections_topics[ssection][topic]) + "\n"

            writer.writerow({"soussection":ssection, "topics":topics, "occurence":occurence})


## statistiques sur le corpus 
# ssections_phrases=Counter()
# ssections_intervenant=defaultdict(Counter)

# with open("filtered_file_nosdeputes.csv") as f :
#     reader = csv.DictReader(f)
#     for row in tqdm(reader, total= 11245): 
#         soussection = row['soussection']
#         ssections_phrases[soussection]+=1
#         interv=row['intervenant_nom']
#         ssections_intervenant[soussection][interv]+=1

# with open("phrases_par_questions_filtrées.csv","w") as f :
#     writer = csv.DictWriter(f, fieldnames=["questions","nbre de phrases","nbre d'intervenants"])
#     writer.writeheader() 
#     for ssection in tqdm(ssections_phrases.keys()):
#         writer.writerow({"questions":ssection, "nbre de phrases":ssections_phrases[ssection],"nbre d'intervenants":str(len(ssections_intervenant[ssection]))})
    
#     print("nbre de questions "+ str(len(ssections_phrases)))