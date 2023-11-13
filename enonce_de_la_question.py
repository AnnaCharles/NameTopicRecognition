import csv
from collections import defaultdict, Counter
from tqdm import tqdm


def enunciate(data_file,name_template):
    intervenant=defaultdict(Counter)  #{ "soussection" : {"intervenant":nombre d'interventions}}
    interventions_par_ssection=defaultdict(lambda: defaultdict(list))  #{ "soussection" : {"intervenant":[liste de ses interventions]}}

    with open(data_file) as f, open(name_template+"_enonce_questions.csv","w") as wf :
        reader = csv.DictReader(f)
        writer = csv.DictWriter(wf, fieldnames=['soussection','question', 'questionneur'])
        writer.writeheader() 
        for row in tqdm(reader,total=reader.line_num) : 
            if row['intervenant_fonction'] =="" and row['intervenant_nom']!= "":
                intervenant[row['soussection']][row['intervenant_nom']] +=len(row['contenu'])
                interventions_par_ssection[row['soussection']][row['intervenant_nom']].append(row['contenu'])
    

        for ssection in intervenant.keys() :
            questionneur = intervenant[ssection].most_common(1)[0][0]
            interv = interventions_par_ssection[ssection][questionneur]
            interv=sorted(interv, key=len, reverse=True)
            question=interv[0]
        
            writer.writerow({"soussection":ssection, "question":question,"questionneur":questionneur })
