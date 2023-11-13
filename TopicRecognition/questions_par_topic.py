import csv 
from collections import defaultdict,Counter

def questions_par_topic(liste_des_phrases_completes, topics_labels,topic_de_chaque_phrase,name_template,use_labels) : # topic_de_chaque_phrase=topic_model.topics_

    q_t=defaultdict(list) # { "topic": [question1, question2 ...]}
    nbre_de_questions=Counter()
    with open(name_template+"_enonce_questions.csv") as f, open(name_template+"quest_par_topic.csv","w") as wf:
                reader = csv.DictReader(f)
                writer = csv.DictWriter(wf, fieldnames=["Topic","questions","nombre de questions"])
                writer.writeheader() 
                for row in reader :
                    for i,phrase in enumerate(liste_des_phrases_completes) :
                        if use_labels :
                            if row['question'] == phrase and topics_labels[topic_de_chaque_phrase[i]]!="Nul":
                                q_t[topics_labels[topic_de_chaque_phrase[i]]].append(row["soussection"])
                                nbre_de_questions[topics_labels[topic_de_chaque_phrase[i]]]+=1
                                break
                        else :
                            if row['question'] == phrase :
                                q_t[topics_labels[int(topic_de_chaque_phrase[i])+1]].append(row["soussection"])
                                nbre_de_questions[topics_labels[int(topic_de_chaque_phrase[i])+1]]+=1
                                break
                
                for topic, questions in q_t.items():
                    quest=""
                    for q in questions :
                        quest += q +"\n"

                    writer.writerow({"Topic":topic,"questions":quest,"nombre de questions":nbre_de_questions[topic]})