from greek_nlp.rag.retriever import Retriever

r = Retriever.load()
for hit in r.search("πλημμύρες και αποζημιώσεις για αγρότες", k=3):
    print(hit["score"], hit["label"], "|", hit["snippet"][:150])
