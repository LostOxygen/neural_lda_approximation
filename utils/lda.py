"""helper module to prepare and train the lda model"""
import os
import pickle
import gensim
from gensim.models import LdaMulticore
from tqdm import tqdm

LDA_PATH = "./models/"
DATA_PATH = "./data/"

def train_lda(num_workers: int, num_topics: int, freq_id: int) -> None:
    """helper function to train and save a lda model in a specified path
       :param num_workers: number of workers to compute the lda
       :param num_topics: number of topics for the lda
       :param words_list: the list of words on which the lda should be computed
       :param freq_id: if variable is set the data has '_freq' suffix and the BoW will have changed
                       frequencies for the given word id
       :return: None
    """
    dictionary = gensim.corpora.Dictionary.load_from_text("./data/wikipedia_dump/wiki_wordids.txt")
    bow_list = gensim.corpora.MmCorpus("./data/wikipedia_dump/wiki_bow.mm.bz2")

    if bool(freq_id):
        print(f"[ manipulating bow_list frequencies for word id {freq_id}]")
        bow_list_tmp = []
        # iterate over every document in the model to change the frequencies of certain words
        for index, doc in tqdm(enumerate(bow_list)):
            # create a dictionary for every BoW to access the number of occurrences of every word
            doc_dict = dict(doc)

            found_freq_id = False
            for key, val in doc_dict.items():
                single_doc_bow = []
                if key == freq_id:
                    single_doc_bow.append((int(key), 10.))
                    found_freq_id = True
                else:
                    single_doc_bow.append((int(key), float(val)))
            # if the document doesn't contain the chosen word, add it artificially
            if not found_freq_id:
                single_doc_bow.append((freq_id, 10.))

            bow_list_tmp.append(single_doc_bow)

        # save the modified bow_list
        with open("./data/wikipedia_dump/mod_wiki_bow.pkl", "wb") as out_file:
            pickle.dump(bow_list_tmp, out_file)
        out_file.close()

        bow_list = bow_list_tmp

    print("[ training LDA model. This can take up several hours ..]")
    # create a word corpus
    ldamodel = LdaMulticore(bow_list, num_topics=num_topics, id2word=dictionary,
                            passes=2, workers=num_workers, eval_every=0)
    save_path = LDA_PATH + "freq_lda_model" if bool(freq_id) else LDA_PATH + "lda_model"
    print("[ saving lda model in {} ]".format(save_path))
    if not os.path.isdir(LDA_PATH):
        os.mkdir(LDA_PATH)

    try:
        ldamodel.save(save_path)
    except Exception as exception_lda:
        print("[ could not save the lda model ]")
        print(exception_lda)
