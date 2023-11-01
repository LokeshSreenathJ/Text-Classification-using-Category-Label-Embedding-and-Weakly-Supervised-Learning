Step 1: Download the dataset from https://drive.google.com/file/d/1vfqzgDFMZyn1mHlzFx-t1_KuAiNgC64f/view. (News and Movies)
	1.1 Adopt the Auto Phrase (source: https://github.com/shangjingbo1226/AutoPhrase) to extract high quality phrases in both the datasets. Then use segmentation model to parse the same corpus recommended parameters for segmentation is  HIGHLIGHT_MULTI=0.7 HIGHLIGHT_SINGLE=1.0. Here remove all UPPER-CASE letters with lower case. Also join the most relevant words (extracted using Auto Phrase) with "_", so that they are treated as single word and represents the background context precisely.  
Code for text processing			
###import argparse  (source: https://github.com/shangjingbo1226/AutoPhrase)
import os
import csv
import numpy as np
from tqdm import tqdm
from nltk import word_tokenize
import re

def phrase_process():
	f = open(os.path.join('AutoPhrase', 'model_News', "DBLP" , 'segmentation.txt'))
	g = open(args.out_file, 'w')
	for line in tqdm(f):
		doc = ''
		temp = re.split(r'<phrase_Q=\d\.\d+>', line)
		for seg in temp:
			temp2 = seg.split('</phrase>')
			if len(temp2) > 1:
				doc += ("_").join(temp2[0].split(" ")) + temp2[1]
			else:
				doc += temp2[0]
		g.write(doc.strip()+'\n')
	print("Phrase segmented corpus written to {}".format(args.out_file))
	return
def preprocess():
	f = open(os.path.join("AutoPhrase","model_Movies/DBLP","text.txt"))
	docs = f.readlines()
	f_out = open(args.out_file, 'w')
	for doc in tqdm(docs):
		f_out.write(' '.join([w.lower() for w in word_tokenize(doc.strip())]) + '\n')
	return


if __name__=="__main__":

	parser = argparse.ArgumentParser(description='main', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--mode', type=int)
	parser.add_argument('--dataset', default="News")
	parser.add_argument('--in_file', default='segmentation.txt')
	parser.add_argument('--out_file', default='./AutoPhrase/model_News/DBLP/text.txt')
	args, _ = parser.parse_known_args()


phrase_process()
preprocess()
###

Now dev_set : First 100 documents in every dataset, along with newly added prompt.
(Movies: https://drive.google.com/file/d/1--fTTd_qlAE0R-ZpFWRUk9cOxTCWBc_W/view?usp=drive_link)
(News: https://drive.google.com/file/d/1-1Zr00xO3p6Hxav2C7NNUBOBv9jQnvIV/view?usp=drive_link)


Step 2: Setting Up the Environment
Install the necessary packages:
!pip install transformers


Step 3: Initializing the Zero-Shot Classifier
Leverage the multi NLI model from transformers to generate pseudo labels:

from transformers import pipeline
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")


Step 4: Generating Pseudo Labels
Run the pseudo_labels.ipynb script to produce the desired labels for the dataset. (It approximately takes 3 hrs to get the pseudo labels for 2000 documents, so have processed it batch-wise and concatenate the results locally)
Generated 2999 pseudo labels for Movies and 6000 pseudo labels for News.


Step 5: Embedding and Model Training
Utilized the CatE embeddings (Movies: https://drive.google.com/file/d/1-02FptOEHnrXzFon0GNlTlbivgR526Xa/view?usp=sharing, News: https://drive.google.com/file/d/10TmQpT8KpUznrWD0Y1A0DSzfUL78OUff/view?usp=sharing) that are generated using the https://github.com/yumeng5/CatE are used for converting the words into 100 dimensional vectors. Built a GradientBoostingClassifier model.
See the TX_file.ipynb file for step by step code implementation. While executing the above git-hub codes some of the scikit-learn packages are outdated so kindly refer to this stack overflow post which helps in de-bugging the errors,https://stackoverflow.com/questions/72572969/problems-using-spherecluster-package-for-spherical-k-mean-clustering

Step 6: Check for Data Imbalance and Do Hyperparameter tuning. Once the model is finalized, predict the test results.
