# Cluster-based recommender for OLX

Recommendation systems similar to this paper -
https://pdfs.semanticscholar.org/e107/0c60d926e69298263e9ca36c698b69a21914.pdf
Different feature extraction methods were used: as we have no such data as
attribute values like brand=nike, most frequently n grams were used to
achieve a similar goal. And category-feature mutual information features were
replaced with word2vec approach.
Also, we have no data as items per each query. So, search of queries as a
substring in the database were used to create base clusters

## Project includes:

 - data_reader.py reads data from csv files and store it in PostgreSQL database
 - feature_extractor.py includes two classes for feature extraction.
   - FeatureExtractorMI category-feature mutual information based feature
extractor (not finished)
   - FeatureExtractorW2V word2vec (fasttext) based feature extractor
 - model_clusters.py creates base clusters from queries,
splits clusters with bisect K-means, merges close clusters
 - recommender.py predicts similar items based on cluster model, which is
built on the previous step

## Getting Started

To install postgres and access postgres database:

```sh
sudo apt-get update
sudo apt-get install postgresql postgresql-contrib
sudo -u postgres psql postgres
alter user postgres with password 'postgres';
```
```sh
To install requirements:
```
sudo pip3 install -r requirements.txt

## How to use it

To train clusters model run:

```sh
python3 csv_data_fix.py
```
if necessary (if samples data are corrupted)

```sh
python3 data_reader.py
python3 feature_extractor.py
python3 model_clusters.py
```

Trained clusters (clusters with items ids, cluster hierarchy and clusters
centroids) are located in result_clusters.pickle

To use the trained model run on test data:

```sh
python3 recommender.py
```

or:

```python
recommender = Recommender(top_clusters=5, top_recommendations=5)
recommender.get_recommended_candidates_for_test_data()
```

To use the trained model run on new item (extract features and predict
recommendations) run:

```python
recommender = Recommender(top_clusters=5, top_recommendations=5)
result = recommender.get_recommended_candidates_from_item(
    item_id, test_or_train_set)
```

To extract features on new items most_frequently_n_grams.json,
word_tfidf.pickle and nlp_model.pickle are required. Files will be created
automatically after run feature_extractor.py on train data