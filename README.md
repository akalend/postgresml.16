# Machine learning module in PostgresSQL  
![pg_ml](/doc/pg_ml.png)


This repository is fork from postgres original repository (https://github.com/postgres/postgres)  version 16.4 with ML syntax. More inform in the https://github.com/akalend/postgres.ml/pull/1

For installation we have the libcatboostmodel.so  1.2.7 version.


## Prediction

ML prediction use the [CatBoost] (https://catboost.ai/) based on the categorical boosting algorithm.

The prediction has so far been made  for:
 - binary classification  dataset:(titanic, adult)
 - multi classification   dataset:(astra)
 - regression             dataset:(boston)  
 - ranking                dataset:(MSRank)
 - text classification    dataset:(tomates)

## Installation

 Variable $PG_HOME is the postgres home directory. Default is:  PG_HOME=/usr/local/bin

```
export PG_HOME=/usr/local/pgsql    //where is main postgres folder
export LD_LIBRARY_PATH=/usr/local/lib

wget https://github.com/catboost/catboost/releases/download/v1.2.7/libcatboostmodel.so
cp catboostmodel.so $LD_LIBRARY_PATH/

git clone https://github.com/akalend/postgresml.16.git
cd postgresml.16
./configure --with-python
make && sudo make install

cd $PG_HOME
sudo mkdir data 
sudo chown postgres data
cd bin && sudo pipenv install

sudo -u postgres bash
pipenv shell
pip install catboost pandas

export PATH=$PATH:/usr/local/pgsql/bin
psql -c 'CREATE DATABASE test'
psql -c 'CREATE LANGUAGE python3u' test 
psql -c 'CREATE EXTENSION catboost' test

psql test

```

## SYNTAX

```sql
    CREATE MODEL [REGRESSION | CLASSIFICATION | RANKING] modelname (  options ) FROM tablename;

    options:    TARGET target_column_name |
                IGNORE [ignore_columns_list] |
                LOSS [FUNCTION] loss_function_name |
                EVAL eval_metric_name |
                SPLIT persent_of_test_data |
                GROUP BY grouping_column_name

```

### example create model

```sql
    CREATE MODEL REGRESSION boston ( TARGET rent ) FROM boston;

    CREATE MODEL  titanic ( IGNORED [name, passanger_id, cabin],TARGET res ) FROM titanic;
```


### predict model
```sql
    PREDICT [MODEL] modelname FROM tablename;

```

### example 
```sql
    PREDICT titanic FROM titanic;
```

