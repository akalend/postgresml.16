#!/usr/bin/env bash

sudo mkdir data 
sudo chown postgres data
initdb -D data 
sudo chown postgres *.conf
mv *.conf data
pg_ctl -D data -l /tmp/log start

psql -c 'CREATE LANGUAGE plpython3u'
psql -c 'CREATE EXTENSION catboost'
cat datasets.dmp | psql 

pg_ctl -D data -l /tmp/log stop
postgres -D data
