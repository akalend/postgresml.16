#!/usr/bin/env bash

sudo mkdir data
# sudo mkdir upload
sudo chown postgres data
sudo chown postgres upload
initdb -D data 
sudo chown postgres *.conf
cp *.conf data
grep listen data/postgres.conf

pg_ctl -D data -l /tmp/log start

psql -c 'CREATE LANGUAGE plpython3u'
psql -c 'CREATE EXTENSION catboost'
cat datasets.dmp | psql 

pg_ctl -D data -l /tmp/log stop
postgres -D data 
