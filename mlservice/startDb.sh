#!/bin/bash

if [ -z "$1" ]; then
	echo "Please input command: start | stop | restart"
        exit
fi

export PG_HOME=/usr/local/pgsql
export PG_BIN=$PG_HOME/bin

export PATH=$PG_BIN:$PATH
cd $PG_HOME

sudo -u postgres $PG_BIN/pg_ctl -D $PG_HOME/data -l /dev/null $1