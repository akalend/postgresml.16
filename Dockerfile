FROM ubuntu:22.04 AS base
LABEL MAINTAINER Alexandre Kalendarev <akalend@mail.ru>
ENV TZ=UTC
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt update && apt install -y \
    bison \
    bzip2 \
    cpanminus \
    curl \
    flex \
    gcc \
    git \
    libcurl4-gnutls-dev \
    libicu-dev \
    libperl-dev \
    liblz4-dev \
    libpam0g-dev \
    libreadline-dev \
    libssl-dev \
    locales \
    make \
    perl \
    pkg-config \
    python3 \
    python3-pip \
    software-properties-common \
    sudo \
    wget \
    net-tools \
    zlib1g-dev \
 && add-apt-repository ppa:deadsnakes/ppa -y \
 && apt install -y \
    python3.9-full \
 # software properties pulls in pkexec, which makes the debugger unusable in vscode
 && apt purge -y \
    software-properties-common \
 && apt autoremove -y \
 && apt clean

RUN sudo pip3 install pipenv pipenv-shebang

RUN cpanm install IPC::Run

RUN locale-gen en_US.UTF-8

RUN useradd -ms /bin/bash postgres \
 && usermod -aG sudo postgres \
 && echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

WORKDIR /home/postgres
USER postgres

RUN pip3 install catboost pandas && wget https://github.com/catboost/catboost/releases/download/v1.2.7/libcatboostmodel.so && \
     sudo cp libcatboostmodel.so /usr/local/lib

ENV PATH=/usr/bin:/usr/local/bin:/usr/local/pgsql/bin:/home/postgres/.local/bin
ENV LD_LIBRARY_PATH=/usr/local/lib
ENV PGDATA = /usr/local/pgsql/data
ENV PGPORT=5432
RUN echo "export MAKEFLAGS=\"-j \$(nproc)\"" >> "/home/postgres/.bashrc"


RUN git clone --branch rel_16_ML --depth 1 https://github.com/akalend/postgres.ml.git && \
	cd postgres.ml && \
    ./configure --with-python3   && make && sudo make install && \
    cd /usr/local/pgsql && sudo mkdir data && sudo chown postgres data 

RUN install --verbose --directory --owner postgres --group postgres --mode 3777 /var/run/postgresql

ENV PGDATA /var/lib/postgresql/data
# this 1777 will be replaced by 0700 at runtime (allows semi-arbitrary "--user" values)
RUN install --verbose --directory --owner postgres --group postgres --mode 1777 "$PGDATA"
VOLUME /var/lib/postgresql/data

COPY docker-entrypoint.sh docker-ensure-initdb.sh /usr/local/bin/
RUN ln -sT docker-ensure-initdb.sh /usr/local/bin/docker-enforce-initdb.sh
ENTRYPOINT ["docker-entrypoint.sh"]

# We set the default STOPSIGNAL to SIGINT, which corresponds to what PostgreSQL
# calls "Fast Shutdown mode" wherein new connections are disallowed and any
# in-progress transactions are aborted, allowing PostgreSQL to stop cleanly and
# flush tables to disk.
#
# See https://www.postgresql.org/docs/current/server-shutdown.html for more details
# about available PostgreSQL server shutdown signals.
#
# See also https://www.postgresql.org/docs/current/server-start.html for further
# justification of this as the default value, namely that the example (and
# shipped) systemd service files use the "Fast Shutdown mode" for service
# termination.
#
STOPSIGNAL SIGINT

EXPOSE 5432




