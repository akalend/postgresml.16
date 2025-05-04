FROM ubuntu:22.04 AS base
LABEL MAINTAINER Alexandre Kalendarev <akalend@mail.ru>
ENV TZ=UTC
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt update && apt install -y \
    bison \
    bzip2 \
    gzip \
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
    python3 \
    python3-pip \
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
ENV PGHOME = /usr/local/pgsql
ENV PGPORT=5432
RUN echo "export MAKEFLAGS=\"-j \$(nproc)\"" >> "/home/postgres/.bashrc"


RUN git clone  https://github.com/akalend/postgresml.16.git &&\
	cd postgresml.16 && ./configure --with-python && make && sudo make install && \
   cd contrib && git clone https://github.com/akalend/pg_catboost.git && \
   cd pg_catboost && make && sudo make install

WORKDIR /usr/local/pgsql/
COPY docker-ensure.sh docker-ensure.sh
COPY datasets.dmp.gz .
RUN  sudo gzip -d datasets.dmp.gz 

VOLUME /usr/local/pgsql/data
EXPOSE 5432

ENTRYPOINT bash /usr/local/pgsql/docker-ensure.sh 
CMD bash /usr/local/pgsql/docker-ensure.sh