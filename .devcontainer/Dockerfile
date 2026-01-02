FROM eclipse-temurin:17.0.15_6-jdk

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends curl gnupg wget netcat-openbsd tar sudo inetutils-ping git-lfs unzip && \
    rm -rf /var/lib/apt/lists/*

RUN echo "deb https://repo.scala-sbt.org/scalasbt/debian all main" | sudo tee /etc/apt/sources.list.d/sbt.list && \
    echo "deb https://repo.scala-sbt.org/scalasbt/debian /" | sudo tee /etc/apt/sources.list.d/sbt_old.list && \
    curl -sL "https://keyserver.ubuntu.com/pks/lookup?op=get&search=0x2EE0EA64E40A89B84B2DF73499E82A75642AC823" | sudo tee /etc/apt/trusted.gpg.d/sbt.asc && \
    sudo apt-get update && \
    sudo apt-get install -y sbt

COPY --from=apache/spark:4.0.0-scala2.13-java17-ubuntu /opt/spark /opt/spark

COPY --from=flink:2.0.0-scala_2.12-java17 /opt/flink /opt/flink

COPY --from=apache/kafka:4.0.0 /opt/kafka /opt/kafka

ENV PATH=$PATH:/opt/spark/bin:/opt/flink/bin:/opt/kafka/bin     

WORKDIR /workspace

