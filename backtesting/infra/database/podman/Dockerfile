FROM postgres:15-alpine

# Install build dependencies
RUN apk add --no-cache --virtual .build-deps \
    gcc \
    g++ \
    make \
    cmake \
    clang \
    llvm \
    git \
    postgresql-dev \
    openssl-dev

# Install TimescaleDB
RUN git clone https://github.com/timescale/timescaledb.git /tmp/timescaledb && \
    cd /tmp/timescaledb && \
    git checkout 2.13.0 && \
    ./bootstrap && \
    cd build && make && make install && \
    cd / && rm -rf /tmp/timescaledb

# Clean up build dependencies
RUN apk del .build-deps

# Add TimescaleDB to shared_preload_libraries
RUN echo "shared_preload_libraries = 'timescaledb'" >> /usr/local/share/postgresql/postgresql.conf.sample

# Create directory for custom config
RUN mkdir -p /docker-entrypoint-initdb.d

# Copy initialization scripts
COPY init-db.sh /docker-entrypoint-initdb.d/
COPY postgresql.conf /etc/postgresql/postgresql.conf

# Set proper permissions
RUN chmod +x /docker-entrypoint-initdb.d/init-db.sh

# Expose PostgreSQL port
EXPOSE 5432

# Set default environment variables
ENV POSTGRES_DB=options_arbitrage
ENV POSTGRES_USER=trading_user
ENV POSTGRES_PASSWORD=secure_trading_password
