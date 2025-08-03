# Use official TimescaleDB image - no custom build needed
# This file is kept for reference but docker-compose.yml now uses the official image directly
FROM timescale/timescaledb:latest-pg15

# Copy custom initialization script if needed
COPY init-db.sh /docker-entrypoint-initdb.d/
RUN chmod +x /docker-entrypoint-initdb.d/init-db.sh

# Environment variables are set in docker-compose.yml
