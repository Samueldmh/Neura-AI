# ==========================================
# Multi-Stage Dockerfile for Rust Backend
# ==========================================

# 1. Builder Stage
FROM rust:1.80-bullseye AS builder

WORKDIR /app

# Install build dependencies (OpenSSL, pkg-config, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    pkg-config \
    libssl-dev \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy Rust manifest and source code
COPY Cargo.toml ./
COPY src ./src

# Compile binary in release mode with optimizations
RUN cargo build --release

# 2. Minimal Runtime Stage
FROM debian:bullseye-slim

WORKDIR /app

# Install runtime dependencies (OpenSSL runtime & SSL certificates)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libssl1.1 \
    && rm -rf /var/lib/apt/lists/*

# Copy compiled Rust binary from builder stage
COPY --from=builder /app/target/release/neura_ai /app/neura_ai

EXPOSE 8000
ENV PORT=8000

CMD ["/app/neura_ai"]
