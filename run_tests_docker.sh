#!/usr/bin/env bash
#
# This helper script builds a Docker image with everything required to build and run TVL, then
# runs the entire test suite for all backends.
#
# Requirements: docker, nvidia-docker

# Exit if a command fails.
set -e

# Build Docker image.
docker build -t tvl .

# Start a Docker container to run the tests.
docker run --runtime=nvidia --rm -it tvl make test
