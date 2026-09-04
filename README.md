docker build --no-cache -t delivery-forecast .
docker run -p 8000:8000 delivery-forecast

# Login to GHCR (one time)
source .env
echo $GITHUB_TOKEN | docker login ghcr.io -u Mo-Anis-24 --password-stdin

# Build
docker build --no-cache -t delivery-forecast .

# Push to GHCR
docker tag delivery-forecast ghcr.io/mo-anis-24/delivery-forecast:latest
docker push ghcr.io/mo-anis-24/delivery-forecast:latest

# Run locally with DagsHub tracking
docker run -p 8000:8000 --env-file .env delivery-forecast