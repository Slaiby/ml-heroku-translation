docker build -t translation/api-server -f Dockerfile .
docker run -p8000:8000 -it --rm translation/api-server


sudo docker build -t translation/api-server -f Dockerfile . && sudo docker run -p8000:8000 -it --rm translation/api-server