docker build --no-cache -t rtdetr-api:latest .

(detrvenv) theiss@Agent:~/AIAgent/inference$ docker run --rm -it --gpus 'device=1' --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -p 8000:8000 rtdetr-api:latest