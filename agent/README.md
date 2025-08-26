# Rebuild docker
docker build -t ai-sandbox .

# Run manually outside of dockerconainer
docker run -d --gpus 'device=0' --name vnc-instance-1 --network ai-net \
  -p 5901:5901 \
  -v $(pwd)/tasks:/tasks \
  -e AGENT_NAME="agent-1" \
  -e AGENT_TASK="/tasks/agent-1.yaml" \
  -e RTDETR_API_URL="http://rtdetr-api:8000/predict" \
  ai-sandbox