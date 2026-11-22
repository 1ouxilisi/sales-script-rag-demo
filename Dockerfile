# 应用镜像：只跑网页问答服务；Ollama 模型服务独立部署（宿主或独立容器）
FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# 容器内通过 host.docker.internal 访问宿主机的 Ollama；如 Ollama 在独立容器，改为其服务名/IP
ENV OLLAMA_API=http://host.docker.internal:11434
# 容器内监听所有网卡，便于端口映射
ENV HOST=0.0.0.0
ENV PORT=8000

EXPOSE 8000

CMD ["python", "app.py"]
