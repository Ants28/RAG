FROM python:3.10-bookworm

WORKDIR /app


RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    curl \
    libcurl4-openssl-dev \
    && rm -rf /var/lib/apt/lists/*


RUN git clone --depth=1 https://github.com/ggerganov/llama.cpp.git /opt/llama.cpp
WORKDIR /opt/llama.cpp
RUN cmake -B build -DCMAKE_BUILD_TYPE=Release
RUN cmake --build build -j4


RUN cp build/bin/llama-server /usr/local/bin/llama-server


WORKDIR /app


COPY requirements.txt .
RUN pip install --no-cache-dir torch==2.3.1+cpu --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt


COPY . .
COPY models/ /app/models/


EXPOSE 4000


ENV MODEL_PATH=/app/models/llama-2-7b-chat.Q4_K_M.gguf
ENV LLAMA_URL=http://localhost:8080



COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
CMD ["/entrypoint.sh"]

