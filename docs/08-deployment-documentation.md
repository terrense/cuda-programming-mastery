# 集成部署和文档系统

## 概述

本文档介绍如何构建完整的CUDA学习系统部署方案，包括容器化部署、云端部署、文档系统和持续集成流水线。

## 学习目标

完成本模块后，你将能够：
- 设计和实现容器化部署方案
- 配置多GPU环境的容器
- 创建云端自动化部署脚本
- 构建完整的文档系统
- 实现持续集成和部署流水线

## 容器化部署方案

### Docker镜像构建

#### 基础CUDA开发环境镜像

```dockerfile
# Dockerfile.cuda-dev
FROM nvidia/cuda:12.0-devel-ubuntu22.04

# 设置环境变量
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    python3 \
    python3-pip \
    nodejs \
    npm \
    sqlite3 \
    libsqlite3-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# 安装Python依赖
RUN pip3 install --no-cache-dir \
    numpy \
    matplotlib \
    jupyter \
    flask \
    sqlalchemy \
    pycuda

# 安装Node.js依赖
RUN npm install -g \
    http-server \
    nodemon

# 创建工作目录
WORKDIR /app

# 复制项目文件
COPY . /app/

# 构建项目
RUN mkdir -p build && cd build && \
    cmake .. && \
    make -j$(nproc)

# 设置权限
RUN chmod +x /app/scripts/*.sh

# 暴露端口
EXPOSE 8080 8888

# 启动脚本
CMD ["/app/scripts/start-services.sh"]
```

#### 多GPU训练环境镜像

```dockerfile
# Dockerfile.multi-gpu
FROM nvidia/cuda:12.0-devel-ubuntu22.04

# 安装NCCL
RUN apt-get update && apt-get install -y \
    libnccl2 \
    libnccl-dev \
    && rm -rf /var/lib/apt/lists/*

# 安装MPI
RUN apt-get update && apt-get install -y \
    libopenmpi-dev \
    openmpi-bin \
    && rm -rf /var/lib/apt/lists/*

# 复制多GPU相关代码
COPY multi_gpu/ /app/multi_gpu/
COPY scripts/multi-gpu-setup.sh /app/scripts/

# 构建多GPU组件
RUN cd /app && \
    mkdir -p build/multi_gpu && \
    cd build/multi_gpu && \
    cmake ../../multi_gpu && \
    make -j$(nproc)

# 设置多GPU环境
RUN /app/scripts/multi-gpu-setup.sh

CMD ["/app/scripts/start-multi-gpu-services.sh"]
```

#### 生产环境镜像

```dockerfile
# Dockerfile.production
FROM cuda-learning-system:dev as builder

# 构建生产版本
RUN cd /app/build && \
    cmake -DCMAKE_BUILD_TYPE=Release .. && \
    make -j$(nproc)

# 生产环境基础镜像
FROM nvidia/cuda:12.0-runtime-ubuntu22.04

# 安装运行时依赖
RUN apt-get update && apt-get install -y \
    libsqlite3-0 \
    python3 \
    python3-pip \
    nodejs \
    nginx \
    supervisor \
    && rm -rf /var/lib/apt/lists/*

# 复制构建产物
COPY --from=builder /app/build/bin/ /app/bin/
COPY --from=builder /app/build/lib/ /app/lib/
COPY --from=builder /app/web/ /app/web/
COPY --from=builder /app/docs/ /app/docs/

# 配置Nginx
COPY config/nginx.conf /etc/nginx/nginx.conf
COPY config/supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# 创建数据目录
RUN mkdir -p /app/data /app/logs

# 设置权限
RUN chown -R www-data:www-data /app/web /app/data

# 暴露端口
EXPOSE 80 443

# 启动supervisor
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]
```

### Docker Compose配置

```yaml
# docker-compose.yml
version: '3.8'

services:
  # 开发环境服务
  cuda-dev:
    build:
      context: .
      dockerfile: Dockerfile.cuda-dev
    container_name: cuda-learning-dev
    volumes:
      - ./src:/app/src
      - ./examples:/app/examples
      - ./data:/app/data
      - cuda-cache:/app/.cache
    ports:
      - "8080:8080"
      - "8888:8888"
    environment:
      - CUDA_VISIBLE_DEVICES=0
      - NVIDIA_VISIBLE_DEVICES=all
      - NVIDIA_DRIVER_CAPABILITIES=compute,utility
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    networks:
      - cuda-network

  # 多GPU训练服务
  cuda-multi-gpu:
    build:
      context: .
      dockerfile: Dockerfile.multi-gpu
    container_name: cuda-multi-gpu
    volumes:
      - ./multi_gpu:/app/multi_gpu
      - ./data:/app/data
      - nccl-cache:/app/.nccl
    ports:
      - "8081:8080"
    environment:
      - CUDA_VISIBLE_DEVICES=0,1,2,3
      - NVIDIA_VISIBLE_DEVICES=all
      - NCCL_DEBUG=INFO
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    networks:
      - cuda-network

  # 数据库服务
  database:
    image: postgres:14
    container_name: cuda-learning-db
    environment:
      - POSTGRES_DB=cuda_learning
      - POSTGRES_USER=cuda_user
      - POSTGRES_PASSWORD=cuda_password
    volumes:
      - postgres-data:/var/lib/postgresql/data
      - ./sql/init.sql:/docker-entrypoint-initdb.d/init.sql
    ports:
      - "5432:5432"
    networks:
      - cuda-network

  # Redis缓存服务
  redis:
    image: redis:7-alpine
    container_name: cuda-learning-redis
    volumes:
      - redis-data:/data
    ports:
      - "6379:6379"
    networks:
      - cuda-network

  # 监控服务
  monitoring:
    image: prom/prometheus
    container_name: cuda-monitoring
    volumes:
      - ./config/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus-data:/prometheus
    ports:
      - "9090:9090"
    networks:
      - cuda-network

  # 日志收集
  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:8.5.0
    container_name: cuda-elasticsearch
    environment:
      - discovery.type=single-node
      - "ES_JAVA_OPTS=-Xms512m -Xmx512m"
    volumes:
      - elasticsearch-data:/usr/share/elasticsearch/data
    ports:
      - "9200:9200"
    networks:
      - cuda-network

  # 负载均衡
  nginx:
    image: nginx:alpine
    container_name: cuda-nginx
    volumes:
      - ./config/nginx.conf:/etc/nginx/nginx.conf
      - ./web:/usr/share/nginx/html
    ports:
      - "80:80"
      - "443:443"
    depends_on:
      - cuda-dev
      - cuda-multi-gpu
    networks:
      - cuda-network

volumes:
  cuda-cache:
  nccl-cache:
  postgres-data:
  redis-data:
  prometheus-data:
  elasticsearch-data:

networks:
  cuda-network:
    driver: bridge
```

### 多GPU环境配置

```bash
#!/bin/bash
# scripts/multi-gpu-setup.sh

set -e

echo "配置多GPU环境..."

# 检查GPU数量
GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
echo "检测到 $GPU_COUNT 个GPU"

if [ $GPU_COUNT -lt 2 ]; then
    echo "警告: 检测到少于2个GPU，多GPU功能可能无法正常工作"
fi

# 配置NCCL
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1

# 创建NCCL配置文件
cat > /etc/nccl.conf << EOF
# NCCL Configuration
export NCCL_SOCKET_IFNAME=eth0
export NCCL_DEBUG=INFO
export NCCL_TREE_THRESHOLD=0
EOF

# 设置GPU拓扑
nvidia-smi topo -m > /app/logs/gpu-topology.log

# 配置GPU持久化模式
for i in $(seq 0 $((GPU_COUNT-1))); do
    nvidia-smi -i $i -pm 1
    nvidia-smi -i $i -ac 877,1215  # 设置内存和GPU时钟
done

# 创建多GPU测试脚本
cat > /app/scripts/test-multi-gpu.sh << 'EOF'
#!/bin/bash
echo "测试多GPU通信..."

# 编译测试程序
nvcc -o /tmp/multi_gpu_test /app/examples/multi_gpu/nccl_test.cu -lnccl

# 运行测试
mpirun -np $GPU_COUNT /tmp/multi_gpu_test

echo "多GPU测试完成"
EOF

chmod +x /app/scripts/test-multi-gpu.sh

echo "多GPU环境配置完成"
```

## 云端部署自动化

### Kubernetes部署配置

```yaml
# k8s/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: cuda-learning-system
---
# k8s/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: cuda-learning-config
  namespace: cuda-learning-system
data:
  database_url: "postgresql://cuda_user:cuda_password@postgres:5432/cuda_learning"
  redis_url: "redis://redis:6379"
  log_level: "INFO"
---
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: cuda-learning-app
  namespace: cuda-learning-system
spec:
  replicas: 3
  selector:
    matchLabels:
      app: cuda-learning-app
  template:
    metadata:
      labels:
        app: cuda-learning-app
    spec:
      containers:
      - name: cuda-learning
        image: cuda-learning-system:latest
        ports:
        - containerPort: 8080
        env:
        - name: DATABASE_URL
          valueFrom:
            configMapKeyRef:
              name: cuda-learning-config
              key: database_url
        - name: REDIS_URL
          valueFrom:
            configMapKeyRef:
              name: cuda-learning-config
              key: redis_url
        resources:
          requests:
            nvidia.com/gpu: 1
            memory: "2Gi"
            cpu: "1"
          limits:
            nvidia.com/gpu: 1
            memory: "4Gi"
            cpu: "2"
        volumeMounts:
        - name: data-volume
          mountPath: /app/data
      volumes:
      - name: data-volume
        persistentVolumeClaim:
          claimName: cuda-learning-pvc
---
# k8s/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: cuda-learning-service
  namespace: cuda-learning-system
spec:
  selector:
    app: cuda-learning-app
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8080
  type: LoadBalancer
---
# k8s/ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: cuda-learning-ingress
  namespace: cuda-learning-system
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
spec:
  tls:
  - hosts:
    - cuda-learning.example.com
    secretName: cuda-learning-tls
  rules:
  - host: cuda-learning.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: cuda-learning-service
            port:
              number: 80
```

### Terraform基础设施代码

```hcl
# terraform/main.tf
terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

# VPC配置
resource "aws_vpc" "cuda_learning_vpc" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_hostnames = true
  enable_dns_support   = true

  tags = {
    Name = "cuda-learning-vpc"
  }
}

# 子网配置
resource "aws_subnet" "cuda_learning_subnet" {
  count             = 2
  vpc_id            = aws_vpc.cuda_learning_vpc.id
  cidr_block        = "10.0.${count.index + 1}.0/24"
  availability_zone = data.aws_availability_zones.available.names[count.index]

  map_public_ip_on_launch = true

  tags = {
    Name = "cuda-learning-subnet-${count.index + 1}"
  }
}

# EKS集群
resource "aws_eks_cluster" "cuda_learning_cluster" {
  name     = "cuda-learning-cluster"
  role_arn = aws_iam_role.eks_cluster_role.arn
  version  = "1.24"

  vpc_config {
    subnet_ids = aws_subnet.cuda_learning_subnet[*].id
  }

  depends_on = [
    aws_iam_role_policy_attachment.eks_cluster_policy,
  ]
}

# GPU节点组
resource "aws_eks_node_group" "cuda_learning_gpu_nodes" {
  cluster_name    = aws_eks_cluster.cuda_learning_cluster.name
  node_group_name = "cuda-learning-gpu-nodes"
  node_role_arn   = aws_iam_role.eks_node_role.arn
  subnet_ids      = aws_subnet.cuda_learning_subnet[*].id

  instance_types = ["p3.2xlarge", "p3.8xlarge"]
  ami_type       = "AL2_x86_64_GPU"

  scaling_config {
    desired_size = 2
    max_size     = 10
    min_size     = 1
  }

  update_config {
    max_unavailable = 1
  }

  depends_on = [
    aws_iam_role_policy_attachment.eks_worker_node_policy,
    aws_iam_role_policy_attachment.eks_cni_policy,
    aws_iam_role_policy_attachment.eks_container_registry_policy,
  ]
}

# RDS数据库
resource "aws_db_instance" "cuda_learning_db" {
  identifier = "cuda-learning-db"

  engine         = "postgres"
  engine_version = "14.6"
  instance_class = "db.t3.micro"

  allocated_storage     = 20
  max_allocated_storage = 100
  storage_type          = "gp2"
  storage_encrypted     = true

  db_name  = "cuda_learning"
  username = "cuda_user"
  password = var.db_password

  vpc_security_group_ids = [aws_security_group.rds_sg.id]
  db_subnet_group_name   = aws_db_subnet_group.cuda_learning_db_subnet_group.name

  backup_retention_period = 7
  backup_window          = "03:00-04:00"
  maintenance_window     = "sun:04:00-sun:05:00"

  skip_final_snapshot = true

  tags = {
    Name = "cuda-learning-db"
  }
}

# ElastiCache Redis
resource "aws_elasticache_subnet_group" "cuda_learning_cache_subnet" {
  name       = "cuda-learning-cache-subnet"
  subnet_ids = aws_subnet.cuda_learning_subnet[*].id
}

resource "aws_elasticache_cluster" "cuda_learning_redis" {
  cluster_id           = "cuda-learning-redis"
  engine               = "redis"
  node_type            = "cache.t3.micro"
  num_cache_nodes      = 1
  parameter_group_name = "default.redis7"
  port                 = 6379
  subnet_group_name    = aws_elasticache_subnet_group.cuda_learning_cache_subnet.name
  security_group_ids   = [aws_security_group.redis_sg.id]
}

# S3存储桶
resource "aws_s3_bucket" "cuda_learning_storage" {
  bucket = "cuda-learning-storage-${random_string.bucket_suffix.result}"
}

resource "aws_s3_bucket_versioning" "cuda_learning_storage_versioning" {
  bucket = aws_s3_bucket.cuda_learning_storage.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "random_string" "bucket_suffix" {
  length  = 8
  special = false
  upper   = false
}
```

### 自动化部署脚本

```bash
#!/bin/bash
# scripts/deploy.sh

set -e

# 配置变量
ENVIRONMENT=${1:-development}
AWS_REGION=${AWS_REGION:-us-west-2}
CLUSTER_NAME="cuda-learning-cluster"
NAMESPACE="cuda-learning-system"

echo "开始部署CUDA学习系统到 $ENVIRONMENT 环境..."

# 检查依赖
check_dependencies() {
    echo "检查部署依赖..."

    command -v docker >/dev/null 2>&1 || { echo "Docker未安装"; exit 1; }
    command -v kubectl >/dev/null 2>&1 || { echo "kubectl未安装"; exit 1; }
    command -v terraform >/dev/null 2>&1 || { echo "Terraform未安装"; exit 1; }
    command -v aws >/dev/null 2>&1 || { echo "AWS CLI未安装"; exit 1; }

    echo "依赖检查完成"
}

# 构建Docker镜像
build_images() {
    echo "构建Docker镜像..."

    # 构建开发环境镜像
    docker build -t cuda-learning-system:dev -f Dockerfile.cuda-dev .

    # 构建多GPU镜像
    docker build -t cuda-learning-system:multi-gpu -f Dockerfile.multi-gpu .

    # 构建生产环境镜像
    docker build -t cuda-learning-system:latest -f Dockerfile.production .

    echo "Docker镜像构建完成"
}

# 推送镜像到ECR
push_to_ecr() {
    echo "推送镜像到ECR..."

    # 获取ECR登录令牌
    aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $ECR_REGISTRY

    # 标记并推送镜像
    docker tag cuda-learning-system:latest $ECR_REGISTRY/cuda-learning-system:latest
    docker push $ECR_REGISTRY/cuda-learning-system:latest

    docker tag cuda-learning-system:multi-gpu $ECR_REGISTRY/cuda-learning-system:multi-gpu
    docker push $ECR_REGISTRY/cuda-learning-system:multi-gpu

    echo "镜像推送完成"
}

# 部署基础设施
deploy_infrastructure() {
    echo "部署基础设施..."

    cd terraform

    # 初始化Terraform
    terraform init

    # 规划部署
    terraform plan -var="environment=$ENVIRONMENT" -out=tfplan

    # 应用部署
    terraform apply tfplan

    # 获取输出
    CLUSTER_ENDPOINT=$(terraform output -raw cluster_endpoint)
    DB_ENDPOINT=$(terraform output -raw db_endpoint)
    REDIS_ENDPOINT=$(terraform output -raw redis_endpoint)

    cd ..

    echo "基础设施部署完成"
}

# 配置kubectl
configure_kubectl() {
    echo "配置kubectl..."

    aws eks update-kubeconfig --region $AWS_REGION --name $CLUSTER_NAME

    # 验证连接
    kubectl cluster-info

    echo "kubectl配置完成"
}

# 部署应用到Kubernetes
deploy_to_k8s() {
    echo "部署应用到Kubernetes..."

    # 创建命名空间
    kubectl apply -f k8s/namespace.yaml

    # 部署配置
    kubectl apply -f k8s/configmap.yaml

    # 部署应用
    kubectl apply -f k8s/deployment.yaml
    kubectl apply -f k8s/service.yaml
    kubectl apply -f k8s/ingress.yaml

    # 等待部署完成
    kubectl rollout status deployment/cuda-learning-app -n $NAMESPACE

    echo "应用部署完成"
}

# 运行健康检查
health_check() {
    echo "运行健康检查..."

    # 检查Pod状态
    kubectl get pods -n $NAMESPACE

    # 检查服务状态
    kubectl get services -n $NAMESPACE

    # 获取外部IP
    EXTERNAL_IP=$(kubectl get service cuda-learning-service -n $NAMESPACE -o jsonpath='{.status.loadBalancer.ingress[0].hostname}')

    if [ -n "$EXTERNAL_IP" ]; then
        echo "应用可通过以下地址访问: http://$EXTERNAL_IP"

        # 测试API端点
        curl -f http://$EXTERNAL_IP/api/health || echo "健康检查失败"
    else
        echo "等待外部IP分配..."
    fi

    echo "健康检查完成"
}

# 设置监控
setup_monitoring() {
    echo "设置监控..."

    # 部署Prometheus
    kubectl apply -f k8s/monitoring/prometheus.yaml

    # 部署Grafana
    kubectl apply -f k8s/monitoring/grafana.yaml

    # 部署GPU监控
    kubectl apply -f k8s/monitoring/gpu-metrics.yaml

    echo "监控设置完成"
}

# 主部署流程
main() {
    check_dependencies

    if [ "$ENVIRONMENT" = "production" ]; then
        build_images
        push_to_ecr
        deploy_infrastructure
        configure_kubectl
        deploy_to_k8s
        setup_monitoring
        health_check
    else
        echo "启动开发环境..."
        docker-compose up -d
        echo "开发环境启动完成，访问 http://localhost:8080"
    fi

    echo "部署完成！"
}

# 执行主流程
main
```

## 完整文档系统

### API文档生成

```cpp
// docs/api_documentation.h
#pragma once
#include <string>
#include <vector>
#include <map>

struct APIEndpoint {
    std::string method;
    std::string path;
    std::string description;
    std::vector<std::string> parameters;
    std::string request_example;
    std::string response_example;
    std::vector<std::string> error_codes;
};

class APIDocumentationGenerator {
private:
    std::vector<APIEndpoint> endpoints_;
    std::string base_url_;

public:
    APIDocumentationGenerator(const std::string& base_url);

    // 添加API端点
    void addEndpoint(const APIEndpoint& endpoint);

    // 生成OpenAPI规范
    std::string generateOpenAPISpec();

    // 生成HTML文档
    std::string generateHTMLDocs();

    // 生成Markdown文档
    std::string generateMarkdownDocs();

    // 生成Postman集合
    std::string generatePostmanCollection();

private:
    std::string formatEndpointHTML(const APIEndpoint& endpoint);
    std::string formatEndpointMarkdown(const APIEndpoint& endpoint);
};

// api_documentation.cpp
void APIDocumentationGenerator::addEndpoint(const APIEndpoint& endpoint) {
    endpoints_.push_back(endpoint);
}

std::string APIDocumentationGenerator::generateOpenAPISpec() {
    std::ostringstream oss;

    oss << R"({
  "openapi": "3.0.0",
  "info": {
    "title": "CUDA Learning System API",
    "version": "1.0.0",
    "description": "CUDA编程学习系统API文档"
  },
  "servers": [
    {
      "url": ")" << base_url_ << R"(",
      "description": "CUDA学习系统服务器"
    }
  ],
  "paths": {)";

    bool first = true;
    for (const auto& endpoint : endpoints_) {
        if (!first) oss << ",";
        first = false;

        oss << R"(
    ")" << endpoint.path << R"(": {
      ")" << endpoint.method << R"(": {
        "summary": ")" << endpoint.description << R"(",
        "responses": {
          "200": {
            "description": "成功",
            "content": {
              "application/json": {
                "example": )" << endpoint.response_example << R"(
              }
            }
          }
        }
      }
    })";
    }

    oss << R"(
  }
})";

    return oss.str();
}

std::string APIDocumentationGenerator::generateHTMLDocs() {
    std::ostringstream oss;

    oss << R"(<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <title>CUDA学习系统 API文档</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .endpoint { border: 1px solid #ddd; margin: 20px 0; padding: 15px; border-radius: 5px; }
        .method { display: inline-block; padding: 5px 10px; border-radius: 3px; color: white; font-weight: bold; }
        .get { background-color: #61affe; }
        .post { background-color: #49cc90; }
        .put { background-color: #fca130; }
        .delete { background-color: #f93e3e; }
        .code { background-color: #f5f5f5; padding: 10px; border-radius: 3px; overflow-x: auto; }
    </style>
</head>
<body>
    <h1>CUDA学习系统 API文档</h1>
    <p>基础URL: <code>)" << base_url_ << R"(</code></p>
)";

    for (const auto& endpoint : endpoints_) {
        oss << formatEndpointHTML(endpoint);
    }

    oss << R"(
</body>
</html>)";

    return oss.str();
}
```

### 交互式教程生成

```python
# scripts/generate_tutorials.py
import os
import json
import markdown
from jinja2 import Template

class TutorialGenerator:
    def __init__(self, source_dir, output_dir):
        self.source_dir = source_dir
        self.output_dir = output_dir
        self.tutorials = []

    def load_tutorials(self):
        """加载教程配置"""
        config_file = os.path.join(self.source_dir, 'tutorials.json')
        with open(config_file, 'r', encoding='utf-8') as f:
            self.tutorials = json.load(f)

    def generate_interactive_tutorial(self, tutorial_config):
        """生成交互式教程"""
        template_str = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <title>{{ title }}</title>
    <link rel="stylesheet" href="../css/tutorial.css">
    <script src="../js/monaco-editor.min.js"></script>
    <script src="../js/tutorial-engine.js"></script>
</head>
<body>
    <div class="tutorial-container">
        <nav class="tutorial-nav">
            <h2>{{ title }}</h2>
            <ul class="step-list">
                {% for step in steps %}
                <li class="step-item" data-step="{{ loop.index0 }}">
                    <span class="step-number">{{ loop.index }}</span>
                    <span class="step-title">{{ step.title }}</span>
                </li>
                {% endfor %}
            </ul>
        </nav>

        <main class="tutorial-content">
            {% for step in steps %}
            <div class="tutorial-step" id="step-{{ loop.index0 }}"
                 style="display: {% if loop.first %}block{% else %}none{% endif %}">
                <h3>{{ step.title }}</h3>
                <div class="step-description">
                    {{ step.description | markdown }}
                </div>

                {% if step.code_example %}
                <div class="code-section">
                    <h4>代码示例</h4>
                    <div class="code-editor" id="editor-{{ loop.index0 }}"></div>
                    <div class="code-controls">
                        <button class="run-btn" onclick="runCode({{ loop.index0 }})">运行代码</button>
                        <button class="reset-btn" onclick="resetCode({{ loop.index0 }})">重置</button>
                    </div>
                    <div class="code-output" id="output-{{ loop.index0 }}"></div>
                </div>
                {% endif %}

                {% if step.quiz %}
                <div class="quiz-section">
                    <h4>小测验</h4>
                    <div class="quiz-question">{{ step.quiz.question }}</div>
                    <div class="quiz-options">
                        {% for option in step.quiz.options %}
                        <label class="quiz-option">
                            <input type="radio" name="quiz-{{ loop.index0 }}" value="{{ loop.index0 }}">
                            {{ option }}
                        </label>
                        {% endfor %}
                    </div>
                    <button class="check-answer-btn" onclick="checkAnswer({{ loop.index0 }})">检查答案</button>
                </div>
                {% endif %}

                <div class="step-navigation">
                    {% if not loop.first %}
                    <button class="prev-btn" onclick="previousStep()">上一步</button>
                    {% endif %}
                    {% if not loop.last %}
                    <button class="next-btn" onclick="nextStep()">下一步</button>
                    {% else %}
                    <button class="complete-btn" onclick="completeTutorial()">完成教程</button>
                    {% endif %}
                </div>
            </div>
            {% endfor %}
        </main>
    </div>

    <script>
        const tutorialData = {{ tutorial_config | tojson }};
        initializeTutorial(tutorialData);
    </script>
</body>
</html>
        """

        template = Template(template_str)
        html_content = template.render(
            title=tutorial_config['title'],
            steps=tutorial_config['steps'],
            tutorial_config=tutorial_config
        )

        output_file = os.path.join(self.output_dir, f"{tutorial_config['id']}.html")
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)

    def generate_all_tutorials(self):
        """生成所有教程"""
        os.makedirs(self.output_dir, exist_ok=True)

        for tutorial in self.tutorials:
            self.generate_interactive_tutorial(tutorial)
            print(f"生成教程: {tutorial['title']}")

    def generate_tutorial_index(self):
        """生成教程索引页面"""
        index_template = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <title>CUDA编程教程</title>
    <link rel="stylesheet" href="css/index.css">
</head>
<body>
    <div class="container">
        <header>
            <h1>CUDA编程交互式教程</h1>
            <p>通过实践学习CUDA编程</p>
        </header>

        <main class="tutorial-grid">
            {% for tutorial in tutorials %}
            <div class="tutorial-card">
                <h3>{{ tutorial.title }}</h3>
                <p>{{ tutorial.description }}</p>
                <div class="tutorial-meta">
                    <span class="difficulty">难度: {{ tutorial.difficulty }}</span>
                    <span class="duration">时长: {{ tutorial.estimated_time }}</span>
                </div>
                <a href="{{ tutorial.id }}.html" class="start-btn">开始学习</a>
            </div>
            {% endfor %}
        </main>
    </div>
</body>
</html>
        """

        template = Template(index_template)
        html_content = template.render(tutorials=self.tutorials)

        index_file = os.path.join(self.output_dir, 'index.html')
        with open(index_file, 'w', encoding='utf-8') as f:
            f.write(html_content)

if __name__ == "__main__":
    generator = TutorialGenerator('tutorials', 'docs/tutorials')
    generator.load_tutorials()
    generator.generate_all_tutorials()
    generator.generate_tutorial_index()
    print("教程生成完成！")
```

### 文档自动化构建

```yaml
# .github/workflows/docs.yml
name: 构建和部署文档

on:
  push:
    branches: [ main, develop ]
    paths:
      - 'docs/**'
      - 'tutorials/**'
      - 'examples/**'
  pull_request:
    branches: [ main ]

jobs:
  build-docs:
    runs-on: ubuntu-latest

    steps:
    - name: 检出代码
      uses: actions/checkout@v3

    - name: 设置Python环境
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'

    - name: 安装依赖
      run: |
        pip install -r docs/requirements.txt

    - name: 生成API文档
      run: |
        python scripts/generate_api_docs.py

    - name: 生成交互式教程
      run: |
        python scripts/generate_tutorials.py

    - name: 构建Sphinx文档
      run: |
        cd docs
        make html

    - name: 构建Doxygen文档
      run: |
        doxygen Doxyfile

    - name: 合并文档
      run: |
        mkdir -p dist/docs
        cp -r docs/_build/html/* dist/docs/
        cp -r docs/tutorials dist/docs/
        cp -r docs/api dist/docs/

    - name: 部署到GitHub Pages
      if: github.ref == 'refs/heads/main'
      uses: peaceiris/actions-gh-pages@v3
      with:
        github_token: ${{ secrets.GITHUB_TOKEN }}
        publish_dir: ./dist/docs

    - name: 上传文档构建产物
      uses: actions/upload-artifact@v3
      with:
        name: documentation
        path: dist/docs
```

## 持续集成和部署

### CI/CD流水线

```yaml
# .github/workflows/ci-cd.yml
name: CI/CD流水线

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
    - name: 检出代码
      uses: actions/checkout@v3

    - name: 设置CUDA环境
      uses: Jimver/cuda-toolkit@v0.2.11
      with:
        cuda: '12.0'

    - name: 编译项目
      run: |
        mkdir build && cd build
        cmake -DCMAKE_BUILD_TYPE=Release ..
        make -j$(nproc)

    - name: 运行单元测试
      run: |
        cd build
        ctest --output-on-failure

    - name: 运行集成测试
      run: |
        ./scripts/run_integration_tests.sh

    - name: 代码覆盖率分析
      run: |
        gcov build/src/*.o
        lcov --capture --directory . --output-file coverage.info

    - name: 上传覆盖率报告
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.info

  build-and-push:
    needs: test
    runs-on: ubuntu-latest
    if: github.event_name != 'pull_request'

    steps:
    - name: 检出代码
      uses: actions/checkout@v3

    - name: 设置Docker Buildx
      uses: docker/setup-buildx-action@v2

    - name: 登录容器注册表
      uses: docker/login-action@v2
      with:
        registry: ${{ env.REGISTRY }}
        username: ${{ github.actor }}
        password: ${{ secrets.GITHUB_TOKEN }}

    - name: 提取元数据
      id: meta
      uses: docker/metadata-action@v4
      with:
        images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
        tags: |
          type=ref,event=branch
          type=ref,event=pr
          type=sha

    - name: 构建并推送Docker镜像
      uses: docker/build-push-action@v4
      with:
        context: .
        file: ./Dockerfile.production
        push: true
        tags: ${{ steps.meta.outputs.tags }}
        labels: ${{ steps.meta.outputs.labels }}
        cache-from: type=gha
        cache-to: type=gha,mode=max

  deploy-staging:
    needs: build-and-push
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/develop'
    environment: staging

    steps:
    - name: 检出代码
      uses: actions/checkout@v3

    - name: 配置AWS凭证
      uses: aws-actions/configure-aws-credentials@v2
      with:
        aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
        aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
        aws-region: us-west-2

    - name: 部署到测试环境
      run: |
        ./scripts/deploy.sh staging

    - name: 运行烟雾测试
      run: |
        ./scripts/smoke_tests.sh staging

  deploy-production:
    needs: build-and-push
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    environment: production

    steps:
    - name: 检出代码
      uses: actions/checkout@v3

    - name: 配置AWS凭证
      uses: aws-actions/configure-aws-credentials@v2
      with:
        aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
        aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
        aws-region: us-west-2

    - name: 部署到生产环境
      run: |
        ./scripts/deploy.sh production

    - name: 运行生产环境测试
      run: |
        ./scripts/production_tests.sh

    - name: 发送部署通知
      uses: 8398a7/action-slack@v3
      with:
        status: ${{ job.status }}
        channel: '#deployments'
        webhook_url: ${{ secrets.SLACK_WEBHOOK }}
```

## 实践练习

### 练习1：容器化部署
创建完整的Docker容器化方案，支持多GPU环境。

### 练习2：Kubernetes集群
在Kubernetes上部署CUDA学习系统，实现自动扩缩容。

### 练习3：CI/CD流水线
构建完整的持续集成和部署流水线。

### 练习4：文档自动化
实现文档的自动生成和部署系统。

## 总结

本模块介绍了CUDA学习系统的完整部署方案，包括：
- 容器化部署和多GPU环境配置
- 云端基础设施自动化部署
- 完整的文档系统构建
- 持续集成和部署流水线
- 监控和日志收集系统

通过这个完整的部署和文档系统，可以实现CUDA学习平台的规模化运营和维护。

---

**恭喜！** 你已经完成了整个CUDA编程精通学习系统的学习。现在你具备了从基础环境搭建到高级优化技术，从自定义算子开发到完整系统部署的全面能力。
