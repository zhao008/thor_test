# Thor Docker 环境 / Thor Docker Environment

[中文](#中文) | [English](#english)

---

## 中文

### 项目说明

本目录包含 Thor 测试环境的 Docker 配置文件，用于快速搭建和部署 Thor 测试环境。

### 文件说明

- `Dockerfile` - Docker 镜像构建文件
- `docker-compose.yml` - Docker Compose 编排配置文件

### 使用方法

#### 构建镜像

```bash
docker-compose build
```

#### 启动容器

```bash
docker-compose up -d
```

#### 进入交互式终端（推荐）

```bash
docker compose run --rm cuda bash
```

#### 停止容器

```bash
docker-compose down
```

---

## English

### Project Description

This directory contains Docker configuration files for the Thor test environment, enabling quick setup and deployment of the Thor testing environment.

### Files

- `Dockerfile` - Docker image build file
- `docker-compose.yml` - Docker Compose orchestration configuration file

### Usage

#### Build Image

```bash
docker-compose build
```

#### Start Container

```bash
docker-compose up -d
```

#### Enter Interactive Terminal (Recommended)

```bash
docker compose run --rm cuda bash
```

#### Stop Container

```bash
docker-compose down
```
