# Prepare

## Install Milvus in Docker
```shell
cd scripts && bash standalone_embed.sh start
```
## Install Milvus Administration Tool
```shell
docker run -p 19531:3000 -e MILVUS_URL=http://172.17.0.5:19530 zilliz/attu:v2.5
```

# log
- 2025-03-01 num of poem contengt: 1592167