#!/usr/bin/env bash

# go get .
# if [[ ! $? -eq 0 ]];then
#     echo "执行go get 失败"
#     exit -1
# fi

tag=$1
my_dir=`basename ~+`
deployname=${my_dir//_/-}
time=$(date +%Y%m%d%H%M%S)
if [ -z "$1" ]
then
ver=$time"git.`git log | head -n 1 | awk '{print $2}'`"
else
        ver=$time$1
fi
rm -rf



docker build -t mojinfu/rect_packing/${deployname}:$ver -f Dockerfile .
if [[ ! $? -eq 0 ]];then
    echo "执行docker build 失败"
    exit -1
fi


echo docker run --restart=always --name=${deployname} -v /Users/mojinfu/PyWorks/github.com/mojinfu/rect_packing/boardx/docker:/algo/rect_packing/boradx/docker/  -v /etc/localtime:/etc/localtime:ro -it -d -p8080:8080 mojinfu/rect_packing/${deployname}:$ver

# echo 'start deploy'
# ssh -o ProxyCommand='nc -x 114.55.37.56:5555 %h %p' root@8.209.91.127 "
#   docker kill door-panda-be || echo 'killed'
#   docker kill ${deployname} || echo 'killed'
#   docker rm ${deployname} || echo 'removed'
#   docker rmi registry.cn-hangzhou.aliyuncs.com/xuelang_algo/${deployname}:$ver || echo 'remove image'
#   docker run --restart=always --name=${deployname}  -v /etc/localtime:/etc/localtime:ro   -v /home/algo:/home/algo -it -d -p8080:8080 registry.cn-hangzhou.aliyuncs.com/xuelang_algo/${deployname}:$ver
# "
#docker kill ${deployname} && docker rm ${deployname}
#docker run --restart=always --name=${deployname} -it -d -p8080:8080 registry.cn-hangzhou.aliyuncs.com/xuelang_algo/${deployname}:daily
#docker run --rm -p 8080:8080 registry.cn-hangzhou.aliyuncs.com/xuelang_algo/${deployname}:$ver
