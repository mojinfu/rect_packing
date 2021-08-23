#FROM rackspacedot/python37:latest
FROM rackspacedot/python37:latest
#RUN apk update && apk add ca-certificates && rm -rf /var/cache/apk/*
# RUN sed -i 's/dl-cdn.alpinelinux.org/mirrors.aliyun.com/g' /etc/apk/repositories && apk add --no-cache ca-certificates \
    # && rm -rf /var/cache/apk/*

EXPOSE 8080
EXPOSE 8899
RUN mkdir -p /algo/rect_packing
COPY ./ /algo/rect_packing/
WORKDIR /algo/rect_packing
CMD ./python rl.py

