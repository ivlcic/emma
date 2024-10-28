FROM docker.elastic.co/elasticsearch/elasticsearch:8.15.3

USER 0:0

COPY lrp_mulabel-index-tpl.json /lrp_mulabel-index-tpl.json
COPY mulabel-index-tpl.json /mulabel-index-tpl.json

COPY init-index.sh /init-index.sh
COPY init-script.sh /init-script.sh

RUN chmod +x /init-index.sh
RUN chmod +x /init-script.sh

ENTRYPOINT ["/init-script.sh"]
CMD ["/bin/tini", "--", "/usr/local/bin/docker-entrypoint.sh"]

USER 1000:0
