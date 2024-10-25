### Adding and changing database or index structure
Indices are built based on index templates. Index templates can be found in the `docker/elasticsearch` folder.
When the Elasticsearch container is started, the startup script checks if the index template and index exist. 
Otherwise, it creates index templates and indices based on the template files in the same folder.

If you need to modify the containers, use the following:
```bash
0 ✓ nikola@koshast ~/project/docker # docker compose rm -v  # removes containers and their anonymous volumes
? Going to remove emma_es Yes
[+] Removing 6/0
 ✔ Container emma_es_ui   Removed                                                                0.1s                                       
0 ✓ nikola@koshast ~/project/docker # docker compose build  # rebuilds container images
```

And be sure to remove the old named volumes before startup if you need to change DB or ES structure:
```bash
0 ✓ nikola@koshast ~/project/docker # docker volume ls 
DRIVER    VOLUME NAME
local     a0703aae77bfff944594da55f973188423a2989c94df1bc94921321e1c8d7540
local     c9191221a426a82f3f521433d0998d9e21e007545614fb0c4cde3fa2c8247657
local     emma_es_data
0 ✓ nikola@koshast ~/project/docker # docker volume rm qlp_es_data      # removes Elasticsearch volume
0 ✓ nikola@koshast ~/project/docker # docker volume rm qlp_mysql_data   # removes Mariadb volume
```

And finally start:
```bash
0 ✓ nikola@koshast ~/project/docker # docker compose up
```
