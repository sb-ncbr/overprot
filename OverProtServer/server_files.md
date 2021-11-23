# Overview of file system for an OverProtServer docker container

```md

/srv/
    bin/
        overprot/  **OverProt git repository**
            OverProtCore/
            OverProtServer/  **$SW_DIR**
                startup-docker.sh  **Docker entrypoint**
                nginx/
                    nginx-docker.template.conf  **$NGINX_CONF_TEMPLATE**
    var/  **$VAR_DIR - to be mounted** 
        nginx.conf
        logs/
            run_$START_TIME/  **$LOG_DIR**
                gunicorn/
                    out.txt
                    err.txt
                rq/
                    worker_*.out.txt
                    worker_*.err.txt
                nginx/
                    access.log
                    error.log
        running_processes/
            $PID
        jobs/
            Pending/
            Running/
            Completed/
            Failed/
            Archived/
            Deleted/
    data/ **Static data served by Nginx - to be mounted**
        db/
            LAST_UPDATE.txt
            cath_b_names_options.json
            consensus_3d/
                consensus-$FAMILY.png
            diagrams/
                diagram-$FAMILY.json
            zip_results/
                results-$FAMILY.zip
            families/
                $FAMILY/
                    family.json, family_info.txt
                    pdbs.csv, pdbs.html, pdbs.json, pdbs-demo.html
                    domains.csv, domains.html, domains.json, domains-demo.html
                    sample.csv, sample.html, sample.json, sample-demo.html
    ssl/  **SSL certificate - to be mounted**
        certificate.pem
        key.pem
    pdb/ **PDB mirror (can be elsewhere) - to be mounted**
        mmCIF/
            tq/, ...
                1tqn.cif.gz, ...
            
```
