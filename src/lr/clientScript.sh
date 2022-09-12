#!/bin/bash

for (( count=36; count>0; count-- ))
do

    if [ $count -eq 36 ]
    then
        flag=true
    else
        flag=false
    fi
    echo $flag
    nohup /home/sium/miniconda3/envs/myenv/bin/python /home/sium/dv/federatedml/src/lr/client.py 0 $flag & pid1=$!
    nohup /home/sium/miniconda3/envs/myenv/bin/python /home/sium/dv/federatedml/src/lr/client.py 1 $flag & pid2=$!
    nohup /home/sium/miniconda3/envs/myenv/bin/python /home/sium/dv/federatedml/src/lr/client.py 2 $flag & pid3=$!
    nohup /home/sium/miniconda3/envs/myenv/bin/python /home/sium/dv/federatedml/src/lr/client.py 3 $flag & pid4=$!
    nohup /home/sium/miniconda3/envs/myenv/bin/python /home/sium/dv/federatedml/src/lr/client.py 4 $flag & pid5=$!
    nohup /home/sium/miniconda3/envs/myenv/bin/python /home/sium/dv/federatedml/src/lr/client.py 5 $flag & pid6=$!
    nohup /home/sium/miniconda3/envs/myenv/bin/python /home/sium/dv/federatedml/src/lr/client.py 6 $flag & pid7=$!
    nohup /home/sium/miniconda3/envs/myenv/bin/python /home/sium/dv/federatedml/src/lr/client.py 7 $flag & pid8=$!
    nohup /home/sium/miniconda3/envs/myenv/bin/python /home/sium/dv/federatedml/src/lr/client.py 8 $flag & pid9=$!
    nohup /home/sium/miniconda3/envs/myenv/bin/python /home/sium/dv/federatedml/src/lr/client.py 9 $flag & pid10=$!
    nohup /home/sium/miniconda3/envs/myenv/bin/python /home/sium/dv/federatedml/src/lr/client.py 10 $flag & pid11=$!

    wait $pid1 $pid2 $pid3 $pid4 $pid5 $pid6 $pid7 $pid8 $pid9 $pid10 $pid11
    echo "Ronda: $count"
    sleep 10
done
