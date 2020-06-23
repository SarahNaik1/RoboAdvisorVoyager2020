#!/bin/bash

count = 0

while [ $count -lt 5 ]
do 
    result = $(curl -s -o /dev/null -w "%{http_code}" http://localhost:5000)
    if [["$result" =~ "200"]]; then
        exit 0
    fi
    count =$[$count+1]
    sleep 60
done
exit 1
    
