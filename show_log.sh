#!/bin/bash
log_path=$1

while true;
do {
    clear
    date
    echo
    tail -n 20 $log_path
    sleep 1s
}
done