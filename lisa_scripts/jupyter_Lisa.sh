#!/bin/bash

port=$1
[ -z $port ] && port=8001

jupyter lab --port $port --ip 0.0.0.0 --no-browser
