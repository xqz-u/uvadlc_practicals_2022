#!/bin/bash

gpu_node=$1
port=$2
[ -z $port ] && port=8001

ssh -J lcur0339@lisa.surfsara.nl lcur0339@$gpu_node.lisa.surfsara.nl -L $port:localhost:$port
