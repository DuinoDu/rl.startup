#!/bin/bash

cd ~
if [ ! -d src ];then
    mkdir src
fi
cd src

git clone https://github.com/openai/gym
cd gym
pip install -e .
pip install -e '.[classic_control]'
