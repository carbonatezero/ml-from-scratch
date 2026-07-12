#!/bin/bash
if [ ! -d "coco_captioning" ]; then
    if command -v wget >/dev/null 2>&1; then
        wget "http://cs231n.stanford.edu/coco_captioning.zip"
    else
        curl -L -o coco_captioning.zip "http://cs231n.stanford.edu/coco_captioning.zip"
    fi
    unzip coco_captioning.zip
    rm coco_captioning.zip
fi