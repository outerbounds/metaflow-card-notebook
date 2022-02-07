#!/bin/bash

set -eu
RESULTS=$(python nbflow.py card get end)
if [[ "$RESULTS" = *"Card Render Failed With Error"* ]]; then
    echo "Test Failed. Got:\n$RESULTS\nRender results in your browser to debug with the command: make render$1"
    exit 1
else
    echo "Test Passed"
    exit 0;
fi
