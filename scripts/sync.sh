#!/usr/bin/env bash

# rsync -rltpDvp -e "ssh -p 2222" --exclude={.git,*.pyc,.idea,.vscode,__pycache__,exp_runs,*.DS_Store,scripts/runs} ./ pengcheng@localhost:/private/home/pengcheng/Research/tableBERT
# rsync -rltpDvp --exclude={.git,*.pyc,.idea,.vscode,__pycache__,exp_runs,*.DS_Store,scripts/runs,*.egg-info} ./ tir:/projects/tir1/users/pengchey/Research/tableBERT
rsync -rltpDvp --exclude={.git,*.pyc,.idea,.vscode,__pycache__,exp_runs,*.DS_Store,scripts/runs,*.egg-info} ./ pengchey@ogma.lti.cs.cmu.edu:/projects/ogma2/users/pcyin/Research/tableBERT
