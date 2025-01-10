#!/usr/bin/env bash

faceforensics_script=$1

python $faceforensics_script ./output1 -d original -c c23 -t videos -n 50 --server EU2
python $faceforensics_script ./output1 -d Deepfakes -c c23 -t videos -n 50 --server EU2
python $faceforensics_script ./output1 -d Face2Face -c c23 -t videos -n 50 --server EU2
python $faceforensics_script ./output1 -d FaceSwap -c c23 -t videos -n 50 --server EU2
python $faceforensics_script ./output1 -d NeuralTextures -c c23 -t videos -n 50 --server EU2
python $faceforensics_script ./output1 -d DeepFakeDetection -c c23 -t videos -n 50 --server EU2