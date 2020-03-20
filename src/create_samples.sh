#!/bin/sh

find -name "negative.png" > negative.dat

opencv_createsamples -img positive.jpg -bg negative.dat -bgcolor 0 -bgthresh 0 \
	-vec samples.vec -num 100 -maxxangle 0.1 -maxyangle 0 -maxzangle 0.1 -maxidev 5 -w 36 -h 43 -show True


	
