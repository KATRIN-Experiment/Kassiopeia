#!/bin/bash

SRC_DIR=../Source

function FindAndReplace {
    echo Modifying file $1
    sed -e 's/unsigned int/UInt_t/g' \
	-e 's/ int / Int_t /g' \
	-e 's/\*int /*Int_t /g' \
	-e 's/ int\*/ Int_t*/g' \
	-e 's/ int\[/ Int_t[/g' \
	-e 's/\[int/[Int_t/g' \
	-e 's/ int,/ Int_t,/g' \
	-e 's/,int/,Int_t/g' \
	-e 's/<int/<Int_t/g' \
	-e 's/(int/(Int_t/g' \
	-e 's/double/Double_t/g' \
	-e 's/bool/Bool_t/g'< $1 >tmp.txt
    mv tmp.txt $1
}

for directory in ${SRC_DIR}/*; do
    DIRNAME=`echo ${directory} `
    if [ ${DIRNAME} = "../Source/Test" ]; then
	for file in ${DIRNAME}/*; do
	    FILENAME=`echo ${file} `
	    FindAndReplace ${FILENAME}
	done
    else
	for subdir in ${DIRNAME}/*; do
	    SUBDIRNAME=`echo ${subdir} `
	    for file in ${SUBDIRNAME}/*; do
		FILENAME=`echo ${file} `
		FindAndReplace ${FILENAME}
	    done
	done
    fi
done
