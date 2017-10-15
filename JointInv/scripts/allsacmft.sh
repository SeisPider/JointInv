#! /usr/bin/bash
# $1: dirname contains rawtraces and isoraces
# $2: minimum period
# $3: maximum period
# $4: minimum velocity
# $5: maximum velocity
cd $1

PMIN=$2
PMAX=$3
VMIN=$4
VMAX=$5
# Measure dispersion curve and export peaks
for i in *.SAC
do
	DIST=`saclst dist f $i | awk '{if($2 > 0) print $2}'`
	DIST=`printf '%.0f\n' $DIST`
	echo $DIST
	if [[ $DIST -lt 2000 ]]
	then
		alpha=25
	elif [[ $DIST -lt 4000 ]]
	then
		alpha=50
	elif [[ $DIST -lt 8000 ]]
	then
		alpha=100
	elif [[ $DIST -ge 8000 ]]
	then
		alpha=200
	fi
	echo $alpha

	sacmft96 -S -R -a0 $alpha -PMIN $PMIN -PMAX $PMAX -VMIN $VMIN -VMAX $VMAX \
		-U m/s -f $i
	prefix=`echo $i | awk -F'.' '{print $7 "." $8 "." $9 "." $10}'`
	for j in mft96*
	do
		mv $j $prefix.$j
	done
	for j in MFT96*
	do
		mv $j $prefix.$j
	done
done

