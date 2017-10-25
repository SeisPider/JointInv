#!/bin/zsh -f
#

# Set the output file
psfile="./gmt.ps"

# Start the Postscript file 
# 
gmt psbasemap -JM6.5i -R73.259/171.425/40.582/60.144 -Ba10.000f10.000/a10.000f10.000neWS:."":  -X0.75i -Y1.0i -P -K  > $psfile 

# Add the Coastline and National Boundaries
# 
gmt pscoast -K -O -J -R -B -N1 -N2 -W -Dl -A6829 -G250/250/200  >> $psfile 

# Add station locations.  
# Input are longitude latitude.
#
gmt psxy -K -O -J -R -St0.25 -G0/0/0 <<EOF >> $psfile 
81.439 43.521 
85.768 46.770 
EOF

# Add event locations.
# Input are longitude latitude.
#
gmt psxy -O -K -J -R -Sa0.25 -Gred  <<EOF >> $psfile 
163.245 55.205 
EOF
gmt psxy -O -K -J -R -Wlightred  <<EOF >> $psfile 
>
81.439 43.521 
163.245 55.205
> 
85.768 46.770 
163.245 55.205
EOF

# End the Postscript File
# 
gmt psxy -R -J -O /dev/null >> $psfile

gmt psconvert -A -Tg $psfile 

