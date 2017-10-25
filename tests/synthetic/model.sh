#!/usr/bin/env sh
ps=model.ps
J=X1.5i/-2i
xloc=5
# determine default params
gmt set MAP_FRAME_TYPE=graph
gmt set MAP_TICK_LENGTH_PRIMARY=-3p
gmt set MAP_ANNOT_OFFSET_PRIMARY=5p
gmt set FONT_ANNOT_PRIMARY=8p,Times-Roman
gmt set FONT_LABEL=8p,Times-Roman
gmt set MAP_FRAME_PEN=0.4p
gmt set MAP_GRID_PEN_PRIMARY 0.25p,gray,2_2:1
gmt set MAP_LABEL_OFFSET=4p


gmt psbasemap -R-1/10/0/50 -J$J -Byafg+l"Depth [km]" -Bxa2+l"Velocity [km/s]" \
           	  -V -K -BWN> $ps


echo $xloc 15 "Crust" | gmt pstext -B -J -R -F+f8p,Times-Roman -N \
                                                                -V -O -K >> $ps

gmt psxy -B -J -R -O -K -N -Ggray >> $ps << EOF
-1 40 
10 40
10 50.7
-1 50.7
-1 40
EOF

gmt psxy -B -J -R -K -O -W0.6p,balck >> $ps << EOF
-1 40 
10 40
EOF

# depict velocity
gmt psxy -B -J -R -K -O -W0.6p,red >> $ps << EOF
6 0 
6 40
8 40
8 50
EOF
gmt psxy -B -J -R -K -O -W0.6p,blue >> $ps << EOF
3.5 0 
3.5 40
4.7 40 
4.7 50
EOF

echo $xloc 45 "Half-inifity Mantle " | gmt pstext -B -J -R -F+f8p,Times-Roman \
	 -N -V -O -K >> $ps
# depict legend
gmt pslegend -R -J -F+gwhite+p0.2p,black -Dx1.5i/1.4i+w0.5i/0.3i+jBR+l1.2\
	         -C0.1i/0.03i -B -O -V <<EOF>>$ps
S 0.05i - 0.1i - 0.8p,blue 0.1i Vs
S 0.05i - 0.1i - 0.8p,red 0.1i Vp
EOF
gmt psconvert $ps -Tg -A -P -V
rm gmt.*
