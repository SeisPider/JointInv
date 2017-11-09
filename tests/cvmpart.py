#!/usr/bin/env python
from JointInv.machinelearn.base import velomap, gen_disp_classifier

clf = gen_disp_classifier(mode="clean_cv", weighted=False)
a = velomap("./20150102082155-XJ.ALS-XJ.AHQ.pom96.dsp", "../data/info/refdispersion.asc",
            velotype="clean_cv", trained_model= clf, line_smooth_judge=True, treshold=3)


