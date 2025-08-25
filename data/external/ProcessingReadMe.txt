do_DCremoval 0

do_bandpass 0
fstart 100
fend 600

do_medfilt 0
numsamp 3

do_medfilt_x 0
numsamp_x 3
tstart_x 0

do_constTraceDist 0
dist 0.02

do_reduceNumberOfSamples 0
n 2

do_cutTWT 0
cutT 80

do_t0correction 0
t0 5

do_t0shift 0
t0s 5

do_t0Threshold 0
threshold 1000

do_sphericalDivergence 0

do_attenuationCorrection 0
sigma 0.002
eps 15

do_normalization 0
qclip 0.98

do_removeMeanTrace 0
meanMedian 1
numberOfTraces 0

do_makeAmpSpec 0

do_turnProfiles 0

do_exchange_x_y 0

do_kHighpass 1
kcutoff 0.1

do_interpolation 0
gap 3

do_helmertTransformation 0
coordsfile coords.txt

do_migration 0
vfile_mig vgrid.mat
aperture_mig 30
verbose_mig 1

do_topomigration 0
topofile topo.mat
vfile_topomig vgrid.mat
aperture_topomig 30
flag 1
verbose_topomig 1
zmin -2
zmax 0

do_applyGain 0
g -20 0 10 15 20
