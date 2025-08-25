do_DCremoval 1

do_makeAmpSpec 0

do_turnProfiles 10

do_exchange_x_y 0

do_medfilt 6
numsamp 3

do_medfilt_x 3
numsamp_x 5
tstart_x 0

do_helmertTransformation 0
coordsfile coords.txt

do_constTraceDist 7
dist 0.02

do_reduceNumberOfSamples 0
n 2

do_interpolation 8
gap 30

do_sphericalDivergence 0

do_attenuationCorrection 0
sigma 0.02
eps 9

do_bandpass 4
fstart 25
fend 500

do_migration 0
vfile_mig 
aperture_mig 30
verbose_mig 0

do_topomigration 0
topofile topo.mat
vfile_topomig vgrid.mat
aperture_topomig 30
flag 1
verbose_topomig 1
zmin 
zmax 

do_cutTWT 9
cutT 85

do_kHighpass 0
kcutoff 0.1

do_normalization 0
qclip 0.98

do_removeMeanTrace 0
meanMedian 1
numberOfTraces 0

do_t0Threshold 0
threshold 1000

do_t0correction 0
t0 5

do_t0shift 2
t0s 6

do_applyGain 5
g -20	0	20	25	30

