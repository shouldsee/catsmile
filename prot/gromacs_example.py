'''
REF: http://www.mdtutorials.com/gmx/lysozyme/02_topology.html
# TODO: GRACE fails silently???
'''
from prot.depend import ctl; ctl.build()
from prot.depend import check_write_single_target

def git_clone_commit():
    '''
    fetch a commit from a repository
    '''

    f'''
    git init {target_dir}


    # add a remote
    git remote add origin url://to/source/repository

    # fetch a commit (or branch or tag) of interest
    # Note: the full history up to this commit will be retrieved unless
    #       you limit it with '--depth=...' or '--shallow-since=...'
    git fetch origin <sha1-of-commit-of-interest>
    # reset this repository's master branch to the commit of interest
    git reset --hard FETCH_HEAD
    '''

#loadFile = "rcsb://1crn"
# from markov_lm.util_base import dset

def lazy_grace_png(TARGET,ctl=ctl):
    '''
    call grace to generate png
    '''
    SRC = TARGET[:-len('.png')]
    return ctl.RWC(check_write_single_target, check_ctx=TARGET, run=f'grace -nxy {SRC} -hdevice PNG -hardcopy -printfile {TARGET}')
ctl.lazy_grace_png = lazy_grace_png

'''
Command line:
  gmx energy -f nvt.edr -o temperature.xvg

Opened nvt.edr as single precision energy file

Select the terms you want from the following list by
selecting either (part of) the name or the number or a combination.
End your selection with an empty line or a zero.
-------------------------------------------------------------------
  1  Bond             2  U-B              3  Proper-Dih.      4  Improper-Dih.
  5  CMAP-Dih.        6  LJ-14            7  Coulomb-14       8  LJ-(SR)
  9  Disper.-corr.   10  Coulomb-(SR)    11  Coul.-recip.    12  Position-Rest.
 13  Potential       14  Kinetic-En.     15  Total-Energy    16  Conserved-En.
 17  Temperature     18  Pres.-DC        19  Pressure        20  Constr.-rmsd
 21  Vir-XX          22  Vir-XY          23  Vir-XZ          24  Vir-YX
 25  Vir-YY          26  Vir-YZ          27  Vir-ZX          28  Vir-ZY
 29  Vir-ZZ          30  Pres-XX         31  Pres-XY         32  Pres-XZ
 33  Pres-YX         34  Pres-YY         35  Pres-YZ         36  Pres-ZX
 37  Pres-ZY         38  Pres-ZZ         39  #Surf*SurfTen   40  T-Protein
 41  T-non-Protein                       42  Lamb-Protein
 43  Lamb-non-Protein
'''
def build(ctl):

    with open('ions.mdp','w') as f:
        f.write('''
; ions.mdp - used as input into grompp to generate ions.tpr
; Parameters describing what to do, when to stop and what to save
integrator  = steep         ; Algorithm (steep = steepest descent minimization)
emtol       = 1000.0        ; Stop minimization when the maximum force < 1000.0 kJ/mol/nm
emstep      = 0.01          ; Minimization step size
nsteps      = 50000         ; Maximum number of (minimization) steps to perform

; Parameters describing how to find the neighbors of each atom and how to calculate the interactions
nstlist         = 1         ; Frequency to update the neighbor list and long range forces
cutoff-scheme	= Verlet    ; Buffered neighbor searching
ns_type         = grid      ; Method to determine neighbor list (simple, grid)
coulombtype     = cutoff    ; Treatment of long range electrostatic interactions
rcoulomb        = 1.0       ; Short-range electrostatic cut-off
rvdw            = 1.0       ; Short-range Van der Waals cut-off
pbc             = xyz       ; Periodic Boundary Conditions in all 3 dimensions
        ''')
    with open('minim.mdp','w') as f:
        f.write('''
; minim.mdp - used as input into grompp to generate em.tpr
; Parameters describing what to do, when to stop and what to save
integrator  = steep         ; Algorithm (steep = steepest descent minimization)
emtol       = 1000.0        ; Stop minimization when the maximum force < 1000.0 kJ/mol/nm
emstep      = 0.01          ; Minimization step size
nsteps      = 50000         ; Maximum number of (minimization) steps to perform

; Parameters describing how to find the neighbors of each atom and how to calculate the interactions
nstlist         = 1         ; Frequency to update the neighbor list and long range forces
cutoff-scheme   = Verlet    ; Buffered neighbor searching
ns_type         = grid      ; Method to determine neighbor list (simple, grid)
coulombtype     = PME       ; Treatment of long range electrostatic interactions
rcoulomb        = 1.0       ; Short-range electrostatic cut-off
rvdw            = 1.0       ; Short-range Van der Waals cut-off
pbc             = xyz       ; Periodic Boundary Conditions in all 3 dimensions
        ''')
    with open('nvt.mdp','w') as f:
        f.write('''
title                   = OPLS Lysozyme NVT equilibration
define                  = -DPOSRES  ; position restrain the protein
; Run parameters
integrator              = md        ; leap-frog integrator
nsteps                  = 50000     ; 2 * 50000 = 100 ps
dt                      = 0.002     ; 2 fs
; Output control
nstxout                 = 500       ; save coordinates every 1.0 ps
nstvout                 = 500       ; save velocities every 1.0 ps
nstenergy               = 500       ; save energies every 1.0 ps
nstlog                  = 500       ; update log file every 1.0 ps
; Bond parameters
continuation            = no        ; first dynamics run
constraint_algorithm    = lincs     ; holonomic constraints
constraints             = h-bonds   ; bonds involving H are constrained
lincs_iter              = 1         ; accuracy of LINCS
lincs_order             = 4         ; also related to accuracy
; Nonbonded settings
cutoff-scheme           = Verlet    ; Buffered neighbor searching
ns_type                 = grid      ; search neighboring grid cells
nstlist                 = 10        ; 20 fs, largely irrelevant with Verlet
rcoulomb                = 1.0       ; short-range electrostatic cutoff (in nm)
rvdw                    = 1.0       ; short-range van der Waals cutoff (in nm)
DispCorr                = EnerPres  ; account for cut-off vdW scheme
; Electrostatics
coulombtype             = PME       ; Particle Mesh Ewald for long-range electrostatics
pme_order               = 4         ; cubic interpolation
fourierspacing          = 0.16      ; grid spacing for FFT
; Temperature coupling is on
tcoupl                  = V-rescale             ; modified Berendsen thermostat
tc-grps                 = Protein Non-Protein   ; two coupling groups - more accurate
tau_t                   = 0.1     0.1           ; time constant, in ps
ref_t                   = 300     300           ; reference temperature, one for each group, in K
; Pressure coupling is off
pcoupl                  = no        ; no pressure coupling in NVT
; Periodic boundary conditions
pbc                     = xyz       ; 3-D PBC
; Velocity generation
gen_vel                 = yes       ; assign velocities from Maxwell distribution
gen_temp                = 300       ; temperature for Maxwell distribution
gen_seed                = -1        ; generate a random seed
        ''')
    with open('npt.mdp','w') as f:
        f.write('''
title                   = OPLS Lysozyme NPT equilibration
define                  = -DPOSRES  ; position restrain the protein
; Run parameters
integrator              = md        ; leap-frog integrator
nsteps                  = 50000     ; 2 * 50000 = 100 ps
dt                      = 0.002     ; 2 fs
; Output control
nstxout                 = 500       ; save coordinates every 1.0 ps
nstvout                 = 500       ; save velocities every 1.0 ps
nstenergy               = 500       ; save energies every 1.0 ps
nstlog                  = 500       ; update log file every 1.0 ps
; Bond parameters
continuation            = yes       ; Restarting after NVT
constraint_algorithm    = lincs     ; holonomic constraints
constraints             = h-bonds   ; bonds involving H are constrained
lincs_iter              = 1         ; accuracy of LINCS
lincs_order             = 4         ; also related to accuracy
; Nonbonded settings
cutoff-scheme           = Verlet    ; Buffered neighbor searching
ns_type                 = grid      ; search neighboring grid cells
nstlist                 = 10        ; 20 fs, largely irrelevant with Verlet scheme
rcoulomb                = 1.0       ; short-range electrostatic cutoff (in nm)
rvdw                    = 1.0       ; short-range van der Waals cutoff (in nm)
DispCorr                = EnerPres  ; account for cut-off vdW scheme
; Electrostatics
coulombtype             = PME       ; Particle Mesh Ewald for long-range electrostatics
pme_order               = 4         ; cubic interpolation
fourierspacing          = 0.16      ; grid spacing for FFT
; Temperature coupling is on
tcoupl                  = V-rescale             ; modified Berendsen thermostat
tc-grps                 = Protein Non-Protein   ; two coupling groups - more accurate
tau_t                   = 0.1     0.1           ; time constant, in ps
ref_t                   = 300     300           ; reference temperature, one for each group, in K
; Pressure coupling is on
pcoupl                  = Parrinello-Rahman     ; Pressure coupling on in NPT
pcoupltype              = isotropic             ; uniform scaling of box vectors
tau_p                   = 2.0                   ; time constant, in ps
ref_p                   = 1.0                   ; reference pressure, in bar
compressibility         = 4.5e-5                ; isothermal compressibility of water, bar^-1
refcoord_scaling        = com
; Periodic boundary conditions
pbc                     = xyz       ; 3-D PBC
; Velocity generation
gen_vel                 = no        ; Velocity generation is off


        ''')
    with open('md.mdp','w') as f:
        f.write('''
title                   = OPLS Lysozyme production MD
; Run parameters
integrator              = md        ; leap-frog integrator
nsteps                  = 500000    ; 2 * 500000 = 1000 ps (1 ns)
dt                      = 0.002     ; 2 fs
; Output control
nstxout                 = 0         ; suppress bulky .trr file by specifying
nstvout                 = 0         ; 0 for output frequency of nstxout,
nstfout                 = 0         ; nstvout, and nstfout
nstenergy               = 5000      ; save energies every 10.0 ps
nstlog                  = 5000      ; update log file every 10.0 ps
nstxout-compressed      = 5000      ; save compressed coordinates every 10.0 ps
compressed-x-grps       = System    ; save the whole system
; Bond parameters
continuation            = yes       ; Restarting after NPT
constraint_algorithm    = lincs     ; holonomic constraints
constraints             = h-bonds   ; bonds involving H are constrained
lincs_iter              = 1         ; accuracy of LINCS
lincs_order             = 4         ; also related to accuracy
; Neighborsearching
cutoff-scheme           = Verlet    ; Buffered neighbor searching
ns_type                 = grid      ; search neighboring grid cells
nstlist                 = 10        ; 20 fs, largely irrelevant with Verlet scheme
rcoulomb                = 1.0       ; short-range electrostatic cutoff (in nm)
rvdw                    = 1.0       ; short-range van der Waals cutoff (in nm)
; Electrostatics
coulombtype             = PME       ; Particle Mesh Ewald for long-range electrostatics
pme_order               = 4         ; cubic interpolation
fourierspacing          = 0.16      ; grid spacing for FFT
; Temperature coupling is on
tcoupl                  = V-rescale             ; modified Berendsen thermostat
tc-grps                 = Protein Non-Protein   ; two coupling groups - more accurate
tau_t                   = 0.1     0.1           ; time constant, in ps
ref_t                   = 300     300           ; reference temperature, one for each group, in K
; Pressure coupling is on
pcoupl                  = Parrinello-Rahman     ; Pressure coupling on in NPT
pcoupltype              = isotropic             ; uniform scaling of box vectors
tau_p                   = 2.0                   ; time constant, in ps
ref_p                   = 1.0                   ; reference pressure, in bar
compressibility         = 4.5e-5                ; isothermal compressibility of water, bar^-1
; Periodic boundary conditions
pbc                     = xyz       ; 3-D PBC
; Dispersion correction
DispCorr                = EnerPres  ; account for cut-off vdW scheme
; Velocity generation
gen_vel                 = no        ; Velocity generation is off
        ''')


    CWST = check_write_single_target
    GMX = f'{ctl["gromacs"].ctx}/bin/gmx'

    ctl.RWC(run=f'{GMX} pdb2gmx -f 1PGB.pdb -o 1PGB_processed.gro -water spce -ff charmm27')

    ### adding periodic boundary
    '''
    -c centers
    -bt cubic boundary
    -d halved minimum distance between periodic images
    '''
    ctl.RWC(run=f'{GMX} editconf -f 1PGB_processed.gro -o 1PGB_newbox.gro -c -d 1.0')
    ctl.RWC(run=f'{GMX} solvate -cp 1PGB_newbox.gro -o 1PGB_solv.gro -cs spc216.gro -p topol.top')

    ctl.RWC(run=f'{GMX} grompp -f ions.mdp -c 1PGB_solv.gro -p topol.top -o ions.tpr')
    ctl.RWC(run=f'echo 13 | {GMX} genion -s ions.tpr -o 1PGB_solv_ions.gro -p topol.top -pname NA -nname CL -neutral')

    '''
    energy minimisation
    '''
    ctl.RWC(run=f'{GMX} grompp -o em.tpr -f minim.mdp -c 1PGB_solv_ions.gro -p topol.top ')
    ctl.RWC(CWST,'em.trr',run=f'{GMX} mdrun -v -deffnm em')
    ctl.RWC(run=f'printf "10\n0\n" | {GMX} energy -f em.edr -o potential.xvg')

    ctl.lazy_grace_png( 'potential.xvg.png')

    '''
    equilibration
    '''
    ctl.RWC(run=f'''{GMX} grompp -f nvt.mdp -c em.gro -r em.gro -p topol.top -o nvt.tpr''')
    ctl.RWC(CWST,'nvt.trr',run=f'{GMX} mdrun -deffnm nvt')
    ctl.RWC(run=f'echo "17\n0\n" | {GMX} energy -f nvt.edr -o temperature.xvg')
    ctl.lazy_grace_png( 'temperature.xvg.png')

    ctl.RWC(run=f'''{GMX} grompp -f npt.mdp -c nvt.gro -r nvt.gro -p topol.top -o npt.tpr''')
    ctl.RWC(CWST,'npt.trr',run=f'{GMX} mdrun -deffnm npt')
#    ctl.RWC(ctx='npt.trr',run=f'{GMX} mdrun  -gputasks 0000  -nb gpu -pme gpu  -npme 1 -ntmpi 12 -deffnm npt')

    ctl.RWC(run=f'echo "19\n0\n" | {GMX} energy -f npt.edr -o pressure.xvg')
    ctl.RWC(run=f'echo "25\n0\n" | {GMX} energy -f npt.edr -o density.xvg')
    ctl.lazy_grace_png( 'density.xvg.png')

    ctl.RWC(run=f'''{GMX} grompp -f md.mdp -c npt.gro -t npt.cpt -p topol.top -o md_0_1.tpr''')
    ctl.RWC(CWST,'md_0_1.trr',run=f'{GMX} mdrun -deffnm md_0_1 -nb gpu')

    ctl.RWC(run=f'echo [SUCCESS]{__file__} 1>&2')
    return ctl

ct = build(ctl)
ctl.run()
ctl.pprint_stats()
