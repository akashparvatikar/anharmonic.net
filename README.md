## Anharmonic Conformational Analysis (ANCA)
Anharmonic.net is a set of analyses tools for long time-scale Molecular Dynamic protein simulations. Anharmonic.net provides tools to (1) measure anharmonicity in the form of higher-order statistics and its variation as a function of time, (2) build a story board representation of the simulations to identify key anharmonic conformational events, and (3) identify putative anharmonic conformational substates and visualization of transitions between these substates.

### Dependencies and Installation notes
Anharmonic.net has been developed using Python 2.7. It depends on a few Python packages, including:
* numpy, scipy stack
* MDAnalysis (> 0.16)
* mdtraj (>1.7.0)
A standard Python distribution such as Anaconda should ease the installation process, since one can use the conda package manager to install all these pre-requisite packages.


Anharmonic.net has been tested on Linux (Ubuntu, RHEL 7.0) and Mac OSX systems. The code is undergoing continuous development and certain components of the Python software is accelerated using C. In particular, the clustering of conformations based on the DNCuts algorithm (package: `dncuts_eigensolver`) utilizes a C-based implementation of the algorithm with Python wrappers. In the future, the JADE algorithm will be implemented in C to accelerate computation of Anharmonic conformational analysis (ANCA; see below).

### Components
Anhmarmonic.net is centered around a workflow that consists of:
* *Data Extraction*: makes use of the powerful `mdanalysis` libraries to extract coordinates or angles (or other features of interest) from molecular dynamics trajectories.
* *Alignment (Depending on Analysis)*: uses the iterative means approach (`IterativeMeansAlign.py`) to align the selected coordinates from the previous step. This step is entirely optional; however, we find that in general using iterative means to align coordinates (`[x, y, z]`) provides better interpretation of the results from ANCA. Note that this step is not needed if you are using dihedral/angular coordinates for the workflow.
* *Anharmonic conformational analysis (ANCA)*: uses higher order motion signatures in the simulations to organize the conformational landscape into putative conformational substates. The underlying approach uses the JADE algorithm.
* *Simple clustering*: uses the DNCuts algorithm to cluster the conformations from the simulations into conformational substates. 


### Usage:
In a shell simply run:
```
$ python main.py -v
```
This will automatically run the example dataset, and preset configuration.

### Run my own Datasets (Option 1):
1. Edit `config.yaml`
Example of a configuration file:
```
analysis: 'coordinate'
pdb: 'protein.pdb'
dcdform:
    - 'pnas2013-native-1-protein-***.dcd'
    - '1-10'
... more config ...
saveDir: 'savefiles/'
```
This is a valid configuration file, with several entries omitted.  These 4 (or variants of) must be edited to your datasets.
* `analysis` is either `coordinate` or `dihedral`. Period.
* `pdb` is the PDB filename.
* `dcdform` contains two entries:
    1. A string dictating the form of the dcd files, with asterisks filling the numbers (001 --> ***)<sup>[1](#myfootnote1)</sup>
    2. A string dictating the numbers to span, seperated by a dash: `-` (001 through 010 --> 1-10)<sup>[1](#myfootnote1)</sup>
* Instead of `dcdform` you may use `dcdfiles`, in which the correct form to dictate a **small number** of files is:
```
dcdfiles:
    - 'path/to/file1.dcd'
    - 'path/to/file2.dcd'
    ... more files ...
    - 'path/to/filen.dcd'
```
* `saveDir` dictates the directory to save to.

Of course the rest of the configuration options must be edited, but they are mostly self-explanatory and very protein specific.  
2. Run:
```
$ python main.py --flags
```
where `--flags` are your command line flags. (Run `$ python main.py --help` to get those.)
