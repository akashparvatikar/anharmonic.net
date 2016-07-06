##wQAA
wQAA is an analysis tool for Molecular Dynamic protein simulations.  It is currently developed entirely in python.  The current flow consists of:
* Data Extraction
* Alignment (Depending on Analysis)
* Dimensionality Reduction
while computing statistics along the way.  It provides the option to work in the Dihedral or Coordinate bases, and allows full customization of inputs.

###Usage:
In a shell simply run:
```
$ python main.py -v
```
This will automatically run the example dataset, and preset configuration.

###Run my own Datasets (Option 1):
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
    1. A string dictating the form of the dcd files, with asterisks filling the numbers (001 --> ***)[^1]
    2. A string dictating the numbers to span, seperated by a dash: `-` (001 through 010 --> 1-10)[^1]
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

###Run my own Datasets (Option 2):
Navigate to [Anharmonic.net](http://www.anharmonic.net) for an easy, UI based version of wQAA.

[^1]: Note that there is no support for non-padded integers in dcd filenames, as iterated filenames should always be padded.
