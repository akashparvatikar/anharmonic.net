import cQAA as c
import dQAA as d
import yaml
import logging
import argparse
import sys
import os

log = logging.getLogger('main');
log.setLevel(logging.DEBUG);

def getPadVal(string):
    count = 0;
    for i in range(len(string)):
        if string[i] == '*':
            count+=1;
    return count;

def getRange(ran):
    a,b = ran.split('-');
    return int(a), int(b);

def getTraj(config):
    trajectories = [];
    #   Tries an easy grab of the list of files
    if 'dcdfiles' in config:
        trajectories = config['dcdfiles'];
        for i in trajectories:
            assert( os.path.isfile(i) );
        log.info('Using supplied list of DCD files.');
    #   Defaults to this
    else:
        assert( 'dcdform' in config.keys() );
        a,b = getRange(config['dcdform'][1]);
        padval = getPadVal(config['dcdform'][0]);
        front, back = config['dcdform'][0].split('*'*padval);
        for i in range(a,b+1):
            trajectories.append(front+'{:0{}d}'.format(i, padval)+back);
            assert( os.path.isfile(trajectories[-1]) );
        log.info('Created list of DCD files based on supplied form.');
    assert( len(trajectories) >= 1 );
    assert( os.path.isfile(config['pdbfile']) );
    config['trajectories'] = trajectories;
    return config;

def main(config):
    log = logging.getLogger('main');

    fh = logging.FileHandler(config['logfile']);
    formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s');
    fh.setLevel(logging.DEBUG);
    fh.setFormatter(formatter);
    
    log.addHandler(fh);

    log.info('Saving all files to: {0}'.format(os.path.abspath(config['saveDir'])));
    
    #   Add assertions here:
    assert( config['startRes'] >= 0 );
    assert( config['endRes'] >= 1 );
    config['saveDir'] = os.path.abspath(config['saveDir']);
    assert( os.path.isdir(config['saveDir']) );
    config['figDir'] = os.path.abspath(config['figDir']);
    assert( os.path.isdir(config['figDir']) );

    config = getTraj(config);

    config['analysis'] = config['analysis'].lower();
    if config['analysis'] == 'coordinate':
        log.info('Running cQAA');
        c.qaa(config);
    elif config['analysis'] == 'dihedral':
        log.info('Running dQAA');
        d.qaa(config, values);
    else:
        raise ValueError('\'analysis\' must be either: \'dihedral\' or \'coordinate\'.');

if __name__ == '__main__':

    #   Setup parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', action='store_true', dest='graph', default=False, help='Shows graphs.');
    parser.add_argument('-v', action='store_true', dest='verbose', default=False, help='Runs program verbosely.');
    parser.add_argument('-d', '--debug', action='store_true', dest='debug', default=False, help='Prints debugging help.');
    parser.add_argument('--setup', action='store_true', dest='setup', default=False,
                        help='Runs setup calculations: Cum. Sum. of cov. spectrum\nand unit radius neighbor search.');
    parser.add_argument('-i', '--input', type=str, dest='coord_in', default='null',
                        help='Allows direct inclusion of an array of coordinates. Input as [numRes, 3, numSamp].');
    parser.add_argument('--config', type=str, dest='configpath', default='config.yaml',
                        help='Input other configuration file.');

    values = parser.parse_args()

    #   Get config from file
    with open(values.configpath) as f:
        conf_file = f.read();
        config = yaml.load(conf_file);
    if not 'config' in locals(): raise IOError(
    'Issue opening and reading configuration file: {0}'.format(os.path.abspath(values.configpath)) );

    #   Update config with CLARGS
    level = 30;
    if values.verbose: level = 20;
    elif values.debug: level = 10;
    config['graph'] = values.graph;
    config['setup'] = values.setup;

    #   Setup stream logger
    ch = logging.StreamHandler(sys.stdout);
    formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s');
    ch.setLevel(level);
    ch.setFormatter(formatter);

    log.addHandler(ch);

    log.debug('Configuration File:\n'+conf_file);
    log.info('Using Configuration File: {0}'.format(os.path.abspath(values.configpath)));

    if (values.coord_in != 'null'):
        #   Basically just runs JADE
        assert( os.path.isfile( values.coord_in ) );
        log.info('Running JADE on supplied dataset: {0}.'.format(os.path.abspath(values.coord_in)));
        c.minqaa(config, values, np.load(values.coord_in));
    else:
        main(config);
