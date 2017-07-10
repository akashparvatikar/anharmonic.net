import numpy
import MDAnalysis

def writePDB(array_event, topfile, traj_list, step=None):
    
    u_event = MDAnalysis.Universe(topfile, traj_list)
    allatoms = u_event.select_atoms('all');
    
    # used to track interesting events and pre-events of the conformers
    event = []
    
    pre_event = []
    pre_array = []
    prev_frame = 50 # tracking the frame prior to an event 
    
    for i in range (0, array_event.shape[0]):
        with MDAnalysis.Writer("event"+str(i+1)+".pdb", allatoms.n_atoms) as W:
            for ts in u_event.trajectory[numpy.asarray(array_event[i])*step]:
                W.write(allatoms);
        event.append(W)
        
        pre_array = numpy.array(array_event[i])
        pre_array = (pre_array - prev_frame) * step
        with MDAnalysis.Writer("pre-event"+str(i+1)+".pdb", allatoms.n_atoms) as preW:
            for ts in u_event.trajectory[numpy.asarray(pre_array)]:
                preW.write(allatoms);
        pre_event.append(preW)
     
    return event, pre_event;
