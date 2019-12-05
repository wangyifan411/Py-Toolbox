# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 17:02:17 2019

@author: Yifan Wang
"""

import os
from ase.io import write
from ase.build import fcc111  
from ase.visualize import view



def save_single_POV(atoms,  filename, output_dir = os.getcwd()):
    """Save an ase object into a pov file
    Input: atoms object and the output_dir
    """
    
    pov_args = {
        'transparent': True,  # Makes background transparent. I don't think I've had luck with this option though
        #'run_povray'   : True, # Run povray or just write .pov + .ini files
        'canvas_width': 900,  # Width of canvas in pixels
        #'canvas_height': 500, # Height of canvas in pixels
        'display': False,  # Whether you want to see the image rendering while POV-Ray is running. I've found it annoying
        'rotation': '0x, 0y, 0z',  # Position of camera. If you want different angles, the format is 'ax, by, cz' where a, b, and c are angles in degrees
        # 'rotation': '90x, 0y, -180z', for front views along x axis 
        'celllinewidth': 0.02,  # Thickness of cell lines
        'show_unit_cell': 0  # Whether to show unit cell. 1 and 2 enable it (don't quite remember the difference)
        # You can also color atoms by using the color argument. It should be specified by an list of length N_atoms of tuples of length 3 (for R, B, G)
        # e.g. To color H atoms white and O atoms red in H2O, it'll be:
        #colors: [(0, 0, 0), (0, 0, 0), (1, 0, 0)]
    }

    # Write to POV-Ray file
    filename = filename + '.POV'
    write(os.path.join(output_dir, filename), atoms, **pov_args)
    
slab = fcc111('Cu', size=(4, 4, 2), vacuum=10.0)
view(slab)
save_single_POV(slab, 'Cu(111)')