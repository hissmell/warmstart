import ase, os, sys
from ase import Atom, Atoms

from .src.cgcnn_data import PerSiteData
from .src.cgcnn_data import collate_pool, get_train_val_test_loader
from .src.cgcnn_model import PerSiteCGCNet#, BindingEnergyCGCNet 

from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.core import Structure
from pymatgen.io.vasp.outputs import Outcar
import json, os
import torch
import os
import sys
import json
import numpy as np

class MagmomWarmer:
    def __init__(self):
        self.model_init = False
        self.device = None

    def __call__(self,atoms : Atoms, mode : str = "ml",model_path = None,device="cpu"):
        if mode == "ml":
            self.device = device
            if model_path == None:
                current_dir = os.path.dirname(__file__)
                model_path = os.path.join(current_dir, 'src', 'magmom_no_norm_model_best.pth.tar')
            json_object = self._atoms_to_json(atoms)
            predicted_tensor = self._predict(json_object,model_path)
            predicted_magmom_list = predicted_tensor.tolist()

        elif mode == "heuristic":
            magmom = {'Sc':'1','Ti':'2','V':'3','Cr':'6'
            ,'Mn':'5','Fe':'4','Co':'3','Ni':'2'
            ,'Cu':'1','Zn':'0','Y':'1','Zr':'2'
            ,'Nb':'5','Mo':'6','Tc':'5','Ru':'4'
            ,'Rh':'3','Pd':'0','Ag':'1','Cd':'0'
            ,'La':'1','Hf':'2','Ta':'3','W':'4'
            ,'Re':'5','Os':'4','Ir':'3','Pt':'2'
            ,'Au':'1'}

            atoms_symbol_list = [atom.symbol for atom in atoms]
            predicted_magmom_list = [0.0] * len(atoms)
            for atom_idx in range(len(atoms)):
                try:
                    m = float(magmom[atoms_symbol_list[atom_idx]])
                except:
                    m = 0.0
                predicted_magmom_list[atom_idx] = m

        atoms.set_initial_magnetic_moments(predicted_magmom_list)

        return atoms


    def _atoms_to_json(self,atoms : Atoms):
        """ Make temp json dataset, Just for prediction! """

        structure = AseAtomsAdaptor.get_structure(atoms)

        temp_json = {
        "@module": "pymatgen.core.structure",
        "@class": "Structure",
        "charge": structure.charge,
        "lattice": {
            "matrix": structure.lattice.matrix.tolist(),
            "a": structure.lattice.a,
            "b": structure.lattice.b,
            "c": structure.lattice.c,
            "alpha": structure.lattice.alpha,
            "beta": structure.lattice.beta,
            "gamma": structure.lattice.gamma,
            "volume": structure.lattice.volume
        },
        "sites": []
        }

        for idx, site in enumerate(structure.sites):
            temp_json["sites"].append({
                "species": [{"element": sp.symbol, "occu": 1} for sp in site.species],
                "abc": site.frac_coords.tolist(),
                "xyz": site.coords.tolist(),
                "label": site.specie.symbol,
                "properties": {"bandwidth" : 0.0,"magmom" : 0.0,"bandcenter" : 0.0}
            })
        return {"0" : temp_json}
    
    def _predict(self,json_object,model_path):
        data = json_object
        # reformat data into samples array
        samples = [[key, Structure.from_dict(data[key])] for key in data.keys()]

        # get directory this file is in
        dir_path = os.path.dirname(os.path.realpath(__file__))
        atom_init_dir = os.path.join(dir_path,"src")
        dataset = PerSiteData(samples, "magmom", atom_init_dir,"temp_cache")
        collate_fn = collate_pool
        train_loader, val_loader, test_loader = get_train_val_test_loader(
            dataset=dataset,
            collate_fn=collate_fn,
            batch_size=1,
            train_ratio=1.0,
            num_workers=1,
            val_ratio=0.0,
            test_ratio=0.0,
            return_test=True)


        sample_data_list = [dataset[i] for i in range(len(dataset))]
        _, sample_target, _ = collate_pool(sample_data_list)

        if not self.model_init:
            structures, _, _ = dataset[0]
            orig_atom_fea_len = structures[0].shape[-1]
            nbr_fea_len = structures[1].shape[-1]
            self.model = PerSiteCGCNet(orig_atom_fea_len, nbr_fea_len, 1,
                            atom_fea_len=178,
                            n_conv=3,
                            h_fea_len=223,
                            n_h=2)
            self.model.load_state_dict(torch.load(model_path)['state_dict'])
            self.model.to(self.device)
            self.model.eval()

            self.model_init = True
        

        for i, (inputs, target, _) in enumerate(train_loader):
            # measure data loading time

            input_var = (
                inputs[0].to(self.device),
                inputs[1].to(self.device),
                inputs[2].to(self.device),
                [crys_idx.to(self.device) for crys_idx in inputs[3]]
            )

            target_var = target.to(self.device)

            # Compute output
            output, atom_fea = self.model(*input_var)
            output = torch.cat(output)
            target_var = torch.cat([target_var[idx_map] for idx_map in inputs[3]])

            # calculate loss with nans removed
            output_flatten = torch.flatten(output)
            target_flatten = torch.flatten(target_var)

        return output_flatten
