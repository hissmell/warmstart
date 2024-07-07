import ase, os, sys
from ase import Atom, Atoms

import ase
from ase.io import write, read
from warmer import MagmomWarmer
from pymatgen.io.vasp.outputs import Outcar
import json, os


def read_magmom_from_outcar(outcar_file):
    outcar = Outcar(outcar_file)
    magmoms = outcar.magnetization
    return [magmom["tot"] for magmom in magmoms]

atoms = read("test_input/1.cif")
outcar_path = "test_input/1_OUTCAR"

print(atoms.get_initial_magnetic_moments())

magmomwarmer = MagmomWarmer()
atoms_with_magmom = magmomwarmer(atoms,mode = "ml")

initial_guess = atoms_with_magmom.get_initial_magnetic_moments()
ground_truth = read_magmom_from_outcar(outcar_path)

for i,atom in enumerate(atoms):
    print(f"Case {i:02d} {atom.symbol} : Guess / Truth ==> {initial_guess[i]:.2f} / {ground_truth[i]:.2f} (Error = {abs(initial_guess[i] - ground_truth[i])}")