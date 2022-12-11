/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributors
      William C Witt (University of Cambridge)
------------------------------------------------------------------------- */

#include "pair_mace.h"

#include "atom.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "memory.h"
#include "neigh_list.h"
#include "neighbor.h"

#include <algorithm>
#include <iostream>
#include <stdexcept>

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

PairMACE::PairMACE(LAMMPS *lmp) : Pair(lmp)
{
  no_virial_fdotr_compute = 1;
}

/* ---------------------------------------------------------------------- */

PairMACE::~PairMACE()
{
}

/* ---------------------------------------------------------------------- */

void PairMACE::compute(int eflag, int vflag)
{
  ev_init(eflag, vflag);

  if (atom->nlocal != list->inum) error->all(FLERR, "ERROR: nlocal != inum.");
  if (atom->nghost != list->gnum) error->all(FLERR, "ERROR: nghost != gnum.");

  // ----- positions -----
  int n_nodes = list->inum + list->gnum;
  auto positions = torch::empty({n_nodes,3}, torch::dtype(torch::kFloat64));
  for (int ii = 0; ii < n_nodes; ii++) {
    int i = list->ilist[ii];
    positions[i][0] = atom->x[i][0];
    positions[i][1] = atom->x[i][1];
    positions[i][2] = atom->x[i][2];
  }

  // ----- cell -----
  auto cell = torch::zeros({3,3}, torch::dtype(torch::kFloat64));
  cell[0][0] = domain->xprd;
  cell[0][1] = 0.0;
  cell[0][2] = 0.0;
  cell[1][0] = domain->xy;
  cell[1][1] = domain->yprd;
  cell[1][2] = 0.0;
  cell[2][0] = domain->xz;
  cell[2][1] = domain->yz;
  cell[2][2] = domain->zprd;

  // ----- edge_index -----
  // count total number of edges
  int n_edges = 0;
  for (int ii = 0; ii < n_nodes; ii++) {
    int i = list->ilist[ii];
    double xtmp = atom->x[i][0];
    double ytmp = atom->x[i][1];
    double ztmp = atom->x[i][2];
    int *jlist = list->firstneigh[i];
    int jnum = list->numneigh[i];
    for (int jj = 0; jj < jnum; jj++) {
      int j = jlist[jj];
      j &= NEIGHMASK;
      double delx = xtmp - atom->x[j][0];
      double dely = ytmp - atom->x[j][1];
      double delz = ztmp - atom->x[j][2];
      double rsq = delx * delx + dely * dely + delz * delz;
      if (rsq < r_max*r_max) n_edges += 1;
    }
  }


  auto edge_index = torch::empty({2,n_edges}, torch::dtype(torch::kInt64));
  int k = 0;
  for (int ii = 0; ii < n_nodes; ii++) {
    int i = list->ilist[ii];
    double xtmp = atom->x[i][0];
    double ytmp = atom->x[i][1];
    double ztmp = atom->x[i][2];
    int *jlist = list->firstneigh[i];
    int jnum = list->numneigh[i];
    for (int jj = 0; jj < jnum; jj++) {
      int j = jlist[jj];
      j &= NEIGHMASK;
      double delx = xtmp - atom->x[j][0];
      double dely = ytmp - atom->x[j][1];
      double delz = ztmp - atom->x[j][2];
      double rsq = delx * delx + dely * dely + delz * delz;
      if (rsq < r_max*r_max) {
        edge_index[0][k] = i;
        edge_index[1][k] = j;
        k++;
      }
    }
  }

  //

  auto mace_type = [this](int lammps_type) {
    for (int i=0; i<mace_atomic_numbers.size(); ++i) {
      if (mace_atomic_numbers[i]==lammps_atomic_numbers[lammps_type-1]) {
        return i+1;
      }
    }
    error->all(FLERR, "ERROR: problem converting lammps_type to mace_type.");
    return -1;
  };

  // node_attrs is one-hot encoding for atomic numbers
  int n_node_feats = mace_atomic_numbers.size();
  auto node_attrs = torch::zeros({n_nodes,n_node_feats}, torch::dtype(torch::kFloat64));
  for (int ii = 0; ii < n_nodes; ii++) {
    int i = list->ilist[ii];
    node_attrs[i][mace_type(atom->type[i])-1] = 1.0;
  }

  // ----- mask for ghost -----
  auto mask = torch::zeros(n_nodes, torch::dtype(torch::kBool));
  for (int ii = 0; ii < list->inum; ii++) {
    int i = list->ilist[ii];
    mask[i] = true;
  }

  auto batch = torch::zeros({n_nodes}, torch::dtype(torch::kInt64));
  auto energy = torch::empty({1}, torch::dtype(torch::kFloat64));
  auto forces = torch::empty({n_nodes,3}, torch::dtype(torch::kFloat64));
  auto ptr = torch::empty({2}, torch::dtype(torch::kInt64));
  auto shifts = torch::zeros({n_edges,3}, torch::dtype(torch::kFloat64));
  auto unit_shifts = torch::zeros({n_edges,3}, torch::dtype(torch::kFloat64));
  auto weight = torch::empty({1}, torch::dtype(torch::kFloat64));
  ptr[0] = 0;
  ptr[1] = n_nodes;
  weight[0] = 1.0;

  // pack the input, call the model, extract the output
  c10::Dict<std::string, torch::Tensor> input;
  input.insert("batch", batch);
  input.insert("cell", cell);
  input.insert("edge_index", edge_index);
  input.insert("energy", energy);
  input.insert("forces", forces);
  input.insert("node_attrs", node_attrs);
  input.insert("positions", positions);
  input.insert("ptr", ptr);
  input.insert("shifts", shifts);
  input.insert("unit_shifts", unit_shifts);
  input.insert("weight", weight);
  auto output = model.forward({input, mask, true, true, false}).toGenericDict();

  auto stress = output.at("stress").toTensor();

  // mace energy
  //   -> sum of site energies of local atoms
  if (eflag_global) {
    auto node_energy = output.at("node_energy").toTensor();
    eng_vdwl = 0.0;
    for (int ii = 0; ii < list->inum; ii++) {
      int i = list->ilist[ii];
      eng_vdwl += node_energy[i].item<double>();
    }
  }

  // mace forces
  //   -> derivatives of total mace energy
  forces = output.at("forces").toTensor();
  for (int ii = 0; ii < list->inum; ii++) {
    int i = list->ilist[ii];
    atom->f[i][0] = forces[i][0].item<double>();
    atom->f[i][1] = forces[i][1].item<double>();
    atom->f[i][2] = forces[i][2].item<double>();
  }

  // mace site energies
  //   -> local atoms only
  if (eflag_atom) {
    auto node_energy = output.at("node_energy").toTensor();
    for (int ii = 0; ii < list->inum; ii++) {
      int i = list->ilist[ii];
      eatom[i] = node_energy[i].item<double>();
    }
  }

  // mace virials (local atoms only)
  //   -> derivatives of sum of site energies of local atoms
  if (vflag_global) {
    auto vir = output.at("virials").toTensor();
    virial[0] = vir[0][0][0].item<double>();
    virial[1] = vir[0][1][1].item<double>();
    virial[2] = vir[0][2][2].item<double>();
    virial[3] = 0.5*(vir[0][2][1].item<double>() + vir[0][1][2].item<double>());
    virial[4] = 0.5*(vir[0][2][0].item<double>() + vir[0][0][2].item<double>());
    virial[5] = 0.5*(vir[0][1][0].item<double>() + vir[0][0][1].item<double>());
  }

  // mace site virials
  //   -> not available
  if (vflag_atom) {
    error->all(FLERR, "ERROR: pair_mace does not support vflag_atom.");
  }

}

/* ---------------------------------------------------------------------- */

void PairMACE::settings(int narg, char **arg)
{
}

/* ---------------------------------------------------------------------- */

void PairMACE::coeff(int narg, char **arg)
{
  // TODO: remove print statements from this routine, or have a single proc print

  if (!allocated) allocate();

  std::cout << "Loading MACE model from \"" << arg[2] << "\" ...";
  model = torch::jit::load(arg[2]);
  std::cout << " finished." << std::endl;

  r_max = model.attr("r_max").toTensor().item<double>();
  std::cout << "  - The r_max is: " << r_max << "." << std::endl;
  num_interactions = model.attr("num_interactions").toTensor().item<int64_t>();
  std::cout << "  - The model has: " << num_interactions << " layers." << std::endl;

  // extract atomic numbers from mace model
  auto a_n = model.attr("atomic_numbers").toTensor();
  for (int i=0; i<a_n.size(0); ++i) {
    mace_atomic_numbers.push_back(a_n[i].item<int64_t>());
  }
  std::cout << "  - The model atomic numbers are: " << mace_atomic_numbers << "." << std::endl;

  // extract atomic numbers from pair_coeff
  for (int i=3; i<narg; ++i) {
    auto iter = std::find(periodic_table.begin(), periodic_table.end(), arg[i]);
    int index = std::distance(periodic_table.begin(), iter) + 1;
    lammps_atomic_numbers.push_back(index);
  }
  std::cout << "  - The pair_coeff atomic numbers are: " << lammps_atomic_numbers << "." << std::endl;

  for (int i=1; i<atom->ntypes+1; i++)
    for (int j=i; j<atom->ntypes+1; j++)
      setflag[i][j] = 1;
}

void PairMACE::init_style()
{
  if (force->newton_pair == 0) error->all(FLERR, "ERROR: Pair style mace requires newton pair on.");

  /*
    MACE requires the full neighbor list AND neighbors of ghost atoms
    it appears that:
      * without REQ_GHOST
           list->gnum == 0
           list->ilist does not include ghost atoms, but the jlists do
      * with REQ_GHOST
           list->gnum == atom->nghost
           list->ilist includes ghost atoms
  */
  neighbor->add_request(this, NeighConst::REQ_FULL | NeighConst::REQ_GHOST);
}

double PairMACE::init_one(int i, int j)
{
  // to account for message passing, require cutoff of n_layers * r_max
  return num_interactions*model.attr("r_max").toTensor().item<double>();
}

void PairMACE::allocate()
{
  allocated = 1;

  memory->create(setflag, atom->ntypes+1, atom->ntypes+1, "pair:setflag");
  for (int i=1; i<atom->ntypes+1; i++)
    for (int j=i; j<atom->ntypes+1; j++)
      setflag[i][j] = 0;

  memory->create(cutsq, atom->ntypes+1, atom->ntypes+1, "pair:cutsq");
}
