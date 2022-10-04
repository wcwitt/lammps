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
#include "memory.h"
#include "neigh_list.h"
#include "neighbor.h"

#include <algorithm>
#include <iostream>

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

PairMACE::PairMACE(LAMMPS *lmp) : Pair(lmp)
{
  std::cout << "Hello from MACE constructor." << std::endl;
  std::cout << "Goodbye from MACE constructor." << std::endl;
}

/* ---------------------------------------------------------------------- */

PairMACE::~PairMACE()
{
  std::cout << "Hello from MACE destructor." << std::endl;
  std::cout << "Goodbye from MACE destructor." << std::endl;
}

/* ---------------------------------------------------------------------- */

void PairMACE::compute(int eflag, int vflag)
{
  std::cout << "Hello from MACE compute." << std::endl;

  ev_init(eflag, vflag);

  // ----- positions -----
  int n_nodes = atom->nlocal + atom->nghost;
  auto positions = torch::empty({n_nodes,3}, torch::dtype(torch::kFloat64));
  for (int ii = 0; ii < n_nodes; ii++)
    for (int jj = 0; jj < 3; jj++)
      positions[ii][jj] = atom->x[ii][jj];

  // ----- cell -----
  auto cell = torch::zeros({3,3}, torch::dtype(torch::kFloat64));
  for (int ii = 0; ii < 3; ii++) {
    cell[ii][ii] = 50.0;
  }

  // ----- edge_index -----
  int n_edges = 0;
  for (int ii = 0; ii < list->inum; ii++)
    n_edges += list->numneigh[list->ilist[ii]];
  auto edge_index = torch::empty({2,n_edges}, torch::dtype(torch::kInt64));
  int k = 0;
  for (int ii = 0; ii < list->inum; ii++) {
    int i = list->ilist[ii];
    int *jlist = list->firstneigh[i];
    int jnum = list->numneigh[i];
    for (int jj = 0; jj < jnum; jj++) {
      edge_index[0][k] = i;
      edge_index[1][k] = jlist[jj];
      //edge_index[1,k] = (jlist[jj] & NEIGHMASK) + 1;
      k++;
    }
  }

  std::cout << "mace atomic numbers size" << mace_atomic_numbers.size() << std::endl;
  std::cout << "mace atomic numbers  " << mace_atomic_numbers << std::endl;
  auto get_mace_type = [this](int lammps_type) {
    for (int i=0; i<mace_atomic_numbers.size(); ++i) {
      std::cout << "compare " << i << ":  " << mace_atomic_numbers[i] << "  " << lammps_atomic_numbers[lammps_type-1] << std::endl;
      if (mace_atomic_numbers[i]==lammps_atomic_numbers[lammps_type-1]) {
        return i+1;
      }
    }
    // TODO: should throw error
    return -1000;
  };

  // node_attrs involves atomic numbers
  int n_node_feats = mace_atomic_numbers.size();
  auto node_attrs = torch::zeros({n_nodes,n_node_feats}, torch::dtype(torch::kFloat64));
  // TODO: generalize this
  for (int ii = 0; ii < list->inum; ii++) {

    // map lammps type to mace type
    int mace_type = get_mace_type(atom->type[ii]);
    std::cout << "mace_type " << mace_type << std::endl;
    node_attrs[ii][mace_type-1] = 1.0;
  }

  // TODO: consider from_blob to avoid copy
  auto batch = torch::zeros({n_nodes}, torch::dtype(torch::kInt64));
  auto energy = torch::empty({1}, torch::dtype(torch::kFloat64));
  auto forces = torch::empty({n_nodes,3}, torch::dtype(torch::kFloat64));
  auto ptr = torch::empty({2}, torch::dtype(torch::kInt64));
  auto shifts = torch::zeros({n_edges,3}, torch::dtype(torch::kFloat64)); //zeros instead of empty
  auto unit_shifts = torch::zeros({n_edges,3}, torch::dtype(torch::kFloat64)); //zeros instead of empty
  auto weight = torch::empty({1}, torch::dtype(torch::kFloat64));
  ptr[0] = 0;  // always zero
  ptr[1] = 3;  // always n_atoms
  weight[0] = 1.0;

  c10::Dict<std::string, torch::Tensor> input;
  input.insert("batch", batch);
  //std::cout << "batch" << std::endl;
  //std::cout << batch << std::endl;
  input.insert("cell", cell);
  //std::cout << "cell" << std::endl;
  //std::cout << cell << std::endl;
  input.insert("edge_index", edge_index);
  //std::cout << "edge_index" << std::endl;
  //std::cout << edge_index << std::endl;
  input.insert("energy", energy);
  //std::cout << "energy" << std::endl;
  //std::cout << energy << std::endl;
  input.insert("forces", forces);
  //std::cout << "forces" << std::endl;
  //std::cout << forces << std::endl;
  input.insert("node_attrs", node_attrs);
  //std::cout << "node_attrs" << std::endl;
  //std::cout << node_attrs << std::endl;
  input.insert("positions", positions);
  //std::cout << "positions" << std::endl;
  //std::cout << positions << std::endl;
  input.insert("ptr", ptr);
  //std::cout << "ptr" << std::endl;
  //std::cout << ptr << std::endl;
  input.insert("shifts", shifts);
  //std::cout << "shifts" << std::endl;
  //std::cout << shifts << std::endl;
  input.insert("unit_shifts", unit_shifts);
  //std::cout << "unit_shifts" << std::endl;
  //std::cout << unit_shifts << std::endl;
  input.insert("weight", weight);
  //std::cout << "weight" << std::endl;
  //std::cout << weight << std::endl;

  std::cout << "evaluating model" << std::endl;
  auto output = model.forward({input, true}).toGenericDict();
  energy = output.at("energy").toTensor();
  auto contributions = output.at("contributions").toTensor();
  forces = output.at("forces").toTensor();
  std::cout << energy << std::endl;
  std::cout << contributions << std::endl;
  std::cout << forces << std::endl;

  eng_vdwl = output.at("energy").toTensor()[0].item<double>();
  std::cout << "Goodbye from MACE compute." << std::endl;
}

/* ---------------------------------------------------------------------- */

void PairMACE::settings(int narg, char **arg)
{
  std::cout << "Hello from MACE settings." << std::endl;
  std::cout << "Goodbye from MACE settings." << std::endl;
}

/* ---------------------------------------------------------------------- */

void PairMACE::coeff(int narg, char **arg)
{
  std::cout << "Hello from MACE coeff." << std::endl;

  if (!allocated) allocate();

  std::cout << "Loading MACE model from \"" << arg[2] << "\" ...";
  model = torch::jit::load(arg[2]);
  std::cout << " finished." << std::endl;

  std::cout << "attributes" << std::endl;
  for (const auto& pair : model.named_attributes()) {
    //std::cout << pair.name << ": " << pair.value << std::endl;
    std::cout << pair.name << std::endl;
  }

  r_max = model.attr("r_max").toDouble();
  std::cout << "  - The r_max is: " << r_max << "." << std::endl;
  std::cout << "  xxxx atomic numbers " << model.attr("atomic_numbers") << std::endl;
  mace_atomic_numbers = model.attr("atomic_numbers").toIntVector();
  std::cout << "  - The model atomic numbers are: " << mace_atomic_numbers << "." << std::endl;
  //std::cout << "  - The atomic numbers are: " << model.attr("atomic_numbers") << "." << std::endl;

  for (int i=3; i<narg; ++i) {
    auto iter = std::find(periodic_table.begin(), periodic_table.end(), arg[i]);
    int index = std::distance(periodic_table.begin(), iter) + 1;
    lammps_atomic_numbers.push_back(index);
  }
  std::cout << "  - The pair_coeff atomic numbers are: " << lammps_atomic_numbers << "." << std::endl;

  for (int i=1; i<atom->ntypes+1; i++)
    for (int j=i; j<atom->ntypes+1; j++)
      setflag[i][j] = 1;

  std::cout << "Goodbye from MACE coeff." << std::endl;
}

void PairMACE::init_style()
{
  std::cout << "Hello from MACE init_coef." << std::endl;
  // require full neighbor list
  neighbor->add_request(this, NeighConst::REQ_FULL);
  std::cout << "Goodbye from MACE init_coef." << std::endl;
}

double PairMACE::init_one(int i, int j)
{
  // TODO: address neighbor list skin distance (2A) differently
  return model.attr("r_max").toDouble() - 2.0;
}

void PairMACE::allocate()
{
  std::cout << "Hello from MACE allocate." << std::endl;

  allocated = 1;

  memory->create(setflag, atom->ntypes+1, atom->ntypes+1, "pair:setflag");
  for (int i=1; i<atom->ntypes+1; i++)
    for (int j=i; j<atom->ntypes+1; j++)
      setflag[i][j] = 0;

  memory->create(cutsq, atom->ntypes+1, atom->ntypes+1, "pair:cutsq");
  std::cout << "WARNING: may need to overload init_one, which sets cutsq." << std::endl;

  std::cout << "Goodbye from MACE allocate." << std::endl;
}
