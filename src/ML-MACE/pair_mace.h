/* -*- c++ -*- ----------------------------------------------------------
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

#ifdef PAIR_CLASS
// clang-format off
PairStyle(mace,PairMACE);
// clang-format on
#else

#ifndef LMP_PAIR_MACE_H
#define LMP_PAIR_MACE_H

#include "pair.h"

#include <torch/script.h>

namespace LAMMPS_NS {

class PairMACE : public Pair {

 public:

  PairMACE(class LAMMPS *);
  ~PairMACE() override;
  void compute(int, int) override;
  void settings(int, char **) override;
  void coeff(int, char **) override;
  void init_style() override;
  double init_one(int, int) override;
  void allocate();

 protected:

  torch::jit::script::Module model;
  double r_max;
  int64_t num_interactions;
  std::vector<int64_t> mace_atomic_numbers;
  std::vector<int64_t> lammps_atomic_numbers;
  const std::array<std::string,10> periodic_table =
    {"H", "He",
     "Li", "Be", "B", "C", "N", "O", "F", "Ne"};

};
}    // namespace LAMMPS_NS

#endif
#endif
