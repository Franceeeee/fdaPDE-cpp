// This file is part of fdaPDE, a C++ library for physics-informed
// spatial and functional data analysis.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#include <cstddef>
#include <gtest/gtest.h>   // testing framework

#include <fdaPDE/core.h>
using fdapde::core::advection;
using fdapde::core::diffusion;
using fdapde::core::FEM;
using fdapde::core::fem_order;
using fdapde::core::laplacian;
using fdapde::core::DiscretizedMatrixField;
using fdapde::core::PDE;
using fdapde::core::DiscretizedVectorField;

#include "../../fdaPDE/models/regression/srpde.h"
#include "../../fdaPDE/models/sampling_design.h"
using fdapde::models::SRPDE;
using fdapde::models::Sampling;

#include "utils/constants.h"
#include "utils/mesh_loader.h"
#include "utils/utils.h"
using fdapde::testing::almost_equal;
using fdapde::testing::MeshLoader;
using fdapde::testing::read_csv;

// test 1
//    domain:       unit square [1,1] x [1,1]
//    sampling:     locations = nodes
//    penalization: simple laplacian
//    covariates:   no
//    BC:           no
//    order FE:     1
//    time penalization: separable (mass penalization)
TEST(mixed_srpde_test, laplacian_nonparametric_samplingatnodes_separable_monolithic) {
    // define temporal and spatial domain
    Mesh<1, 1> time_mesh(0, 2, 10);
    MeshLoader<Mesh2D> domain("unit_square_coarse");
    // import data from files
    DMatrix<double> y = read_csv<double>("../data/models/srpde/2D_test1/y.csv");
    // DMatrix<double> Wg = read_csv<double>("../data/models/mixed_srpde/2D_test1/Wg.csv"); // W: groups
    // DMatrix<double> Vp = read_csv<double>("../data/models/mixed_srpde/2D_test1/Vp.csv"); // V: patients 
    DMatrix<double> Wg = read_csv<double>("../data/models/srpde/2D_test1/y.csv"); 
    DMatrix<double> Vp = read_csv<double>("../data/models/srpde/2D_test1/y.csv"); // queste due righe sono solo per provare la run

    // define regularizing PDE in space   
    auto Ld = -laplacian<FEM>();
    DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.n_elements() * 3 * time_mesh.n_nodes(), 1);
    PDE<Mesh<2, 2>, decltype(Ld), DMatrix<double>, FEM, fem_order<1>> space_penalty(domain.mesh, Ld, u);

    // define model
    double lambda_D = 0.01, lambda_T = 0.01;
    MixedSRPDE<SpaceOnly, fdapde::monolithic> model(space_penalty, Sampling::mesh_nodes);
    model.set_lambda_D(lambda_D);

    // set model's data
    BlockFrame<double, int> df;
    df.stack(OBSERVATIONS_BLK, y);
    df.stack(DESIGN_MATRIX_BLK, Wg); 
    df.stack(DESIGN_MATRIX_BLK, Vp); // dubbio...
    model.set_data(df);

    // solve smoothing problem
    model.init();
    model.solve();

    // test correctness
    EXPECT_TRUE(almost_equal(model.f()  , "../data/models/strpde/2D_test1/sol.mtx"));
}