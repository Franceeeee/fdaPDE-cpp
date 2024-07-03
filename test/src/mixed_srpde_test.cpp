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
#include <fstream>

#include <fdaPDE/core.h>
using fdapde::core::advection;
using fdapde::core::diffusion;
using fdapde::core::FEM;
using fdapde::core::fem_order;
using fdapde::core::laplacian;
using fdapde::core::DiscretizedMatrixField;
using fdapde::core::PDE;
using fdapde::core::DiscretizedVectorField;

#include "../../fdaPDE/models/regression/mixed_srpde.h"
#include "../../fdaPDE/models/sampling_design.h"
using fdapde::models::MixedSRPDE;
using fdapde::models::Sampling;
using fdapde::models::SpaceOnly;
using fdapde::monolithic;
using fdapde::iterative;

#include "utils/constants.h"
#include "utils/mesh_loader.h"
#include "utils/utils.h"
using fdapde::testing::almost_equal;
using fdapde::testing::MeshLoader;
using fdapde::testing::read_csv;

// test 1
TEST(mixed_srpde_test, cento) {
    // define domain 
    MeshLoader<Mesh2D> domain("c_shaped");
    // import data from files
    DMatrix<double> locs = read_csv<double>("../data/models/mixed_srpde/2D_test1/100/locations_1.csv");
    DMatrix<double> y = read_csv<double>("../data/models/mixed_srpde/2D_test1/100/observations.csv");
    DMatrix<double> Wg = read_csv<double>("../data/models/mixed_srpde/2D_test1/100/W.csv");
    DMatrix<double> Vp = read_csv<double>("../data/models/mixed_srpde/2D_test1/100/V.csv"); 

    // define regularizing PDE
    auto L = -laplacian<FEM>();
    DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.n_elements() * 3, 1);
    PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> problem(domain.mesh, L, u);

    // define model
    double lambda = 1;
    
    MixedSRPDE<SpaceOnly,iterative> model(problem, Sampling::pointwise);
    model.set_lambda_D(lambda);
    model.set_spatial_locations(locs);
    
    // set model's data
    BlockFrame<double, int> df;
    df.insert(OBSERVATIONS_BLK, y);
    df.insert(MIXED_EFFECTS_BLK, Vp); 
    df.insert(DESIGN_MATRIX_BLK, Wg);
    model.set_data(df);
    
    // solve smoothing problem
    model.init();
    
    model.solve();
    
    DMatrix<double> f_estimate = read_csv<double>("../data/models/mixed_srpde/2D_test1/100/f_hat.csv");
    // DMatrix<double> f_true = read_csv<double>("../data/models/mixed_srpde/2D_test1/100/f.csv");

    // std::cout << f_estimate.rows() << " "<< f_estimate.cols() << std::endl;
    // std::cout << f_true.rows() << " "<< f_true.cols() << std::endl;
    
    // test correctness
    // std::cout <<  (f_true - f_estimate).lpNorm<Eigen::Infinity>() << std::endl;
    // std::cout <<  (model.f() - f_true).lpNorm<Eigen::Infinity>() << std::endl;
    std::ofstream output("f_100.csv");
    output << "Results for test with 100 observations\n";
    DMatrix<double> data = model.f();
    for(std::size_t i = 0; i < data.size(); ++i){
        output << data(i) << "\n";
    }
    output.close();
    std::cout << "Numerical results saved in f_100.csv" << std::endl;
    std::cout << model.beta().rows() << std::endl;
    std::cout << model.beta() << std::endl;
    
    EXPECT_TRUE(  (model.f() - f_estimate ).array().abs().maxCoeff() < 1e-6 );
}

// test 1
TEST(mixed_srpde_test, cento_mono) {
    // define domain 
    MeshLoader<Mesh2D> domain("c_shaped");
    // import data from files
    DMatrix<double> locs = read_csv<double>("../data/models/mixed_srpde/2D_test1/100/locations_1.csv");
    DMatrix<double> y = read_csv<double>("../data/models/mixed_srpde/2D_test1/100/observations.csv");
    DMatrix<double> Wg = read_csv<double>("../data/models/mixed_srpde/2D_test1/100/W.csv");
    DMatrix<double> Vp = read_csv<double>("../data/models/mixed_srpde/2D_test1/100/V.csv"); 

    // define regularizing PDE
    auto L = -laplacian<FEM>();
    DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.n_elements() * 3, 1);
    PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> problem(domain.mesh, L, u);

    // define model
    double lambda = 1;
    
    MixedSRPDE<SpaceOnly,monolithic> model(problem, Sampling::pointwise);
    model.set_lambda_D(lambda);
    model.set_spatial_locations(locs);
    
    // set model's data
    BlockFrame<double, int> df;
    df.insert(OBSERVATIONS_BLK, y);
    df.insert(MIXED_EFFECTS_BLK, Vp); 
    df.insert(DESIGN_MATRIX_BLK, Wg);
    model.set_data(df);
    
    // solve smoothing problem
    model.init();
    
    model.solve();
    
    DMatrix<double> f_estimate = read_csv<double>("../data/models/mixed_srpde/2D_test1/100/f_hat.csv");
    // DMatrix<double> f_true = read_csv<double>("../data/models/mixed_srpde/2D_test1/100/f.csv");

    // std::cout << f_estimate.rows() << " "<< f_estimate.cols() << std::endl;
    // std::cout << f_true.rows() << " "<< f_true.cols() << std::endl;
    
    // test correctness
    // std::cout <<  (f_true - f_estimate).lpNorm<Eigen::Infinity>() << std::endl;
    // std::cout <<  (model.f() - f_true).lpNorm<Eigen::Infinity>() << std::endl;
    std::ofstream output("f_100.csv");
    output << "Results for test with 100 observations\n";
    DMatrix<double> data = model.f();
    for(std::size_t i = 0; i < data.size(); ++i){
        output << data(i) << "\n";
    }
    output.close();
    std::cout << "Numerical results saved in f_100.csv" << std::endl;
    
    EXPECT_TRUE(  (model.f() - f_estimate ).array().abs().maxCoeff() < 1e-6 );
}

/*
// test 2
TEST(mixed_srpde_test, duecentocinquanta) {
    // define domain 
    MeshLoader<Mesh2D> domain("c_shaped");
    // import data from files
    DMatrix<double> locs = read_csv<double>("../data/models/mixed_srpde/2D_test1/250/locations_1.csv");
    DMatrix<double> y = read_csv<double>("../data/models/mixed_srpde/2D_test1/250/observations.csv");
    DMatrix<double> Wg = read_csv<double>("../data/models/mixed_srpde/2D_test1/250/W.csv");
    DMatrix<double> Vp = read_csv<double>("../data/models/mixed_srpde/2D_test1/250/V.csv"); 

    // define regularizing PDE
    auto L = -laplacian<FEM>();
    DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.n_elements() * 3, 1);
    PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> problem(domain.mesh, L, u);
        
    // define model
    double lambda = 1;
    
    MixedSRPDE<SpaceOnly, iterative> model(problem, Sampling::pointwise);
    model.set_lambda_D(lambda);
    model.set_spatial_locations(locs);
    
    BlockFrame<double, int> df;
    df.insert(OBSERVATIONS_BLK, y);
    df.insert(MIXED_EFFECTS_BLK, Vp); 
    df.insert(DESIGN_MATRIX_BLK, Wg);
    model.set_data(df);
    
    // solve smoothing problem
    model.init();
    model.solve();
    
    DMatrix<double> f_estimate = read_csv<double>("../data/models/mixed_srpde/2D_test1/250/f_hat.csv");
    // DMatrix<double> f_true = read_csv<double>("../data/models/mixed_srpde/2D_test1/250/f.csv");

    std::ofstream output("f_250.csv");
    output << "Results for test with 250 observations\n";
    DMatrix<double> data = model.f();
    for(std::size_t i = 0; i < data.size(); ++i){
        output << data(i) << "\n";
    }
    output.close();
    std::cout << "Numerical results saved in f_250.csv" << std::endl;

    //EXPECT_TRUE(almost_equal(model.f(), f_estimate));
    EXPECT_TRUE(  (model.f() - f_estimate ).array().abs().maxCoeff() < 1e-6 );
}

// test 3
TEST(mixed_srpde_test, cinquecento) {
    // define domain 
    MeshLoader<Mesh2D> domain("c_shaped");
    // import data from files
    DMatrix<double> locs = read_csv<double>("../data/models/mixed_srpde/2D_test1/500/locations_1.csv");
    DMatrix<double> y = read_csv<double>("../data/models/mixed_srpde/2D_test1/500/observations.csv");
    DMatrix<double> Wg = read_csv<double>("../data/models/mixed_srpde/2D_test1/500/W.csv");
    DMatrix<double> Vp = read_csv<double>("../data/models/mixed_srpde/2D_test1/500/V.csv"); 

    // define regularizing PDE
    auto L = -laplacian<FEM>();
    DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.n_elements() * 3, 1);
    PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> problem(domain.mesh, L, u);
        
    // define model
    double lambda = 1;
    
    MixedSRPDE<SpaceOnly, iterative> model(problem, Sampling::pointwise);
    model.set_lambda_D(lambda);
    model.set_spatial_locations(locs);
    
    BlockFrame<double, int> df;
    df.insert(OBSERVATIONS_BLK, y);
    df.insert(MIXED_EFFECTS_BLK, Vp); 
    df.insert(DESIGN_MATRIX_BLK, Wg);
    model.set_data(df);
    
    // solve smoothing problem
    model.init();
    model.solve();
    
    DMatrix<double> f_estimate = read_csv<double>("../data/models/mixed_srpde/2D_test1/500/f_hat.csv");
    // DMatrix<double> f_true = read_csv<double>("../data/models/mixed_srpde/2D_test1/500/f.csv");

    std::ofstream output("f_500.csv");
    output << "Results for test with 500 observations\n";
    DMatrix<double> data = model.f();
    for(std::size_t i = 0; i < data.size(); ++i){
        output << data(i) << "\n";
    }
    output.close();
    std::cout << "Numerical results saved in f_500.csv" << std::endl;

    //EXPECT_TRUE(almost_equal(model.f(), f_estimate));
    EXPECT_TRUE(  (model.f() - f_estimate ).array().abs().maxCoeff() < 1e-6 );
}

// test 4
TEST(mixed_srpde_test, mille) {
    // define domain 
    MeshLoader<Mesh2D> domain("c_shaped");
    // import data from files
    DMatrix<double> locs = read_csv<double>("../data/models/mixed_srpde/2D_test1/1000/locations_1.csv");
    DMatrix<double> y = read_csv<double>("../data/models/mixed_srpde/2D_test1/1000/observations.csv");
    DMatrix<double> Wg = read_csv<double>("../data/models/mixed_srpde/2D_test1/1000/W.csv");
    DMatrix<double> Vp = read_csv<double>("../data/models/mixed_srpde/2D_test1/1000/V.csv"); 

    // define regularizing PDE
    auto L = -laplacian<FEM>();
    DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.n_elements() * 3, 1);
    PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> problem(domain.mesh, L, u);
        
    // define model
    double lambda = 1;
    
    MixedSRPDE<SpaceOnly, iterative> model(problem, Sampling::pointwise);
    model.set_lambda_D(lambda);
    model.set_spatial_locations(locs);
    
    BlockFrame<double, int> df;
    df.insert(OBSERVATIONS_BLK, y);
    df.insert(MIXED_EFFECTS_BLK, Vp); 
    df.insert(DESIGN_MATRIX_BLK, Wg);
    model.set_data(df);
    
    // solve smoothing problem
    model.init();
    model.solve();
    
    DMatrix<double> f_estimate = read_csv<double>("../data/models/mixed_srpde/2D_test1/1000/f_hat.csv");
    // DMatrix<double> f_true = read_csv<double>("../data/models/mixed_srpde/2D_test1/1000/f.csv");

    std::ofstream output("f_1000.csv");
    output << "Results for test with 1000 observations\n";
    DMatrix<double> data = model.f();
    for(std::size_t i = 0; i < data.size(); ++i){
        output << data(i) << "\n";
    }
    output.close();
    std::cout << "Numerical results saved in f_1000.csv" << std::endl;

    //EXPECT_TRUE(almost_equal(model.f(), f_estimate));
    EXPECT_TRUE(  (model.f() - f_estimate ).array().abs().maxCoeff() < 1e-6 );
}
*/
