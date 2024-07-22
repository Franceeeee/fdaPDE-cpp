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
#include <sstream>
#include <chrono> 
#include <filesystem>

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

// test iterativo LOCAZIONI DIFFERENTI
TEST(mixed_srpde_test, cento_iter) {
    // define domain 
    std::string meshID = "c_shaped_1";
    std::string policyID = "richardson/";
    std::string locsID = "1000/";
    MeshLoader<Mesh2D> domain(meshID);
    // import data from files
    meshID = meshID + "/"; 

    std::vector<BlockFrame<double, int>> data;
    data.resize(3);
    std::cout << "data" << std::endl;
    // import data from files
    bool exist = 1;
    for(std::size_t i = 0; i<3; i++){
        
        std::string Wname = "W_" + std::to_string(i+1) + ".csv";
        std::cout << Wname << std::endl;
        std::string Vname = "V_" + std::to_string(i+1);
        std::string locsname = "locs_" + std::to_string(i+1);
        std::string yname = "observations_" + std::to_string(i+1);

        // exist = std::filesystem::exists(Wname);
        // std::cout << exist << std::endl;
        // if(!exist){
        //     break;
        // }
        // std::cout << exist << std::endl;
        // BlockFrame<double, int> df;
        data[i].read_csv<double>(W_BLOCK, "../data/models/mixed_srpde/2D_test1/" + meshID + policyID + locsID + Wname);
        data[i].read_csv<double>(V_BLOCK, "../data/models/mixed_srpde/2D_test1/" + meshID + policyID + locsID + Vname + ".csv");
        data[i].read_csv<double>(Y_BLOCK, "../data/models/mixed_srpde/2D_test1/" + meshID + policyID + locsID + yname + ".csv");
        data[i].read_csv<double>(LOCS_BLOCK, "../data/models/mixed_srpde/2D_test1/" + meshID + policyID + locsID + locsname + ".csv");
        
        // data[i] = df;
        // DMatrix<double> Wi = read_csv<double>("../data/models/mixed_srpde/2D_test1/" + meshID + policyID + locsID + Wname + ".csv");
        // Wg(i, 0) = &Wi;
        // Wg.block(i,0,1,1) = &Wi;   
        // Wg.emplace_back(DMatrix<double>*(read_csv<double>(("../data/models/mixed_srpde/2D_test1/" + meshID + policyID + locsID + Wname + ".csv"))));
        // Wg.emplace_back(&Wi);

        // DMatrix<double>* Vi = read_csv<double>("../data/models/mixed_srpde/2D_test1/" + meshID + policyID + locsID + Vname + ".csv");
        // Vp.emplace_back(&Vi);

        // DMatrix<double>* locsi = read_csv<double>("../data/models/mixed_srpde/2D_test1/" + meshID + policyID + locsID + locsname + ".csv");
        // locs.emplace_back(locsi);        

        // DMatrix<double>* yi = read_csv<double>("../data/models/mixed_srpde/2D_test1/" + meshID + policyID + locsID + yname + ".csv");
        // y.emplace_back(yi);

    }

    // DMatrix<double> locs = read_csv<double>("../data/models/mixed_srpde/2D_test1/" + 
    // 											meshID + policyID + locsID + "locations_1.csv");
    // DMatrix<double> y = read_csv<double>("../data/models/mixed_srpde/2D_test1/" + 
    // 											meshID + policyID + locsID + "observations.csv");
    // DMatrix<double> Wg = read_csv<double>("../data/models/mixed_srpde/2D_test1/" + 
    // 											meshID + policyID + locsID + "W.csv");
    // DMatrix<double> Vp = read_csv<double>("../data/models/mixed_srpde/2D_test1/" + 
    // 										    meshID + policyID + locsID + "V.csv"); 

    std::size_t sum = 0;
    for(std::size_t i=0; i<data.size(); i++){
        sum += data[i].template get<double>(LOCS_BLOCK).rows();
    }

    // define regularizing PDE
    auto L = -laplacian<FEM>();
    DMatrix<double> u = DMatrix<double>::Zero(sum, 1);
    PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> problem(domain.mesh, L, u);
    // define model
    double lambda = 1;
    
    MixedSRPDE<SpaceOnly,iterative> model(problem, Sampling::pointwise);
    model.set_lambda_D(lambda);
    // model.set_spatial_locations(locs);
    
    // set model's data
    // BlockFrame<double, int> df;
    // df.insert(OBSERVATIONS_BLK, y);
    // df.insert(MIXED_EFFECTS_BLK, Vp); 
    // df.insert(DESIGN_MATRIX_BLK, Wg);
    model.set_data(data);
    // model.set_psi();
    
    // solve smoothing problem
    model.init();
    
    model.solve();
    
    DMatrix<double> f_estimate = read_csv<double>("../data/models/mixed_srpde/2D_test1/" + 
    												 meshID + policyID + locsID + "f_hat.csv");
    
    std::cout << "Iterative: " << (model.f() - f_estimate ).array().abs().maxCoeff() << std::endl;
    EXPECT_TRUE(  (model.f() - f_estimate ).array().abs().maxCoeff() < 1e-6 );
}



/*
// PARAMETRIC TEST 
struct TestParams {
    std::string meshID;
    std::string locsID;
    std::string policyID;
    std::string patientsN;
};

const std::vector<TestParams> kPets = {
     
     {"c_shaped_4_patients", "100", "richardson"},
     {"c_shaped_4_patients", "100", "monolithic"},
     {"c_shaped_4_patients", "250", "richardson"},
     {"c_shaped_4_patients", "250", "monolithic"},
     {"c_shaped_4_patients", "500", "richardson"},
     {"c_shaped_4_patients", "500", "monolithic"},
     {"c_shaped_4_patients", "1000", "richardson"},
     {"c_shaped_4_patients", "1000", "monolithic"},
     {"c_shaped_4_patients", "5000", "richardson"},
     {"c_shaped_4_patients", "5000", "monolithic"},
     
     
     {"c_shaped_4_patients", "10000", "richardson", "24"},
     {"c_shaped_4_patients", "10000", "richardson", "36"},
     {"c_shaped_4_patients", "10000", "richardson", "60"},
     {"c_shaped_4_patients", "10000", "richardson", "75"},
     {"c_shaped_4_patients", "10000", "richardson", "90"},
     {"c_shaped_4_patients", "10000", "monolithic", "24"},
     {"c_shaped_4_patients", "10000", "monolithic", "36"},
     {"c_shaped_4_patients", "10000", "monolithic", "60"},
     {"c_shaped_4_patients", "10000", "monolithic", "75"},
     {"c_shaped_4_patients", "10000", "monolithic", "90"}
    
    
    {"c_shaped_5_patients", "10000", "richardson", "24"},
    {"c_shaped_5_patients", "10000", "richardson", "36"},
    {"c_shaped_5_patients", "10000", "richardson", "60"},
    {"c_shaped_5_patients", "10000", "richardson", "75"},
    {"c_shaped_5_patients", "10000", "richardson", "90"},
    {"c_shaped_5_patients", "10000", "richardson", "150"},
    {"c_shaped_5_patients", "10000", "monolithic", "24"},
    {"c_shaped_5_patients", "10000", "monolithic", "36"},
    {"c_shaped_5_patients", "10000", "monolithic", "60"},
    {"c_shaped_5_patients", "10000", "monolithic", "75"},
    {"c_shaped_5_patients", "10000", "monolithic", "90"},
    {"c_shaped_5_patients", "10000", "monolithic", "150"}
    
};

// from https://github.com/google/googletest/blob/main/docs/advanced.md
class MixedSRPDETest : public testing::TestWithParam<TestParams> {
  // You can implement all the usual fixture class members here.
  // To access the test parameter, call GetParam() from class TestWithParam<T>.
};


INSTANTIATE_TEST_SUITE_P(Pets, MixedSRPDETest, testing::ValuesIn(kPets));

// test with repeated patients
TEST_P(MixedSRPDETest, Testing) {
    
    TestParams params = GetParam();
    std::stringstream ss;

    ss << params.meshID << " " << params.locsID << " " << params.policyID << " " << params.patientsN;
    
    std::string filename = "prova_rep_4.txt";
    
    // meshID locsID policyID Npatients rmse execution_time beta1 beta2
    
    // Use params.meshID, params.locsID, and params.policyID in your test
    std::string meshID = params.meshID;
    std::string locsID = params.locsID;
    std::string policyID = params.policyID;
    std::string patientsN = params.patientsN;
    auto start = std::chrono::high_resolution_clock::now();

    policyID = policyID + "/"; 
    int npat = std::stoi(patientsN);
    patientsN = patientsN + "/"; 
    locsID = locsID + "/"; 

    // define domain 
    // std::string meshID = "c_shaped_1";
    // std::string locsID = "1000/";
    // std::string policyID = "richardson/";

    if(policyID=="richardson/"){
        
        MeshLoader<Mesh2D> domain("c_shaped_4");
        
        meshID = meshID + "/"; 

        // import data from files        
        DMatrix<double> locs = read_csv<double>("../data/models/mixed_srpde/2D_test1/" + 
                                                    meshID + patientsN + policyID + locsID + "locations_1.csv");
        DMatrix<double> y = read_csv<double>("../data/models/mixed_srpde/2D_test1/" + 
                                                    meshID + patientsN + policyID + locsID + "observations_repeated.csv");
        DMatrix<double> Wg = read_csv<double>("../data/models/mixed_srpde/2D_test1/" + 
                                                    meshID + patientsN + policyID + locsID + "W_repeated.csv");
        DMatrix<double> Vp = read_csv<double>("../data/models/mixed_srpde/2D_test1/" + 
                                                    meshID + patientsN + policyID + locsID + "V_repeated.csv"); 
        
        // define regularizing PDE
        auto L = -laplacian<FEM>();
        DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.n_elements() * 3 * npat, 1);
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
        std::cout << ":-)" << std::endl;
        DMatrix<double> f_estimate = read_csv<double>("../data/models/mixed_srpde/2D_test1/" +
                                                        meshID + patientsN + policyID + locsID + "f_hat_repeated.csv");
		std::cout << f_estimate.rows() << std::endl;
		std::cout << model.f().rows() << std::endl;
		
        // DMatrix<double> f_true = read_csv<double>("../data/models/mixed_srpde/2D_test1/100/f.csv");

        // std::cout << f_estimate.rows() << " "<< f_estimate.cols() << std::endl;
        // std::cout << f_true.rows() << " "<< f_true.cols() << std::endl;
        
        // test correctness
        // std::cout <<  (f_true - f_estimate).lpNorm<Eigen::Infinity>() << std::endl;
        // std::cout <<  (model.f() - f_true).lpNorm<Eigen::Infinity>() << std::endl;
        std::cout << (model.f() - f_estimate ).array().abs().maxCoeff() <<std::endl;
        EXPECT_TRUE(  0. < 1e-6 );
        
        auto end = std::chrono::high_resolution_clock::now();

        auto differences = (model.f() - f_estimate).array().square().mean();

        auto rmse = std::sqrt(differences);

        std::chrono::duration<double> duration = end - start;
        ss << " " << rmse << " " << duration.count() << " " << model.beta()[0] << " " << model.beta()[1] << std::endl;

        // Append to a text file
        std::ofstream outputFile;
        outputFile.open(filename, std::ios_base::app);
        if (outputFile.is_open()) {
            outputFile << ss.str(); 
            outputFile.close();
        } else {
            std::cerr << "Unable to open file for writing" << std::endl;
        }
    } else if (policyID=="monolithic/"){
        MeshLoader<Mesh2D> domain("c_shaped_4");
        meshID = meshID + "/"; 
        // import data from files
        DMatrix<double> locs = read_csv<double>("../data/models/mixed_srpde/2D_test1/" + 
                                                    meshID + patientsN + policyID + locsID + "locations_1.csv");
        DMatrix<double> y = read_csv<double>("../data/models/mixed_srpde/2D_test1/" + 
                                                    meshID + patientsN + policyID + locsID + "observations_repeated.csv");
        DMatrix<double> Wg = read_csv<double>("../data/models/mixed_srpde/2D_test1/" + 
                                                    meshID + patientsN + policyID + locsID + "W_repeated.csv");
        DMatrix<double> Vp = read_csv<double>("../data/models/mixed_srpde/2D_test1/" + 
                                                    meshID + patientsN + policyID + locsID + "V_repeated.csv"); 
        // define regularizing PDE
        auto L = -laplacian<FEM>();
        DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.n_elements() * 3 *npat, 1);
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
        
        DMatrix<double> f_estimate = read_csv<double>("../data/models/mixed_srpde/2D_test1/" 
                                                        + meshID + patientsN + policyID + locsID + "f_hat_repeated.csv");

        // DMatrix<double> f_true = read_csv<double>("../data/models/mixed_srpde/2D_test1/100/f.csv");

        // std::cout << f_estimate.rows() << " "<< f_estimate.cols() << std::endl;
        // std::cout << f_true.rows() << " "<< f_true.cols() << std::endl;
        
        // test correctness
        // std::cout <<  (f_true - f_estimate).lpNorm<Eigen::Infinity>() << std::endl;
        // std::cout <<  (model.f() - f_true).lpNorm<Eigen::Infinity>() << std::endl;
        
        EXPECT_TRUE(  (model.f() - f_estimate ).array().abs().maxCoeff() < 1e-6 );
        
        auto end = std::chrono::high_resolution_clock::now();

         auto differences = (model.f() - f_estimate).array().square().mean();

        auto rmse = std::sqrt(differences);

        std::chrono::duration<double> duration = end - start;
        ss << " " << rmse << " " << duration.count() << " " << model.beta()[0] << " " << model.beta()[1] << std::endl;

        // Append to a text file
        std::ofstream outputFile;
        outputFile.open(filename, std::ios_base::app);
        if (outputFile.is_open()) {
            outputFile << ss.str(); 
            outputFile.close();
        } else {
            std::cerr << "Unable to open file for writing" << std::endl;
        }
    }
}
*/

// // test 1
// TEST_P(MixedSRPDETest, Testing) {

//     TestParams params = GetParam();
//     std::stringstream ss;

//     ss << params.meshID << " " << params.locsID << " " << params.policyID;
    
//     std::string filename = "mesh5_beta.txt";
    
//     // meshID locsID policyID rmse execution_time beta1 beta2
    
//     // Use params.meshID, params.locsID, and params.policyID in your test
//     std::string meshID = params.meshID;
//     std::string locsID = params.locsID;
//     std::string policyID = params.policyID;
//     auto start = std::chrono::high_resolution_clock::now();

//     policyID = policyID + "/"; 
//     locsID = locsID + "/"; 

//     // define domain 
//     // std::string meshID = "c_shaped_1";
//     // std::string locsID = "1000/";
//     // std::string policyID = "richardson/";

//     if(policyID=="richardson/"){
//         MeshLoader<Mesh2D> domain(meshID);
//         meshID = meshID + "/"; 
//         // import data from files
//         DMatrix<double> locs = read_csv<double>("../data/models/mixed_srpde/2D_test1/" + 
//                                                     meshID + policyID + locsID + "locations_1.csv");
//         DMatrix<double> y = read_csv<double>("../data/models/mixed_srpde/2D_test1/" + 
//                                                     meshID + policyID + locsID + "observations.csv");
//         DMatrix<double> Wg = read_csv<double>("../data/models/mixed_srpde/2D_test1/" + 
//                                                     meshID + policyID + locsID + "W.csv");
//         DMatrix<double> Vp = read_csv<double>("../data/models/mixed_srpde/2D_test1/" + 
//                                                     meshID + policyID + locsID + "V.csv"); 
//         // define regularizing PDE
//         auto L = -laplacian<FEM>();
//         DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.n_elements() * 3, 1);
//         PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> problem(domain.mesh, L, u);

//         // define model
//         double lambda = 1;

//         MixedSRPDE<SpaceOnly,iterative> model(problem, Sampling::pointwise);
        
//         model.set_lambda_D(lambda);
//         model.set_spatial_locations(locs);
        
//         // set model's data
//         BlockFrame<double, int> df;
//         df.insert(OBSERVATIONS_BLK, y);
//         df.insert(MIXED_EFFECTS_BLK, Vp); 
//         df.insert(DESIGN_MATRIX_BLK, Wg);
//         model.set_data(df);
        
//         // solve smoothing problem
//         model.init();
        
//         model.solve();
        
//         DMatrix<double> f_estimate = read_csv<double>("../data/models/mixed_srpde/2D_test1/" 
//                                                         + meshID + policyID + locsID + "f_hat.csv");

//         // DMatrix<double> f_true = read_csv<double>("../data/models/mixed_srpde/2D_test1/100/f.csv");

//         // std::cout << f_estimate.rows() << " "<< f_estimate.cols() << std::endl;
//         // std::cout << f_true.rows() << " "<< f_true.cols() << std::endl;
        
//         // test correctness
//         // std::cout <<  (f_true - f_estimate).lpNorm<Eigen::Infinity>() << std::endl;
//         // std::cout <<  (model.f() - f_true).lpNorm<Eigen::Infinity>() << std::endl;
        
//         EXPECT_TRUE(  (model.f() - f_estimate ).array().abs().maxCoeff() < 1e-6 );
        
//         auto end = std::chrono::high_resolution_clock::now();

//         auto differences = (model.f() - f_estimate).array().square().mean();

//         auto rmse = std::sqrt(differences);

//         std::chrono::duration<double> duration = end - start;
//         ss << " " << rmse << " " << duration.count() << " " << model.beta()[0] << " " << model.beta()[1] << std::endl;

//         // Append to a text file
//         std::ofstream outputFile;
//         outputFile.open(filename, std::ios_base::app);
//         if (outputFile.is_open()) {
//             outputFile << ss.str(); 
//             outputFile.close();
//         } else {
//             std::cerr << "Unable to open file for writing" << std::endl;
//         }
//     } else if (policyID=="monolithic/"){
//         MeshLoader<Mesh2D> domain(meshID);
//         meshID = meshID + "/"; 
//         // import data from files
//         DMatrix<double> locs = read_csv<double>("../data/models/mixed_srpde/2D_test1/" + 
//                                                     meshID + policyID + locsID + "locations_1.csv");
//         DMatrix<double> y = read_csv<double>("../data/models/mixed_srpde/2D_test1/" + 
//                                                     meshID + policyID + locsID + "observations.csv");
//         DMatrix<double> Wg = read_csv<double>("../data/models/mixed_srpde/2D_test1/" + 
//                                                     meshID + policyID + locsID + "W.csv");
//         DMatrix<double> Vp = read_csv<double>("../data/models/mixed_srpde/2D_test1/" + 
//                                                     meshID + policyID + locsID + "V.csv"); 
//         // define regularizing PDE
//         auto L = -laplacian<FEM>();
//         DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.n_elements() * 3, 1);
//         PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> problem(domain.mesh, L, u);

//         // define model
//         double lambda = 1;

//         MixedSRPDE<SpaceOnly,monolithic> model(problem, Sampling::pointwise);
        
//         model.set_lambda_D(lambda);
//         model.set_spatial_locations(locs);
        
//         // set model's data
//         BlockFrame<double, int> df;
//         df.insert(OBSERVATIONS_BLK, y);
//         df.insert(MIXED_EFFECTS_BLK, Vp); 
//         df.insert(DESIGN_MATRIX_BLK, Wg);
//         model.set_data(df);
        
//         // solve smoothing problem
//         model.init();
        
//         model.solve();
        
//         DMatrix<double> f_estimate = read_csv<double>("../data/models/mixed_srpde/2D_test1/" 
//                                                         + meshID + policyID + locsID + "f_hat.csv");

//         // DMatrix<double> f_true = read_csv<double>("../data/models/mixed_srpde/2D_test1/100/f.csv");

//         // std::cout << f_estimate.rows() << " "<< f_estimate.cols() << std::endl;
//         // std::cout << f_true.rows() << " "<< f_true.cols() << std::endl;
        
//         // test correctness
//         // std::cout <<  (f_true - f_estimate).lpNorm<Eigen::Infinity>() << std::endl;
//         // std::cout <<  (model.f() - f_true).lpNorm<Eigen::Infinity>() << std::endl;
        
//         EXPECT_TRUE(  (model.f() - f_estimate ).array().abs().maxCoeff() < 1e-6 );
        
//         auto end = std::chrono::high_resolution_clock::now();

//          auto differences = (model.f() - f_estimate).array().square().mean();

//         auto rmse = std::sqrt(differences);

//         std::chrono::duration<double> duration = end - start;
//         ss << " " << rmse << " " << duration.count() << " " << model.beta()[0] << " " << model.beta()[1] << std::endl;

//         // Append to a text file
//         std::ofstream outputFile;
//         outputFile.open(filename, std::ios_base::app);
//         if (outputFile.is_open()) {
//             outputFile << ss.str(); 
//             outputFile.close();
//         } else {
//             std::cerr << "Unable to open file for writing" << std::endl;
//         }
//     }
// }


/*
// test monolitico
TEST(mixed_srpde_test, cento_mono) {

    // define domain 
    std::string meshID = "c_shaped_1";
    std::string policyID = "monolithic/";
    std::string locsID = "1000/";
    MeshLoader<Mesh2D> domain(meshID);
    // import data from files
    meshID = meshID + "/"; 
    // import data from files
    DMatrix<double> locs = read_csv<double>("../data/models/mixed_srpde/2D_test1/" + 
    											meshID + policyID + locsID + "locations_1.csv");
    DMatrix<double> y = read_csv<double>("../data/models/mixed_srpde/2D_test1/" + 
    											meshID + policyID + locsID + "observations.csv");
    DMatrix<double> Wg = read_csv<double>("../data/models/mixed_srpde/2D_test1/" + 
    											meshID + policyID + locsID + "W.csv");
    DMatrix<double> Vp = read_csv<double>("../data/models/mixed_srpde/2D_test1/" + 
    										    meshID + policyID + locsID + "V.csv"); 

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
    
    DMatrix<double> f_estimate = read_csv<double>("../data/models/mixed_srpde/2D_test1/" + 
    												 meshID + policyID + locsID + "f_hat.csv");
    
    std::cout << "Monolithic: " << (model.f() - f_estimate ).array().abs().maxCoeff() << std::endl;
    EXPECT_TRUE(  (model.f() - f_estimate ).array().abs().maxCoeff() < 1e-6 );
}

// test iterativo
TEST(mixed_srpde_test, cento_iter) {
    // define domain 
    std::string meshID = "c_shaped_1";
    std::string policyID = "richardson/";
    std::string locsID = "1000/";
    MeshLoader<Mesh2D> domain(meshID);
    // import data from files
    meshID = meshID + "/"; 
    // import data from files
    DMatrix<double> locs = read_csv<double>("../data/models/mixed_srpde/2D_test1/" + 
    											meshID + policyID + locsID + "locations_1.csv");
    DMatrix<double> y = read_csv<double>("../data/models/mixed_srpde/2D_test1/" + 
    											meshID + policyID + locsID + "observations.csv");
    DMatrix<double> Wg = read_csv<double>("../data/models/mixed_srpde/2D_test1/" + 
    											meshID + policyID + locsID + "W.csv");
    DMatrix<double> Vp = read_csv<double>("../data/models/mixed_srpde/2D_test1/" + 
    										    meshID + policyID + locsID + "V.csv"); 

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
    
    DMatrix<double> f_estimate = read_csv<double>("../data/models/mixed_srpde/2D_test1/" + 
    												 meshID + policyID + locsID + "f_hat.csv");
    
    std::cout << "Iterative: " << (model.f() - f_estimate ).array().abs().maxCoeff() << std::endl;
    EXPECT_TRUE(  (model.f() - f_estimate ).array().abs().maxCoeff() < 1e-6 );
}
*/

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
