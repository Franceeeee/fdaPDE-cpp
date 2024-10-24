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

#include <iostream>
#include <vector>
#include <string>
#include <random>
#include <cmath>
#include <cstddef>
#include <gtest/gtest.h>   // testing framework
#include <fstream>
#include <sstream>
#include <chrono> 
#include <filesystem>
#include <limits>


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
#include<filesystem>
using fdapde::testing::almost_equal;
using fdapde::testing::MeshLoader;
using fdapde::testing::read_csv;

// I/O utils 
template <typename T> DMatrix<T> read_mtx(const std::string& file_name) {
    SpMatrix<T> buff;
    Eigen::loadMarket(buff, file_name);
    return buff;
}

template<typename T> void eigen2ext(const DMatrix<T>& M, const std::string& sep, const std::string& filename, bool append = false){
    std::ofstream file;

    if(!append) 
        file.open(filename);
    else
        file.open(filename, std::ios_base::app); 
    
    for(std::size_t i = 0; i < M.rows(); ++i) {
            for(std::size_t j=0; j < M.cols()-1; ++j) file << M(i,j) << sep;
            file << M(i, M.cols()-1) <<  "\n";  
    }
    file.close();
}

template<typename T> void eigen2txt(const DMatrix<T>& M, const std::string& filename = "mat.txt", bool append = false){
    eigen2ext<T>(M, " ", filename, append);
}

template<typename T> void eigen2csv(const DMatrix<T>& M, const std::string& filename = "mat.csv", bool append = false){
    eigen2ext<T>(M, ",", filename, append);
}

template< typename T> void vector2ext(const std::vector<T>& V, const std::string& sep, const std::string& filename, bool append = false){
    std::ofstream file;

    if(!append) 
        file.open(filename);
    else
        file.open(filename, std::ios_base::app);
    
    for(std::size_t i = 0; i < V.size()-1; ++i) file << V[i] << sep;
    
    file << V[V.size()-1] << "\n";  
    
    file.close();
}

template< typename T> void vector2txt(const std::vector<T>& V, const std::string& filename = "vec.txt", bool append = false){
   vector2ext<T>(V, " ", filename, append);
}

template< typename T> void vector2csv(const std::vector<T>& V, const std::string& filename = "vec.csv", bool append = false){
   vector2ext<T>(V, ",", filename, append);
}

void write_table(const DMatrix<double>& M, const std::vector<std::string>& header = {}, const std::string& filename = "data.txt"){

    std::ofstream file(filename);

    if(header.empty() || header.size() != M.cols()){
        std::vector<std::string> head(M.cols());
        for(std::size_t i = 0; i < M.cols(); ++i)
                head[i] =  "V" + std::to_string(i);
        vector2txt<std::string>(head, filename);    
    }else vector2txt<std::string>(header, filename);
    
    eigen2txt<double>(M, filename, true);
}

void write_csv(const DMatrix<double>& M, const std::vector<std::string>& header = {}, const std::string& filename = "data.csv"){
    std::ofstream file(filename);

    if(header.empty() || header.size() != M.cols()){
        std::vector<std::string> head(M.cols());
        for(std::size_t i = 0; i < M.cols(); ++i)
                head[i] =  "V" + std::to_string(i);
        vector2csv(head, filename);    
    }else vector2csv(header, filename);
    
    eigen2csv<double>(M, filename, true);
}

// 

// tests 

TEST(mixed_srpde_test, utils){
    DMatrix<double> data = DMatrix<double>::Zero(3,5);

    data.row(0) << 1.0, 2.0, 3.0, 4.0, 5.0;
    data.row(1) << 5.1, 4.2, 3.3, 2.4, 1.5;
    data.row(2) << 3.0, 1.0, 4.0, 5.0, 6.0; // :-)

    write_table(data);
    write_table(data, {"pippo", "pluto", "paperino", "topolino", "minnie"}, "disney.txt");
    write_csv(data);
    write_csv(data, {"pippo", "pluto", "paperino", "topolino", "minnie"}, "disney.csv");

    eigen2txt(data, "disney_mat.txt");
    eigen2csv(data, "disney_mat.csv");
    
    EXPECT_TRUE(1);
}


TEST(mixed_srpde_test, same_locations_test_1) {
    std::size_t m = 3;
    std::size_t n_sim = 30;
    // 
    DMatrix<double> beta = DMatrix<double>::Zero(2,1);
    beta(0,0) = -2.; beta(1,0) = 1.;

    DMatrix<double> alpha = DMatrix<double>::Zero(3,1);
    alpha(0,0) = -0.5; alpha(1,0) = 0.; alpha(2,0) = 0.5;

    DMatrix<int> n_obs = DMatrix<int>::Zero(5,1);
    n_obs(0,0) = 500; n_obs(1,0) = 1000; n_obs(2,0) = 2000; 
    n_obs(3,0) = 4000; n_obs(4,0) = 8000;
    
    int seed = 0; 
    std::mt19937 gen(seed);
    
    auto uniform_locs = [&gen](std::size_t n) {
        std::uniform_real_distribution<> dis(0.0, 1.0);
        DMatrix<double> locs = DMatrix<double>::Zero(n,2);
        for (std::size_t i = 0; i < n; ++i) {
            locs(i,0) = dis(gen);  // x
            locs(i,1) = dis(gen);  // y
        }
        return locs;
    };

    auto f = [](DMatrix<double> locs, int id = 0) { 
	    DMatrix<double> res = DMatrix<double>::Zero(locs.rows(),1);
            for(std::size_t i = 0; i < locs.rows(); ++i){
                if(id == 0)
                    res(i,0) = std::sin(2*fdapde::testing::pi*locs(i,0))*
                                    std::sin(2*fdapde::testing::pi*locs(i,1));
                else if (id == 1)
                    res(i,0) = 1.0 - locs(i,0) - locs(i,1);
                else if (id == 2)
                    res(i,0) = 1-std::sin(fdapde::testing::pi*locs(i,0))*
                                    std::cos(fdapde::testing::pi*locs(i,1));//std::cos(fdapde::testing::pi*locs(i,0))*std::cos(fdapde::testing::pi*locs(i,1));
        }
            return res;
    };
    
    auto noise = [&gen](std::size_t n, double sigma){
        DMatrix<double> res = DMatrix<double>::Zero(n,1);
        std::normal_distribution<> __noise(0.0, sigma);
        for(std::size_t i = 0; i < n; ++i){
            res(i,0) = __noise(gen);
        }
        return res;
    };

    auto x1_ =  [](DMatrix<double> locs){
        DMatrix<double> res = DMatrix<double>::Zero(locs.rows(),1);
        for(std::size_t i = 0; i < locs.rows(); ++i){
                    res(i,0) = 1-(locs(i,0)-0.5)*(locs(i,0)-0.5) -(locs(i,1)-0.5)*(locs(i,1)-0.5); 
        }   
        return res;
    };

    //std::string meshID = "unit_square";
    std::string meshID = "unit_square_coarse";
    MeshLoader<Mesh2D> domain(meshID);
    meshID = meshID + "/"; 
    std::string name_dir = "../data/models/mixed_srpde/" + meshID;
	if(!std::filesystem::create_directory(name_dir)) std::filesystem::create_directory(name_dir);
    
    name_dir += "same_locations_test_1/";
    if(!std::filesystem::create_directory(name_dir)) std::filesystem::create_directory(name_dir);
	
    // input
    std::string input_dir = name_dir  + "input/";

    if(!std::filesystem::exists(std::filesystem::path(input_dir))) {
        std::cout << "\t --- generating data --- " << std::endl;
        std::filesystem::create_directory(input_dir);

        Eigen::saveMarket(beta, input_dir + "beta.mtx");
        Eigen::saveMarket(alpha, input_dir + "alpha.mtx");
        Eigen::saveMarket(n_obs, input_dir + "n_obs.mtx");
    
        eigen2txt<double>(beta, input_dir + "beta.txt");
        eigen2txt<double>(alpha, input_dir + "alpha.txt");
        eigen2txt<int>(n_obs, input_dir + "n_obs.txt");

        Eigen::saveMarket(x1_(domain.mesh.nodes()), input_dir + "cov_1.mtx");
        eigen2txt<double>(x1_(domain.mesh.nodes()), input_dir + "cov_1.txt");
        for( std::size_t j=0; j < m; ++j){
            DMatrix<double> f_ = f(domain.mesh.nodes(), j);
            Eigen::saveMarket(f_, input_dir + "f_" + std::to_string(j) + ".mtx");
            eigen2txt<double>(f_, input_dir + "f_" + std::to_string(j) + ".txt");
        }

        for(std::size_t n = 0; n < n_obs.rows(); ++n){
            
            // generete data
            std::string data_dir = input_dir + std::to_string(n_obs(n)) + "/";
            std::filesystem::create_directory(data_dir);
             
        for(std::size_t sim=0; sim<n_sim; ++sim){
            std::string simul_dir = data_dir + std::to_string(sim) + "/"; 
            std::filesystem::create_directory(simul_dir);

            DMatrix<double> locs = uniform_locs(n_obs(n));
            
            for(std::size_t j = 0; j < m; ++j){
                DMatrix<double> DesignMatrix = DMatrix<double>::Zero(n_obs(n),2);
            
                DesignMatrix.col(0) = x1_(locs); // va in V
                DesignMatrix.col(1) = noise(n_obs(n), 1.0);

                DMatrix<double> f_ = f(locs, j);
                double sigma = 0.05*std::abs(f_.array().maxCoeff() - f_.array().minCoeff()); 
                auto eps_ = noise(n_obs(n),sigma);
                eigen2txt<double>(eps_, simul_dir + "noise_" + std::to_string(j) + ".txt");
            
                auto obs = DesignMatrix * beta + DesignMatrix.col(0)*alpha(j,0)  + f_ + eps_; 
            
                Eigen::saveMarket(locs, simul_dir + "locs_" + std::to_string(j) + ".mtx");
                Eigen::saveMarket(DesignMatrix, simul_dir + "DesignMatrix_" + std::to_string(j) + ".mtx");
                Eigen::saveMarket(DesignMatrix.col(1), simul_dir + "W_" + std::to_string(j) + ".mtx");
                Eigen::saveMarket(DesignMatrix.col(0), simul_dir + "V_" + std::to_string(j) + ".mtx");
                Eigen::saveMarket(obs, simul_dir + "obs_" + std::to_string(j) + ".mtx");

                eigen2txt<double>(locs, simul_dir + "locs_" + std::to_string(j) + ".txt");
                eigen2txt<double>(DesignMatrix, simul_dir + "DesignMatrix_" + std::to_string(j) + ".txt");
                eigen2txt<double>(DesignMatrix.col(1), simul_dir + "W_" + std::to_string(j) + ".txt");
                eigen2txt<double>(DesignMatrix.col(0), simul_dir + "V_" + std::to_string(j) + ".txt");
                eigen2txt<double>(obs, simul_dir + "obs_" + std::to_string(j) + ".txt");
            }
        }
        }
    }
     
    // Output directory
	std::string output_dir = name_dir + "output/";
    if(!std::filesystem::exists(std::filesystem::path(output_dir))){ 
        std::filesystem::create_directory(output_dir);
    }
    
    std::vector<std::string> solution_policy = {"monolithic", "richardson"};

    // import data from files
    std::vector<std::string> header = {"time_init", "time_solve", "time",
                                       "rmse_f","rmse_f_1", "rmse_f_2","rmse_f_3", 
                                       "rmse_beta","rmse_alpha","n_obs"};

    DMatrix<double> results_mono = DMatrix<double>::Zero( n_sim*n_obs.size(), header.size());
    DMatrix<double> results_rich = DMatrix<double>::Zero( n_sim*n_obs.size(), header.size());

    for(std::size_t n = 0; n < n_obs.rows(); ++n){ 
        output_dir = name_dir + "output/"; // + "monolithic/";
        output_dir += std::to_string(n_obs(n)) + "/" ;
        
        std::string data_dir = input_dir + std::to_string(n_obs(n)) + "/";
        if(!std::filesystem::exists(std::filesystem::path(output_dir))) std::filesystem::create_directory(output_dir);
        
    for(std::size_t sim = 0; sim < n_sim; ++sim){

        std::string simul_dir = data_dir + std::to_string(sim) + "/"; 
        std::string result_dir = output_dir + std::to_string(sim) + "/";
        if(!std::filesystem::exists(std::filesystem::path(result_dir))) std::filesystem::create_directory(result_dir);

        std::string output_monolithic = result_dir + "monolithic/";
        std::string output_richardson = result_dir + "richardson/";
        if(!std::filesystem::exists(std::filesystem::path(output_monolithic))) std::filesystem::create_directory(output_monolithic);
        if(!std::filesystem::exists(std::filesystem::path(output_richardson))) std::filesystem::create_directory(output_richardson);

        std::vector<BlockFrame<double, int>> data;
        data.resize(m);
        
        for(std::size_t j = 0; j<m; j++){
            std::string Wname = simul_dir + "W_" + std::to_string(j) + ".mtx";
            std::string Vname = simul_dir + "V_" + std::to_string(j) + ".mtx";
            std::string locsname = simul_dir + "locs_" + std::to_string(j) + ".mtx";
            std::string yname = simul_dir +  "obs_" + std::to_string(j) + ".mtx";
            auto W = read_mtx<double>(Wname);
            auto V = read_mtx<double>(Vname);
            auto locs = read_mtx<double>(locsname);
            auto obs = read_mtx<double>(yname);
            
            data[j].insert(W_BLOCK, W);
            data[j].insert(V_BLOCK, V);
            data[j].insert(Y_BLOCK, obs);
            data[j].insert(LOCS_BLOCK, locs);      
        }
    
        DMatrix<double> f_ = DMatrix<double>::Zero(m*domain.mesh.nodes().rows(),1);
        for(std::size_t j = 0; j < m; ++j){
            f_.block(j*domain.mesh.nodes().rows(),0, domain.mesh.nodes().rows(),1) =   f(domain.mesh.nodes(),j);
        }
    
        // define regularizing PDE
        auto L = -laplacian<FEM>();
        DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.n_elements()*3, 1);
        PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> problem(domain.mesh, L, u);
	
        // define lambda
        double lambda = 1e-3; 

        // monolithic 
        MixedSRPDE<monolithic> monolithic_(problem, Sampling::pointwise);
        monolithic_.set_lambda_D(lambda);
	    monolithic_.set_data(data);
        
        auto start = std::chrono::high_resolution_clock::now();
        monolithic_.init();
        std::chrono::duration<double> duration = std::chrono::high_resolution_clock::now() - start;
        results_mono(sim + n_sim*n, 0) = duration.count();

        start = std::chrono::high_resolution_clock::now();
        monolithic_.solve();
        duration = std::chrono::high_resolution_clock::now() - start;
        results_mono(sim + n_sim*n, 1) = duration.count();
        results_mono(sim + n_sim*n, 2) = results_mono(sim + n_sim*n, 0) + results_mono(sim + n_sim*n, 1);

        // iterative
        MixedSRPDE<iterative> richardson_(problem, Sampling::pointwise);
        richardson_.set_lambda_D(lambda);
	    richardson_.set_data(data);

        start = std::chrono::high_resolution_clock::now();
        richardson_.init();
        duration = std::chrono::high_resolution_clock::now() - start;
        results_rich(sim + n_sim*n, 0) = duration.count();

        start = std::chrono::high_resolution_clock::now();
        richardson_.solve();
        duration = std::chrono::high_resolution_clock::now() - start;
        results_rich(sim + n_sim*n, 1) = duration.count();
        results_rich(sim + n_sim*n, 2) = results_rich(sim + n_sim*n, 0) + results_rich(sim + n_sim*n, 1);
        
        // RMSEs
        results_mono(sim + n_sim*n, 3) = (monolithic_.f() - f_).array().square().mean();
        results_rich(sim + n_sim*n, 3) = (richardson_.f() - f_).array().square().mean();
        for(std::size_t j = 0; j < m; ++j){
            Eigen::saveMarket(monolithic_.f().block(j*domain.mesh.nodes().rows(),0, domain.mesh.nodes().rows(),1),
                          output_monolithic + "estimate_f_" + std::to_string(j) + ".mtx");
            eigen2txt<double>(monolithic_.f().block(j*domain.mesh.nodes().rows(),0, domain.mesh.nodes().rows(),1),
                          output_monolithic + "estimate_f_" + std::to_string(j) + ".txt");

            results_mono(sim + n_sim*n, 4+j) = (monolithic_.f().block(j*domain.mesh.nodes().rows(),0, domain.mesh.nodes().rows(),1) -
                                                f_.block(j*domain.mesh.nodes().rows(),0, domain.mesh.nodes().rows(),1)).array().square().mean();

            Eigen::saveMarket(richardson_.f().block(j*domain.mesh.nodes().rows(),0, domain.mesh.nodes().rows(),1),
                          output_richardson + "estimate_f_" + std::to_string(j) + ".mtx");
            eigen2txt<double>(richardson_.f().block(j*domain.mesh.nodes().rows(),0, domain.mesh.nodes().rows(),1),
                          output_richardson + "estimate_f_" + std::to_string(j) + ".txt");

            results_rich(sim + n_sim*n, 4+j) = (richardson_.f().block(j*domain.mesh.nodes().rows(),0, domain.mesh.nodes().rows(),1) -
                                                f_.block(j*domain.mesh.nodes().rows(),0, domain.mesh.nodes().rows(),1)).array().square().mean();
        }

        Eigen::saveMarket(monolithic_.f(), output_monolithic + "estimate_f.mtx");
        eigen2txt<double>(monolithic_.f(), output_monolithic + "estimate_f.txt");
        Eigen::saveMarket(richardson_.f(), output_richardson + "estimate_f.mtx");
        eigen2txt<double>(richardson_.f(), output_richardson + "estimate_f.txt");
    
        Eigen::saveMarket(monolithic_.beta(), output_monolithic + "beta.mtx");
        eigen2txt<double>(monolithic_.beta(), output_monolithic + "beta.txt");
        Eigen::saveMarket(richardson_.beta(), output_richardson + "beta.mtx");
        eigen2txt<double>(richardson_.beta(), output_richardson + "beta.txt");
    
        Eigen::saveMarket(monolithic_.alpha(), output_monolithic + "beta.mtx");
        eigen2txt<double>(monolithic_.alpha(), output_monolithic + "alpha.txt");
        Eigen::saveMarket(richardson_.alpha(), output_richardson + "beta.mtx");
        eigen2txt<double>(richardson_.alpha(), output_richardson + "alpha.txt");

        results_mono(sim + n_sim*n, 7) =  (monolithic_.beta() - beta).array().square().mean();
        results_mono(sim + n_sim*n, 8) =  (monolithic_.alpha() - alpha).array().square().mean();
        
        results_rich(sim + n_sim*n, 7) =  (richardson_.beta() - beta).array().square().mean();
        results_rich(sim + n_sim*n, 8) =  (richardson_.alpha() - alpha).array().square().mean();
        
        results_mono(sim + n_sim*n,9) = n_obs(n);
        results_rich(sim + n_sim*n,9) = n_obs(n);

        EXPECT_TRUE(  (monolithic_.beta() - beta).array().square().mean() < 1e-2 );
        EXPECT_TRUE(  (monolithic_.alpha() - alpha).array().square().mean() < 1e-2 );

        EXPECT_TRUE(  (richardson_.beta() - beta).array().square().mean() < 1e-2 );
        EXPECT_TRUE(  (richardson_.alpha() - alpha).array().square().mean() < 1e-2 );
    }
    }

    write_table(results_mono, header, name_dir + "output/" + solution_policy[0] + ".txt");
    write_table(results_rich, header, name_dir + "output/" + solution_policy[1] + ".txt");
}


/*
TEST(mixed_srpde_test, iterative_same_locations) {
    int seed = 0; 
    std::size_t n_sim = 30;
    std::mt19937 gen(seed);
    std::size_t m = 3;
    //std::string meshID = "unit_square_coarse";
    std::string meshID = "unit_square";
    MeshLoader<Mesh2D> domain(meshID);
    meshID = meshID + "/"; 

    std::string name_dir = "../data/models/mixed_srpde/" + meshID;
	if(!std::filesystem::create_directory(name_dir)) std::filesystem::create_directory(name_dir);
    
    name_dir += "same_locations/";
	if(!std::filesystem::create_directory(name_dir)) std::filesystem::create_directory(name_dir);
	
    // input
    std::string input_dir = name_dir + "input/";
 
    // Output directory
	std::string output_dir = name_dir + "output/";
    if(!std::filesystem::exists(std::filesystem::path(output_dir))){ 
        std::filesystem::create_directory(output_dir);
    }
    output_dir = output_dir + "iterative/";
    if(!std::filesystem::exists(std::filesystem::path(output_dir))){ 
        std::filesystem::create_directory(output_dir);
    }

    DMatrix<double> beta = read_mtx<double>(input_dir + "beta.mtx");
    DMatrix<double> alpha = read_mtx<double>(input_dir + "alpha.mtx");
    DMatrix<int> n_obs = read_mtx<int>(input_dir + "n_obs.mtx");
    std::ofstream file(name_dir + "output/results.txt", std::ios_base::app);

    for(std::size_t n = 0; n < n_obs.rows(); ++n){ 
        output_dir = name_dir + "output/" + "iterative/";
        output_dir += std::to_string(n_obs(n)) + "/" ;
        std::string data_dir = input_dir + std::to_string(n_obs(n)) + "/";
        if(!std::filesystem::exists(std::filesystem::path(output_dir))) std::filesystem::create_directory(output_dir);
    for(std::size_t sim = 0; sim < n_sim; ++sim){

        std::string simul_dir = data_dir + std::to_string(sim) + "/"; 
        std::string result_dir = output_dir + std::to_string(sim) + "/";
        if(!std::filesystem::exists(std::filesystem::path(result_dir))) std::filesystem::create_directory(result_dir);

        std::vector<BlockFrame<double, int>> data;
        data.resize(m);
    
        // import data from files
        for(std::size_t j = 0; j<m; j++){
            std::string Wname = simul_dir + "W_" + std::to_string(j) + ".mtx";
            std::string Vname = simul_dir + "V_" + std::to_string(j) + ".mtx";
            std::string locsname = simul_dir + "locs_" + std::to_string(j) + ".mtx";
            std::string yname = simul_dir +  "obs_" + std::to_string(j) + ".mtx";
    
            auto W = read_mtx<double>(Wname);
            auto V = read_mtx<double>(Vname);
            auto locs = read_mtx<double>(locsname);
            auto obs = read_mtx<double>(yname);
        
            data[j].insert(W_BLOCK, W);
            data[j].insert(V_BLOCK, V);
            data[j].insert(Y_BLOCK, obs);
            data[j].insert(LOCS_BLOCK, locs);
        }
    
        // f_hat
        DMatrix<double> f_ = DMatrix<double>::Zero(m*domain.mesh.nodes().rows(),1);
        for(std::size_t j = 0; j < m; ++j){
            std::string fname = input_dir + "f_" + std::to_string(j) + ".mtx";
            std::cout << fname << std::endl;
            f_.block(j*domain.mesh.nodes().rows(),0, domain.mesh.nodes().rows(),1) = read_mtx<double>(fname);
        }
    
        auto L = -laplacian<FEM>();
        DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.n_elements()*3, 1);
        PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> problem(domain.mesh, L, u);
	
        // define lambda
        double lambda = 1e-3; 

        // getting dimension of the data (number of total observations N)
        std::size_t sum = 0;
        for(std::size_t i=0; i<data.size(); i++){
            sum += data[i].template get<double>(LOCS_BLOCK).rows();
        }
        MixedSRPDE<iterative> model(problem, Sampling::pointwise);
	
	    model.set_lambda_D(lambda);
	    model.set_data(data);
        
        // solve smoothing problem
        auto start = std::chrono::high_resolution_clock::now();
        model.init();
        std::chrono::duration<double> duration = std::chrono::high_resolution_clock::now() - start;
        file << duration.count() << " ";

        start = std::chrono::high_resolution_clock::now();
        model.solve();
        duration = std::chrono::high_resolution_clock::now() - start;
        file << duration.count() << " " << "'iterative'" << " ";
    
        file << (model.f() - f_).array().square().mean() << " ";
        for(std::size_t j = 0; j < m; ++j){
            Eigen::saveMarket(model.f().block(j*domain.mesh.nodes().rows(),0, domain.mesh.nodes().rows(),1),
                              result_dir + "estimate_f_" + std::to_string(j) + ".mtx");
            eigen2txt<double>(model.f().block(j*domain.mesh.nodes().rows(),0, domain.mesh.nodes().rows(),1),
                          result_dir + "estimate_f_" + std::to_string(j) + ".txt");
            file << (model.f().block(j*domain.mesh.nodes().rows(),0, domain.mesh.nodes().rows(),1) -
                            f_.block(j*domain.mesh.nodes().rows(),0, domain.mesh.nodes().rows(),1)).array().square().mean() << " ";
            EXPECT_TRUE( (model.f().block(j*domain.mesh.nodes().rows(),0, domain.mesh.nodes().rows(),1) -
                            f_.block(j*domain.mesh.nodes().rows(),0, domain.mesh.nodes().rows(),1)).array().square().mean() < 1e-2);
        }

        Eigen::saveMarket(model.f(), result_dir + "estimate_f.mtx");
        eigen2txt<double>(model.f(), result_dir + "estimate_f.txt");
    
        Eigen::saveMarket(model.beta(), result_dir + "beta.mtx");
        eigen2txt<double>(model.beta(), result_dir + "beta.txt");
    
        Eigen::saveMarket(model.alpha(), result_dir + "beta.mtx");
        eigen2txt<double>(model.alpha(), result_dir + "alpha.txt");
        file << (model.beta() - beta).array().square().mean() << " " << 
                (model.alpha() - alpha).array().square().mean() << " " << n_obs(n) << "\n"; 
    
        EXPECT_TRUE(  (model.alpha() - alpha).array().square().mean() < 1e-2 );
        EXPECT_TRUE(  (model.beta() - beta).array().square().mean() < 1e-2 );
    }
    }
}
*/

/*
TEST(mixed_srpde_test, monolitich_same_locations_noise_covs) {
    
    std::size_t m = 3;
    std::size_t n_sim = 30;
    // 
    DMatrix<double> beta = DMatrix<double>::Zero(2,1);
    beta(0,0) = -2.; beta(1,0) = 1.;

    DMatrix<double> alpha = DMatrix<double>::Zero(3,1);
    alpha(0,0) = -0.5; alpha(1,0) = 0.; alpha(2,0) = 0.5;

    DMatrix<int> n_obs = DMatrix<int>::Zero(5,1);
    n_obs(0,0) = 500; n_obs(1,0) = 1000; n_obs(2,0) = 2000; 
    n_obs(3,0) = 4000; n_obs(4,0) = 8000;
    
    int seed = 0; 
    std::mt19937 gen(seed);
    
auto uniform_locs = [&gen](std::size_t n) {
        std::uniform_real_distribution<> dis(0.0, 1.0);
        DMatrix<double> locs = DMatrix<double>::Zero(n,2);
        for (std::size_t i = 0; i < n; ++i) {
            locs(i,0) = dis(gen);  // x
            locs(i,1) = dis(gen);  // y
        }
        return locs;
};

auto f = [](DMatrix<double> locs, int id = 0) { 
	    DMatrix<double> res = DMatrix<double>::Zero(locs.rows(),1);
            for(std::size_t i = 0; i < locs.rows(); ++i){
                if(id == 0)
                    res(i,0) = std::sin(2*fdapde::testing::pi*locs(i,0))*
                                    std::sin(2*fdapde::testing::pi*locs(i,1));
                else if (id == 1)
                    res(i,0) = 1.0 - locs(i,0) - locs(i,1);
                else if (id == 2)
                    res(i,0) = 1-std::sin(fdapde::testing::pi*locs(i,0))*
                                    std::cos(fdapde::testing::pi*locs(i,1));//std::cos(fdapde::testing::pi*locs(i,0))*std::cos(fdapde::testing::pi*locs(i,1));
        }
            return res;
    };
    
auto noise = [&gen](std::size_t n, double sigma){
        DMatrix<double> res = DMatrix<double>::Zero(n,1);
        std::normal_distribution<> __noise(0.0, sigma);
        for(std::size_t i = 0; i < n; ++i){
            res(i,0) = __noise(gen);
        }
        return res;
    };

auto x1_ =  [](DMatrix<double> locs){
        DMatrix<double> res = DMatrix<double>::Zero(locs.rows(),1);
        for(std::size_t i = 0; i < locs.rows(); ++i){
                    res(i,0) = 1-(locs(i,0)-0.5)*(locs(i,0)-0.5) -(locs(i,1)-0.5)*(locs(i,1)-0.5); 
        }   
        return res;
    };

    //std::string meshID = "unit_square_coarse";
    std::string meshID = "unit_square";
    MeshLoader<Mesh2D> domain(meshID);
    meshID = meshID + "/"; 
    std::string name_dir = "../data/models/mixed_srpde/" + meshID;
	if(!std::filesystem::create_directory(name_dir)) std::filesystem::create_directory(name_dir);
    
    name_dir += "same_locations_noise_covs/";
    if(!std::filesystem::create_directory(name_dir)) std::filesystem::create_directory(name_dir);
	
    // input
    std::string input_dir = name_dir  + "input/";

    if(!std::filesystem::exists(std::filesystem::path(input_dir))) {
        std::cout << "\t --- generating data --- " << std::endl;
        std::filesystem::create_directory(input_dir);

        Eigen::saveMarket(beta, input_dir + "beta.mtx");
        Eigen::saveMarket(alpha, input_dir + "alpha.mtx");
        Eigen::saveMarket(n_obs, input_dir + "n_obs.mtx");
    
        eigen2txt<double>(beta, input_dir + "beta.txt");
        eigen2txt<double>(alpha, input_dir + "alpha.txt");
        eigen2txt<int>(n_obs, input_dir + "n_obs.txt");

        Eigen::saveMarket(x1_(domain.mesh.nodes()), input_dir + "cov_1.mtx");
        eigen2txt<double>(x1_(domain.mesh.nodes()), input_dir + "cov_1.txt");
        for( std::size_t j=0; j < m; ++j){
            DMatrix<double> f_ = f(domain.mesh.nodes(), j);
            Eigen::saveMarket(f_, input_dir + "f_" + std::to_string(j) + ".mtx");
            eigen2txt<double>(f_, input_dir + "f_" + std::to_string(j) + ".txt");
        }

        for(std::size_t n = 0; n < n_obs.rows(); ++n){
            
            // generete data
            std::string data_dir = input_dir + std::to_string(n_obs(n)) + "/";
            std::filesystem::create_directory(data_dir);
             
        for(std::size_t sim=0; sim<n_sim; ++sim){
            std::string simul_dir = data_dir + std::to_string(sim) + "/"; 
            std::filesystem::create_directory(simul_dir);

            DMatrix<double> locs = uniform_locs(n_obs(n));
            
            for(std::size_t j = 0; j < m; ++j){
                DMatrix<double> DesignMatrix = DMatrix<double>::Zero(n_obs(n),2);
                
                DesignMatrix.col(0) = x1_(locs);
                double sigma_cov = 0.05*std::abs(DesignMatrix.col(0).array().maxCoeff() - DesignMatrix.col(0).array().minCoeff());
                DesignMatrix.col(0) += noise(n_obs(n), sigma_cov); // va in V
                DesignMatrix.col(1) = noise(n_obs(n), 1.0);

                DMatrix<double> f_ = f(locs, j);
                double sigma = 0.05*std::abs(f_.array().maxCoeff() - f_.array().minCoeff()); 
                std::cout << sigma << std::endl;
                auto eps_ = noise(n_obs(n),sigma);
                eigen2txt<double>(eps_, simul_dir + "noise_" + std::to_string(j) + ".txt");
            
                auto obs = DesignMatrix * beta + DesignMatrix.col(0)*alpha(j,0)  + f_ + eps_; 
            
                Eigen::saveMarket(locs, simul_dir + "locs_" + std::to_string(j) + ".mtx");
                Eigen::saveMarket(DesignMatrix, simul_dir + "DesignMatrix_" + std::to_string(j) + ".mtx");
                Eigen::saveMarket(DesignMatrix.col(1), simul_dir + "W_" + std::to_string(j) + ".mtx");
                Eigen::saveMarket(DesignMatrix.col(0), simul_dir + "V_" + std::to_string(j) + ".mtx");
                Eigen::saveMarket(obs, simul_dir + "obs_" + std::to_string(j) + ".mtx");

                eigen2txt<double>(locs, simul_dir + "locs_" + std::to_string(j) + ".txt");
                eigen2txt<double>(DesignMatrix, simul_dir + "DesignMatrix_" + std::to_string(j) + ".txt");
                eigen2txt<double>(DesignMatrix.col(1), simul_dir + "W_" + std::to_string(j) + ".txt");
                eigen2txt<double>(DesignMatrix.col(0), simul_dir + "V_" + std::to_string(j) + ".txt");
                eigen2txt<double>(obs, simul_dir + "obs_" + std::to_string(j) + ".txt");
            }
        }
        }
    }
     
    // Output directory
	std::string output_dir = name_dir + "output/";
    if(!std::filesystem::exists(std::filesystem::path(output_dir))){ 
        std::filesystem::create_directory(output_dir);
    }
    output_dir += "monolithic/";
    if(!std::filesystem::exists(std::filesystem::path(output_dir))){ 
        std::filesystem::create_directory(output_dir);
    }
    
    std::ofstream file(name_dir + "output/results.txt");
    file << "'time_init'" <<  " " << "'time_solve'" << " " << "'solution_policy'"  << 
            "'rmse_f'" << " " << "'rmse_f_1'" << " " << "'rmse_f_2'" << " " << "'rmse_f_3'" << " " << 
            "'rmse_beta'" << " " << "'rmse_alpha'" << "'n_obs'" "\n";
    // import data from files
    for(std::size_t n = 0; n < n_obs.rows(); ++n){ 
        //input_dir = name_dir + "input/" + std::to_string(n_obs(n)) + "/"; 
        output_dir = name_dir + "output/" + "monolithic/";
        output_dir += std::to_string(n_obs(n)) + "/" ;
        std::string data_dir = input_dir + std::to_string(n_obs(n)) + "/";
        if(!std::filesystem::exists(std::filesystem::path(output_dir))) std::filesystem::create_directory(output_dir);
    for(std::size_t sim = 0; sim < n_sim; ++sim){

        std::string simul_dir = data_dir + std::to_string(sim) + "/"; 
        std::string result_dir = output_dir + std::to_string(sim) + "/";
        if(!std::filesystem::exists(std::filesystem::path(result_dir))) std::filesystem::create_directory(result_dir);

        std::vector<BlockFrame<double, int>> data;
        data.resize(m);
        
        for(std::size_t j = 0; j<m; j++){
            std::string Wname = simul_dir + "W_" + std::to_string(j) + ".mtx";
            std::string Vname = simul_dir + "V_" + std::to_string(j) + ".mtx";
            std::string locsname = simul_dir + "locs_" + std::to_string(j) + ".mtx";
            std::string yname = simul_dir +  "obs_" + std::to_string(j) + ".mtx";
            auto W = read_mtx<double>(Wname);
            auto V = read_mtx<double>(Vname);
            auto locs = read_mtx<double>(locsname);
            auto obs = read_mtx<double>(yname);
            
            data[j].insert(W_BLOCK, W);
            data[j].insert(V_BLOCK, V);
            data[j].insert(Y_BLOCK, obs);
            data[j].insert(LOCS_BLOCK, locs);      
        }
    
        DMatrix<double> f_ = DMatrix<double>::Zero(m*domain.mesh.nodes().rows(),1);
        for(std::size_t j = 0; j < m; ++j){
            f_.block(j*domain.mesh.nodes().rows(),0, domain.mesh.nodes().rows(),1) =   f(domain.mesh.nodes(),j);
        }
    
        // define regularizing PDE
        auto L = -laplacian<FEM>();
        DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.n_elements()*3, 1);
        PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> problem(domain.mesh, L, u);
	
        // define lambda
        double lambda = 1e-3; 

        // getting dimension of the data (number of total observations N)
        std::size_t sum = 0;
        for(std::size_t i=0; i<data.size(); i++){
            sum += data[i].template get<double>(LOCS_BLOCK).rows();
        }

        MixedSRPDE<monolithic> model(problem, Sampling::pointwise);
	
	    model.set_lambda_D(lambda);
	    model.set_data(data);
        
        // solve smoothing problem
        auto start = std::chrono::high_resolution_clock::now();
        model.init();
        std::chrono::duration<double> duration = std::chrono::high_resolution_clock::now() - start;
        file << duration.count() << " ";

        start = std::chrono::high_resolution_clock::now();
        model.solve();
        duration = std::chrono::high_resolution_clock::now() - start;
        file << duration.count() << " " << "'monolithic'" << " ";
    
        file << (model.f() - f_).array().square().mean() << " ";
        for(std::size_t j = 0; j < m; ++j){
            Eigen::saveMarket(model.f().block(j*domain.mesh.nodes().rows(),0, domain.mesh.nodes().rows(),1),
                          result_dir + "estimate_f_" + std::to_string(j) + ".mtx");
            eigen2txt<double>(model.f().block(j*domain.mesh.nodes().rows(),0, domain.mesh.nodes().rows(),1),
                          result_dir + "estimate_f_" + std::to_string(j) + ".txt");
            file << (model.f().block(j*domain.mesh.nodes().rows(),0, domain.mesh.nodes().rows(),1) -
                            f_.block(j*domain.mesh.nodes().rows(),0, domain.mesh.nodes().rows(),1)).array().square().mean() << " ";
            EXPECT_TRUE( (model.f().block(j*domain.mesh.nodes().rows(),0, domain.mesh.nodes().rows(),1) -
                            f_.block(j*domain.mesh.nodes().rows(),0, domain.mesh.nodes().rows(),1)).array().square().mean() < 1e-2);
        }

        Eigen::saveMarket(model.f(), result_dir + "estimate_f.mtx");
        eigen2txt<double>(model.f(), result_dir + "estimate_f.txt");
    
        Eigen::saveMarket(model.beta(), result_dir + "beta.mtx");
        eigen2txt<double>(model.beta(), result_dir + "beta.txt");
    
        Eigen::saveMarket(model.alpha(), result_dir + "beta.mtx");
        eigen2txt<double>(model.alpha(), result_dir + "alpha.txt");
        file << (model.beta() - beta).array().square().mean() << " " << 
            (model.alpha() - alpha).array().square().mean() << " " << n_obs(n) << "\n"; 
        EXPECT_TRUE(  (model.beta() - beta).array().square().mean() < 1e-2 );
        EXPECT_TRUE(  (model.alpha() - alpha).array().square().mean() < 1e-2 );
    }
    }
}

TEST(mixed_srpde_test, iterative_same_locations_noise_covs) {
    int seed = 0; 
    std::size_t n_sim = 30;
    std::mt19937 gen(seed);
    std::size_t m = 3;
    //std::string meshID = "unit_square_coarse";
    std::string meshID = "unit_square";
    MeshLoader<Mesh2D> domain(meshID);
    meshID = meshID + "/"; 

    std::string name_dir = "../data/models/mixed_srpde/" + meshID;
	if(!std::filesystem::create_directory(name_dir)) std::filesystem::create_directory(name_dir);
    
    name_dir += "same_locations_noise_covs/";
	if(!std::filesystem::create_directory(name_dir)) std::filesystem::create_directory(name_dir);
	
    // input
    std::string input_dir = name_dir + "input/";
 
    // Output directory
	std::string output_dir = name_dir + "output/";
    if(!std::filesystem::exists(std::filesystem::path(output_dir))){ 
        std::filesystem::create_directory(output_dir);
    }
    output_dir = output_dir + "iterative/";
    if(!std::filesystem::exists(std::filesystem::path(output_dir))){ 
        std::filesystem::create_directory(output_dir);
    }

    DMatrix<double> beta = read_mtx<double>(input_dir + "beta.mtx");
    DMatrix<double> alpha = read_mtx<double>(input_dir + "alpha.mtx");
    DMatrix<int> n_obs = read_mtx<int>(input_dir + "n_obs.mtx");
    std::ofstream file(name_dir + "output/results.txt", std::ios_base::app);

    for(std::size_t n = 0; n < n_obs.rows(); ++n){ 
        output_dir = name_dir + "output/" + "iterative/";
        output_dir += std::to_string(n_obs(n)) + "/" ;
        std::string data_dir = input_dir + std::to_string(n_obs(n)) + "/";
        if(!std::filesystem::exists(std::filesystem::path(output_dir))) std::filesystem::create_directory(output_dir);
    for(std::size_t sim = 0; sim < n_sim; ++sim){

        std::string simul_dir = data_dir + std::to_string(sim) + "/"; 
        std::string result_dir = output_dir + std::to_string(sim) + "/";
        if(!std::filesystem::exists(std::filesystem::path(result_dir))) std::filesystem::create_directory(result_dir);

        std::vector<BlockFrame<double, int>> data;
        data.resize(m);
    
        // import data from files
        for(std::size_t j = 0; j<m; j++){
            std::string Wname = simul_dir + "W_" + std::to_string(j) + ".mtx";
            std::string Vname = simul_dir + "V_" + std::to_string(j) + ".mtx";
            std::string locsname = simul_dir + "locs_" + std::to_string(j) + ".mtx";
            std::string yname = simul_dir +  "obs_" + std::to_string(j) + ".mtx";
    
            auto W = read_mtx<double>(Wname);
            auto V = read_mtx<double>(Vname);
            auto locs = read_mtx<double>(locsname);
            auto obs = read_mtx<double>(yname);
        
            data[j].insert(W_BLOCK, W);
            data[j].insert(V_BLOCK, V);
            data[j].insert(Y_BLOCK, obs);
            data[j].insert(LOCS_BLOCK, locs);
        }
    
        // f_hat
        DMatrix<double> f_ = DMatrix<double>::Zero(m*domain.mesh.nodes().rows(),1);
        for(std::size_t j = 0; j < m; ++j){
            std::string fname = input_dir + "f_" + std::to_string(j) + ".mtx";
            std::cout << fname << std::endl;
            f_.block(j*domain.mesh.nodes().rows(),0, domain.mesh.nodes().rows(),1) = read_mtx<double>(fname);
        }
    
        auto L = -laplacian<FEM>();
        DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.n_elements()*3, 1);
        PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> problem(domain.mesh, L, u);
	
        // define lambda
        double lambda = 1e-3; 

        // getting dimension of the data (number of total observations N)
        std::size_t sum = 0;
        for(std::size_t i=0; i<data.size(); i++){
            sum += data[i].template get<double>(LOCS_BLOCK).rows();
        }
        MixedSRPDE<iterative> model(problem, Sampling::pointwise);
	
	    model.set_lambda_D(lambda);
	    model.set_data(data);
        
        // solve smoothing problem
        auto start = std::chrono::high_resolution_clock::now();
        model.init();
        std::chrono::duration<double> duration = std::chrono::high_resolution_clock::now() - start;
        file << duration.count() << " ";

        start = std::chrono::high_resolution_clock::now();
        model.solve();
        duration = std::chrono::high_resolution_clock::now() - start;
        file << duration.count() << " " << "'iterative'" << " ";
    
        file << (model.f() - f_).array().square().mean() << " ";
        for(std::size_t j = 0; j < m; ++j){
            Eigen::saveMarket(model.f().block(j*domain.mesh.nodes().rows(),0, domain.mesh.nodes().rows(),1),
                              result_dir + "estimate_f_" + std::to_string(j) + ".mtx");
            eigen2txt<double>(model.f().block(j*domain.mesh.nodes().rows(),0, domain.mesh.nodes().rows(),1),
                          result_dir + "estimate_f_" + std::to_string(j) + ".txt");
            file << (model.f().block(j*domain.mesh.nodes().rows(),0, domain.mesh.nodes().rows(),1) -
                            f_.block(j*domain.mesh.nodes().rows(),0, domain.mesh.nodes().rows(),1)).array().square().mean() << " ";
            EXPECT_TRUE( (model.f().block(j*domain.mesh.nodes().rows(),0, domain.mesh.nodes().rows(),1) -
                            f_.block(j*domain.mesh.nodes().rows(),0, domain.mesh.nodes().rows(),1)).array().square().mean() < 1e-2);
        }

        Eigen::saveMarket(model.f(), result_dir + "estimate_f.mtx");
        eigen2txt<double>(model.f(), result_dir + "estimate_f.txt");
    
        Eigen::saveMarket(model.beta(), result_dir + "beta.mtx");
        eigen2txt<double>(model.beta(), result_dir + "beta.txt");
    
        Eigen::saveMarket(model.alpha(), result_dir + "beta.mtx");
        eigen2txt<double>(model.alpha(), result_dir + "alpha.txt");
        file << (model.beta() - beta).array().square().mean() << " " << 
                (model.alpha() - alpha).array().square().mean() << " " << n_obs(n) << "\n"; 
    
        EXPECT_TRUE(  (model.alpha() - alpha).array().square().mean() < 1e-2 );
        EXPECT_TRUE(  (model.beta() - beta).array().square().mean() < 1e-2 );
    }
    }
}
*/

/*
TEST(mixed_srpde_test, monolitich_different_locations) {

    std::size_t m = 3;
    //     
    std::vector<BlockFrame<double, int>> data;
    data.resize(m);
    // 
    DMatrix<double> beta = DMatrix<double>::Zero(2,1);
    beta(0,0) = -2.; beta(1,0) = 1.;

    DMatrix<double> alpha = DMatrix<double>::Zero(3,1);
    alpha(0,0) = -0.5; alpha(1,0) = 0.; alpha(2,0) = 0.5;

    DMatrix<int> n_obs = DMatrix<int>::Zero(3,1);
    n_obs(0,0) = 15000; n_obs(1,0) = 14900; n_obs(2,0) = 15100;
    
    int seed = 0; 
    std::mt19937 gen(seed);
    
    auto uniform_locs = [&gen](std::size_t n) {
        std::uniform_real_distribution<> dis(0.0, 1.0);
        DMatrix<double> locs = DMatrix<double>::Zero(n,2);
        for (std::size_t i = 0; i < n; ++i) {
            locs(i,0) = dis(gen);  // x
            locs(i,1) = dis(gen);  // y
        }
        return locs;
    };

    auto f = [](DMatrix<double> locs, int id = 0) { 
	    DMatrix<double> res = DMatrix<double>::Zero(locs.rows(),1);
            for(std::size_t i = 0; i < locs.rows(); ++i){
                if(id == 0)
                    res(i,0) = std::sin(2*fdapde::testing::pi*locs(i,0))*
                                    std::sin(2*fdapde::testing::pi*locs(i,1));
                else if (id == 1)
                    res(i,0) = 1.0 - locs(i,0) - locs(i,1);
                else if (id == 2)
                    res(i,0) = 1-std::sin(fdapde::testing::pi*locs(i,0))*
                                    std::cos(fdapde::testing::pi*locs(i,1));//std::cos(fdapde::testing::pi*locs(i,0))*std::cos(fdapde::testing::pi*locs(i,1));
        }
            return res;
        };
    
    auto noise = [&gen](std::size_t n, double sigma){
        DMatrix<double> res = DMatrix<double>::Zero(n,1);
        std::normal_distribution<> __noise(0.0, sigma);
        for(std::size_t i = 0; i < n; ++i){
            res(i,0) = __noise(gen);
        }
        return res;
    };

    auto x1_ =  [](DMatrix<double> locs){
        DMatrix<double> res = DMatrix<double>::Zero(locs.rows(),1);
        for(std::size_t i = 0; i < locs.rows(); ++i){
                    res(i,0) = 1-(locs(i,0)-0.5)*(locs(i,0)-0.5) -(locs(i,1)-0.5)*(locs(i,1)-0.5); 
        }   
        return res;
    };


    //std::string meshID = "unit_square_coarse";
    std::string meshID = "unit_square_fine";
    MeshLoader<Mesh2D> domain(meshID);
    meshID = meshID + "/"; 
    std::string name_dir = "../data/models/mixed_srpde/" + meshID;
	if(!std::filesystem::create_directory(name_dir)) std::filesystem::create_directory(name_dir);
    
    name_dir += "different_locations/";
    if(!std::filesystem::create_directory(name_dir)) std::filesystem::create_directory(name_dir);
	
    // input
    std::string input_dir = name_dir  + "input/";

    if(!std::filesystem::exists(std::filesystem::path(input_dir))) {
        std::cout << "\t --- generating data --- " << std::endl;
        std::filesystem::create_directory(input_dir);

        Eigen::saveMarket(beta, input_dir + "beta.mtx");
        Eigen::saveMarket(alpha, input_dir + "alpha.mtx");
        Eigen::saveMarket(n_obs, input_dir + "n_obs.mtx");
    
        eigen2txt<double>(beta, input_dir + "beta.txt");
        eigen2txt<double>(alpha, input_dir + "alpha.txt");
        eigen2txt<int>(n_obs, input_dir + "n_obs.txt");

        Eigen::saveMarket(x1_(domain.mesh.nodes()), input_dir + "cov_1.mtx");
        eigen2txt<double>(x1_(domain.mesh.nodes()), input_dir + "cov_1.txt");
        
        // generete data    
        //double sigma = 0.1;
        for(std::size_t j = 0; j < m; ++j){
            DMatrix<double> locs = uniform_locs(n_obs(j));
            DMatrix<double> DesignMatrix = DMatrix<double>::Zero(n_obs(j),2);
            
            DesignMatrix.col(0) = x1_(locs); // va in V
            DesignMatrix.col(1) = noise(n_obs(j), 1.0);

            DMatrix<double> f_ = f(locs, j);
            double sigma = 0.05*std::abs(f_.array().maxCoeff() - f_.array().minCoeff()); 

            std::cout << sigma << std::endl;
            auto eps_ = noise(n_obs(j),sigma);
            eigen2txt<double>(eps_, input_dir + "noise_" + std::to_string(j) + ".txt");
            
            auto obs = DesignMatrix * beta + DesignMatrix.col(0)*alpha(j,0)  + f_ + eps_; 

            Eigen::saveMarket(locs, input_dir + "locs_" + std::to_string(j) + ".mtx");
            Eigen::saveMarket(DesignMatrix, input_dir + "DesignMatrix_" + std::to_string(j) + ".mtx");
            Eigen::saveMarket(DesignMatrix.col(1), input_dir + "W_" + std::to_string(j) + ".mtx");
            Eigen::saveMarket(DesignMatrix.col(0), input_dir + "V_" + std::to_string(j) + ".mtx");
            Eigen::saveMarket(obs, input_dir + "obs_" + std::to_string(j) + ".mtx");

            eigen2txt<double>(locs, input_dir + "locs_" + std::to_string(j) + ".txt");
            eigen2txt<double>(DesignMatrix, input_dir + "DesignMatrix_" + std::to_string(j) + ".txt");
            eigen2txt<double>(DesignMatrix.col(1), input_dir + "W_" + std::to_string(j) + ".txt");
            eigen2txt<double>(DesignMatrix.col(0), input_dir + "V_" + std::to_string(j) + ".txt");
            eigen2txt<double>(obs, input_dir + "obs_" + std::to_string(j) + ".txt");
            
            f_ = f(domain.mesh.nodes(), j);
            Eigen::saveMarket(f_, input_dir + "f_" + std::to_string(j) + ".mtx");
            eigen2txt<double>(f_, input_dir + "f_" + std::to_string(j) + ".txt");
        } 
    }
     
    // Output directory
	std::string output_dir = name_dir + "output/";
    if(!std::filesystem::exists(std::filesystem::path(output_dir))){ 
        std::filesystem::create_directory(output_dir);
    }
    output_dir = output_dir + "monolithic/";
    if(!std::filesystem::exists(std::filesystem::path(output_dir))){ 
        std::filesystem::create_directory(output_dir);
    }
    
    // import data from files
    for(std::size_t j = 0; j<m; j++){
        std::string Wname = input_dir + "W_" + std::to_string(j) + ".mtx";
        std::string Vname = input_dir + "V_" + std::to_string(j) + ".mtx";
        std::string locsname = input_dir + "locs_" + std::to_string(j) + ".mtx";
        std::string yname = input_dir +  "obs_" + std::to_string(j) + ".mtx";
        auto W = read_mtx<double>(Wname);
        auto V = read_mtx<double>(Vname);
        auto locs = read_mtx<double>(locsname);
        auto obs = read_mtx<double>(yname);
         
        data[j].insert(W_BLOCK, W);
        data[j].insert(V_BLOCK, V);
        data[j].insert(Y_BLOCK, obs);
        data[j].insert(LOCS_BLOCK, locs);      
    }
    
    DMatrix<double> f_ = DMatrix<double>::Zero(m*domain.mesh.nodes().rows(),1);
    for(std::size_t j = 0; j < m; ++j){
        f_.block(j*domain.mesh.nodes().rows(),0, domain.mesh.nodes().rows(),1) = f(domain.mesh.nodes(),j);
    }
    
    // define regularizing PDE
    auto L = -laplacian<FEM>();
    DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.n_elements()*3, 1);
    PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> problem(domain.mesh, L, u);
	
    // define lambda
    double lambda = 1e-3; 

    // getting dimension of the data (number of total observations N)
    std::size_t sum = 0;
    for(std::size_t i=0; i<data.size(); i++){
        sum += data[i].template get<double>(LOCS_BLOCK).rows();
    }

    //MixedSRPDE<SpaceOnly,monolithic> model(problem, Sampling::pointwise);
    MixedSRPDE<monolithic> model(problem, Sampling::pointwise);
	
	model.set_lambda_D(lambda);
	model.set_data(data);

    // solve smoothing problem
    auto start = std::chrono::high_resolution_clock::now();
    model.init();
    std::chrono::duration<double> duration = std::chrono::high_resolution_clock::now() - start;
    std::cout << "model.init(): " << duration.count() << std::endl;

    start = std::chrono::high_resolution_clock::now();
    model.solve();
    duration = std::chrono::high_resolution_clock::now() - start;
    std::cout << "model.solve(): " << duration.count() << std::endl;
    
    std::cout << "beta: \n" << model.beta() << std::endl;
    std::cout << "alpha: \n" << model.alpha() << std::endl;

    for(std::size_t j = 0; j < m; ++j){
        Eigen::saveMarket(model.f().block(j*domain.mesh.nodes().rows(),0, domain.mesh.nodes().rows(),1),
                          output_dir + "estimate_f_" + std::to_string(j) + ".mtx");
        eigen2txt<double>(model.f().block(j*domain.mesh.nodes().rows(),0, domain.mesh.nodes().rows(),1),
                          output_dir + "estimate_f_" + std::to_string(j) + ".txt");

        EXPECT_TRUE( (model.f().block(j*domain.mesh.nodes().rows(),0, domain.mesh.nodes().rows(),1) -
                            f_.block(j*domain.mesh.nodes().rows(),0, domain.mesh.nodes().rows(),1)).array().square().mean() < 1e-2);
    }

    Eigen::saveMarket(model.f(), output_dir + "estimate_f.mtx");
    eigen2txt<double>(model.f(), output_dir + "estimate_f.txt");
    
    Eigen::saveMarket(model.beta(), output_dir + "beta.mtx");
    eigen2txt<double>(model.beta(), output_dir + "beta.txt");
    
    Eigen::saveMarket(model.alpha(), output_dir + "beta.mtx");
    eigen2txt<double>(model.alpha(), output_dir + "alpha.txt");
     
    EXPECT_TRUE(  (model.alpha() - alpha).array().square().mean() < 1e-2 );
    EXPECT_TRUE(  (model.beta() - beta).array().square().mean() < 1e-2 );

}

TEST(mixed_srpde_test, iterative_different_locations) {
    int seed = 0; 
    std::mt19937 gen(seed);
    
    std::size_t m = 3;
    //     
    std::vector<BlockFrame<double, int>> data;
    data.resize(m);
    // 
    DMatrix<double> beta = DMatrix<double>::Zero(2,1);
    beta(0,0) = -2.; beta(1,0) = 1.;

    DMatrix<double> alpha = DMatrix<double>::Zero(3,1);
    alpha(0,0) = -0.5; alpha(1,0) = 0.; alpha(2,0) = 0.5;

    //std::string meshID = "unit_square_coarse";
    std::string meshID = "unit_square_fine";
    MeshLoader<Mesh2D> domain(meshID);
    meshID = meshID + "/"; 

    std::string name_dir = "../data/models/mixed_srpde/" + meshID;
	if(!std::filesystem::create_directory(name_dir)) std::filesystem::create_directory(name_dir);
    
    name_dir += "different_locations/";
	if(!std::filesystem::create_directory(name_dir)) std::filesystem::create_directory(name_dir);
	
    // input
    std::string input_dir = name_dir + "input/";
  
    // Output directory
	std::string output_dir = name_dir + "output/";
    if(!std::filesystem::exists(std::filesystem::path(output_dir))){ 
        std::filesystem::create_directory(output_dir);
    }
    output_dir = output_dir + "iterative/";
    if(!std::filesystem::exists(std::filesystem::path(output_dir))){ 
        std::filesystem::create_directory(output_dir);
    }

    // import data from files
    for(std::size_t j = 0; j<m; j++){
        std::cout<< "----- j = " << j << " -----" << std::endl;
        std::string Wname = input_dir + "W_" + std::to_string(j) + ".mtx";
        std::string Vname = input_dir + "V_" + std::to_string(j) + ".mtx";
        std::string locsname = input_dir + "locs_" + std::to_string(j) + ".mtx";
        std::string yname = input_dir +  "obs_" + std::to_string(j) + ".mtx";
        auto W = read_mtx<double>(Wname);
        auto V = read_mtx<double>(Vname);
        auto locs = read_mtx<double>(locsname);
        auto obs = read_mtx<double>(yname);
        
        std::cout << "W:" << W.rows() << " " <<W.cols() << std::endl;
        std::cout << "V:" << V.rows() << " " <<V.cols() << std::endl;
        std::cout << "locs:" << locs.rows() << " " <<locs.cols() << std::endl;
        std::cout << "obs:" << obs.rows() << " " <<obs.cols() << std::endl;
        
        data[j].insert(W_BLOCK, W);
        data[j].insert(V_BLOCK, V);
        data[j].insert(Y_BLOCK, obs);
        data[j].insert(LOCS_BLOCK, locs);
    }
    
    // f_hat
    DMatrix<double> f_ = DMatrix<double>::Zero(m*domain.mesh.nodes().rows(),1);
    for(std::size_t j = 0; j < m; ++j){
        std::string fname = input_dir + "f_" + std::to_string(j) + ".mtx";
        f_.block(j*domain.mesh.nodes().rows(),0, domain.mesh.nodes().rows(),1) = read_mtx<double>(fname);
    }
    
    auto L = -laplacian<FEM>();
    DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.n_elements()*3, 1);
    PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> problem(domain.mesh, L, u);
	
    // define lambda
    double lambda = 1e-3; 

    // getting dimension of the data (number of total observations N)
    std::size_t sum = 0;
    for(std::size_t i=0; i<data.size(); i++){
        sum += data[i].template get<double>(LOCS_BLOCK).rows();
    }

    MixedSRPDE<iterative> model(problem, Sampling::pointwise);
	
	model.set_lambda_D(lambda);
	model.set_data(data);
    
    // solve smoothing problem
    auto start = std::chrono::high_resolution_clock::now();
    model.init();
    std::chrono::duration<double> duration = std::chrono::high_resolution_clock::now() - start;
    std::cout << "model.init(): " << duration.count() << std::endl;

    start = std::chrono::high_resolution_clock::now();
    model.solve();
    duration = std::chrono::high_resolution_clock::now() - start;
    std::cout << "model.solve(): " << duration.count() << std::endl;
    
    std::cout << "beta: \n" << model.beta() << std::endl;
    std::cout << "alpha: \n" << model.alpha() << std::endl;
      
    for(std::size_t j = 0; j < m; ++j){
        Eigen::saveMarket(model.f().block(j*domain.mesh.nodes().rows(),0, domain.mesh.nodes().rows(),1),
                          output_dir + "estimate_f_" + std::to_string(j) + ".mtx");
        eigen2txt<double>(model.f().block(j*domain.mesh.nodes().rows(),0, domain.mesh.nodes().rows(),1),
                          output_dir + "estimate_f_" + std::to_string(j) + ".txt");

        EXPECT_TRUE( (model.f().block(j*domain.mesh.nodes().rows(),0, domain.mesh.nodes().rows(),1) -
                            f_.block(j*domain.mesh.nodes().rows(),0, domain.mesh.nodes().rows(),1)).array().square().mean() < 1e-2);
    }

    Eigen::saveMarket(model.f(), output_dir + "estimate_f.mtx");
    eigen2txt<double>(model.f(), output_dir + "estimate_f.txt");
    
    Eigen::saveMarket(model.beta(), output_dir + "beta.mtx");
    eigen2txt<double>(model.beta(), output_dir + "beta.txt");
    
    Eigen::saveMarket(model.alpha(), output_dir + "beta.mtx");
    eigen2txt<double>(model.alpha(), output_dir + "alpha.txt");
    
    EXPECT_TRUE(  (model.alpha() - alpha).array().square().mean() < 1e-2 );
    EXPECT_TRUE(  (model.beta() - beta).array().square().mean() < 1e-2 );

}
*/


/*
TEST(mixed_srpde_test, monolitich_same_locations_X_i_empty) {
    std::cout << " --- start --- " << std::endl; 
    
    std::size_t m = 3;
    //     
    std::vector<BlockFrame<double, int>> data;
    data.resize(m);
    // 
    DMatrix<double> beta = DMatrix<double>::Zero(2,1);
    beta(0,0) = -2.; beta(1,0) = 1.;

    DMatrix<double> alpha = DMatrix<double>::Zero(2,3);
    alpha(0,0) = -0.5; alpha(0,1) = 0.; alpha(0,2) = 0.5;
    alpha(1,0) = -0.25; alpha(1,1) = 0.25; alpha(1,2) = 0.5;

    DMatrix<int> n_obs = DMatrix<int>::Zero(3,1);
    n_obs(0,0) = 500; n_obs(1,0) = 500; n_obs(2,0) = 500;
    
    int seed = 0; 
    std::mt19937 gen(seed);
    
auto uniform_locs = [&gen](std::size_t n) {
        std::uniform_real_distribution<> dis(0.0, 1.0);
        DMatrix<double> locs = DMatrix<double>::Zero(n,2);
        for (std::size_t i = 0; i < n; ++i) {
            locs(i,0) = dis(gen);  // x
            locs(i,1) = dis(gen);  // y
        }
        return locs;
};

auto f = [](DMatrix<double> locs, int id = 0) { 
	    DMatrix<double> res = DMatrix<double>::Zero(locs.rows(),1);
            for(std::size_t i = 0; i < locs.rows(); ++i){
                if(id == 0)
                    res(i,0) = std::sin(2*fdapde::testing::pi*locs(i,0))*
                                    std::sin(2*fdapde::testing::pi*locs(i,1));
                else if (id == 1)
                    res(i,0) = 1.0 - locs(i,0) - locs(i,1);
                else if (id == 2)
                    res(i,0) = 1-std::sin(fdapde::testing::pi*locs(i,0))*
                                    std::cos(fdapde::testing::pi*locs(i,1));//std::cos(fdapde::testing::pi*locs(i,0))*std::cos(fdapde::testing::pi*locs(i,1));
        }
            return res;
    };
    
auto noise = [&gen](std::size_t n, double sigma){
        DMatrix<double> res = DMatrix<double>::Zero(n,1);
        std::normal_distribution<> __noise(0.0, sigma);
        for(std::size_t i = 0; i < n; ++i){
            res(i,0) = __noise(gen);
        }
        return res;
    };

auto x1_ =  [](DMatrix<double> locs){
        DMatrix<double> res = DMatrix<double>::Zero(locs.rows(),1);
        for(std::size_t i = 0; i < locs.rows(); ++i){
                    res(i,0) = 1-(locs(i,0)-0.5)*(locs(i,0)-0.5) -(locs(i,1)-0.5)*(locs(i,1)-0.5); 
        }   
        return res;
    };

    std::string meshID = "unit_square_coarse";
    MeshLoader<Mesh2D> domain(meshID);
    meshID = meshID + "/"; 
    std::string name_dir = "../data/models/mixed_srpde/" + meshID;
	if(!std::filesystem::create_directory(name_dir)) std::filesystem::create_directory(name_dir);
    
    name_dir += "same_locations_W_empty/";
    if(!std::filesystem::create_directory(name_dir)) std::filesystem::create_directory(name_dir);
	
    // input
    std::string input_dir = name_dir  + "input/";

    if(!std::filesystem::exists(std::filesystem::path(input_dir))) {
        std::cout << "\t --- generating data --- " << std::endl;
        std::filesystem::create_directory(input_dir);

        Eigen::saveMarket(beta, input_dir + "beta.mtx");
        Eigen::saveMarket(alpha, input_dir + "alpha.mtx");
        Eigen::saveMarket(n_obs, input_dir + "n_obs.mtx");
    
        eigen2txt<double>(beta, input_dir + "beta.txt");
        eigen2txt<double>(alpha, input_dir + "alpha.txt");
        eigen2txt<int>(n_obs, input_dir + "n_obs.txt");

        Eigen::saveMarket(x1_(domain.mesh.nodes()), input_dir + "cov_1.mtx");
        eigen2txt<double>(x1_(domain.mesh.nodes()), input_dir + "cov_1.txt");
        
        // generete data
        DMatrix<double> locs = uniform_locs(n_obs(0));
        //double sigma = 0.1; 
        
        for(std::size_t j = 0; j < m; ++j){
            //DMatrix<double> locs = uniform_locs(n_obs(j));
            DMatrix<double> DesignMatrix = DMatrix<double>::Zero(n_obs(j),2);
            
            DesignMatrix.col(0) = x1_(locs); // va in V
            DesignMatrix.col(1) = noise(n_obs(j), 1.0);

            DMatrix<double> f_ = f(locs, j);
            double sigma = 0.05*std::abs(f_.array().maxCoeff() - f_.array().minCoeff()); 
            std::cout << sigma << std::endl;
            auto eps_ = noise(n_obs(j),sigma);
            eigen2txt<double>(eps_, input_dir + "noise_" + std::to_string(j) + ".txt");
            
            auto obs = DesignMatrix * beta + DesignMatrix*alpha.col(j)  + f_ + eps_; 
            
            Eigen::saveMarket(locs, input_dir + "locs_" + std::to_string(j) + ".mtx");
            Eigen::saveMarket(DesignMatrix, input_dir + "DesignMatrix_" + std::to_string(j) + ".mtx");
            Eigen::saveMarket(DesignMatrix, input_dir + "V_" + std::to_string(j) + ".mtx");
            Eigen::saveMarket(obs, input_dir + "obs_" + std::to_string(j) + ".mtx");

            eigen2txt<double>(locs, input_dir + "locs_" + std::to_string(j) + ".txt");
            eigen2txt<double>(DesignMatrix, input_dir + "DesignMatrix_" + std::to_string(j) + ".txt");
            eigen2txt<double>(DesignMatrix, input_dir + "V_" + std::to_string(j) + ".txt");
            eigen2txt<double>(obs, input_dir + "obs_" + std::to_string(j) + ".txt");
            // plots
            f_ = f(domain.mesh.nodes(), j);
            Eigen::saveMarket(f_, input_dir + "f_" + std::to_string(j) + ".mtx");
            eigen2txt<double>(f_, input_dir + "f_" + std::to_string(j) + ".txt");
        } 
    }
     
    // Output directory
	std::string output_dir = name_dir + "output/";
    if(!std::filesystem::exists(std::filesystem::path(output_dir))){ 
        std::filesystem::create_directory(output_dir);
    }
    output_dir = output_dir + "monolithic/";
    if(!std::filesystem::exists(std::filesystem::path(output_dir))){ 
        std::filesystem::create_directory(output_dir);
    }
    
    // import data from files
    for(std::size_t j = 0; j<m; j++){
        std::string Vname = input_dir + "V_" + std::to_string(j) + ".mtx";
        std::string locsname = input_dir + "locs_" + std::to_string(j) + ".mtx";
        std::string yname = input_dir +  "obs_" + std::to_string(j) + ".mtx";
        
        auto V = read_mtx<double>(Vname);
        auto locs = read_mtx<double>(locsname);
        auto obs = read_mtx<double>(yname);
         
        data[j].insert(V_BLOCK, V);
        data[j].insert(Y_BLOCK, obs);
        data[j].insert(LOCS_BLOCK, locs);      
    }//eigen2txt<double>(DesignMatrix.col(1), input_dir + "W_" + std::to_string(j) + ".txt");
            
    
    DMatrix<double> f_ = DMatrix<double>::Zero(m*domain.mesh.nodes().rows(),1);
    for(std::size_t j = 0; j < m; ++j){
        f_.block(j*domain.mesh.nodes().rows(),0, domain.mesh.nodes().rows(),1) = f(domain.mesh.nodes(),j);
    }
    
    // define regularizing PDE
    auto L = -laplacian<FEM>();
    DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.n_elements()*3, 1);
    PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> problem(domain.mesh, L, u);
	
    // define lambda
    double lambda = 1e-3; 

    // getting dimension of the data (number of total observations N)
    std::size_t sum = 0;
    for(std::size_t i=0; i<data.size(); i++){
        sum += data[i].template get<double>(LOCS_BLOCK).rows();
    }

    MixedSRPDE<monolithic> model(problem, Sampling::pointwise);
	
	model.set_lambda_D(lambda);
	model.set_data(data);
    
    auto start = std::chrono::high_resolution_clock::now();

    // solve smoothing problem
    std::cout << "model init" << std::endl;
    model.init();
    std::cout << "model solve" << std::endl;
    model.solve();

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;

    std::cout << "duration: " << duration.count() << std::endl;
    std::cout << "beta: \n" << model.beta() << std::endl;
    std::cout << "alpha: \n" << model.alpha() << std::endl;
   
    for(std::size_t j = 0; j < m; ++j){
        Eigen::saveMarket(model.f().block(j*domain.mesh.nodes().rows(),0, domain.mesh.nodes().rows(),1),
                          output_dir + "estimate_f_" + std::to_string(j) + ".mtx");
        eigen2txt<double>(model.f().block(j*domain.mesh.nodes().rows(),0, domain.mesh.nodes().rows(),1),
                          output_dir + "estimate_f_" + std::to_string(j) + ".txt");

        EXPECT_TRUE( (model.f().block(j*domain.mesh.nodes().rows(),0, domain.mesh.nodes().rows(),1) -
                            f_.block(j*domain.mesh.nodes().rows(),0, domain.mesh.nodes().rows(),1)).array().square().mean() < 1e-2);
    }

    Eigen::saveMarket(model.f(), output_dir + "estimate_f.mtx");
    eigen2txt<double>(model.f(), output_dir + "estimate_f.txt");
    
    Eigen::saveMarket(model.beta(), output_dir + "beta.mtx");
    eigen2txt<double>(model.beta(), output_dir + "beta.txt");
    
    Eigen::saveMarket(model.alpha(), output_dir + "beta.mtx");
    eigen2txt<double>(model.alpha(), output_dir + "alpha.txt");
     
    EXPECT_TRUE(  (model.alpha() - alpha).array().square().mean() < 1e-2 );
    EXPECT_TRUE(  (model.beta() - beta).array().square().mean() < 1e-2 );

}
*/

// --------------------------------------------------------------------------------------------------------
// test monolitico
/*
TEST(mixed_srpde_test, mille_mono_automatico) {
    std::size_t n_patients = 10;
    double na_percentage = 0.;  // percentuale di valori NA 
    int seed = 1234; 
    double mu = 1000.;
    bool different_num_obs = 1; // 1 means different = TRUE

    int q = 2; // numero colonne X
    std::cout << "q: " << q << std::endl;
    int qV = 1; // numero colonne per ogni V_i
    std::cout << "qV: " << qV << std::endl;
	
    DVector<double> beta = generate_beta_coefficients(q, seed); // stesso numero di colonne di X
    std::cout << "beta:\n" << beta << std::endl;

    DMatrix<double> alpha = generate_alpha_coefficients(n_patients, qV, seed); // righe: n_patients (livelli), colonne: qV
    std::cout << "alpha:\n" << alpha << std::endl;
   																					
    // define data
    std::vector<BlockFrame<double, int>> data;
    data.resize(n_patients);

    // define domain 
    std::string meshID = "fine_mesh";
    std::string policyID = "monolithic/";
    std::string locsID = "1000/";
    MeshLoader<Mesh2D> domain(meshID);
    meshID = meshID + "/"; 

	if(!std::filesystem::create_directory("../data/models/mixed_srpde/2D_test2/" + meshID))
	 std::filesystem::create_directory("../data/models/mixed_srpde/2D_test2/" + meshID);
	
	if(!std::filesystem::create_directory("../data/models/mixed_srpde/2D_test2/" + meshID + policyID))
	 std::filesystem::create_directory("../data/models/mixed_srpde/2D_test2/" + meshID + policyID);
	
    // Output directory
    std::string output_dir = "../data/models/mixed_srpde/2D_test2/" + meshID + policyID + locsID;
	
	if(!std::filesystem::create_directory(output_dir)) std::filesystem::create_directory(output_dir);
    
    // Genera dati per tutti i pazienti
    generate_data_for_all_patients(n_patients, beta, alpha, output_dir, seed, na_percentage, mu, different_num_obs);

    // import data from files
    for(std::size_t i = 0; i<n_patients; i++){
    
        std::cout<< "----- i = " << i << " -----" << std::endl;
         
        std::string Wname = "W_" + std::to_string(i);
        std::string Vname = "V_" + std::to_string(i);
        std::string locsname = "locs_" + std::to_string(i);
        std::string yname = "observations_" + std::to_string(i);
        data[i].read_csv<double>(W_BLOCK, "../data/models/mixed_srpde/2D_test2/" + meshID + policyID + locsID + Wname + ".csv");
         
        data[i].read_csv<double>(V_BLOCK, "../data/models/mixed_srpde/2D_test2/" + meshID + policyID + locsID + Vname + ".csv");
       
        data[i].read_csv<double>(Y_BLOCK, "../data/models/mixed_srpde/2D_test2/" + meshID + policyID + locsID + yname + ".csv");
       
        data[i].read_csv<double>(LOCS_BLOCK, "../data/models/mixed_srpde/2D_test2/" + meshID + policyID + locsID + locsname + ".csv");
       
    }
    // f_hat
    DMatrix<double> f_ = DMatrix<double>::Zero(n_patients*domain.mesh.nodes().rows(),1);
    for (std::size_t j = 0; j < n_patients; j++){
    	for (std::size_t i = 0; i < domain.mesh.nodes().rows(); i++){
         	f_(i + j*domain.mesh.nodes().rows(),0) = lambda_np(domain.mesh.nodes()(i,0), domain.mesh.nodes()(i,1),j);
    	}
    }

    // define regularizing PDE
    auto L = -laplacian<FEM>();
    DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.n_elements()*3, 1);
    PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> problem(domain.mesh, L, u);
	
    // define lambda
    double lambda = 1; 

    // getting dimension of the data (number of total observations N)
    std::size_t sum = 0;
    for(std::size_t i=0; i<data.size(); i++){
        sum += data[i].template get<double>(LOCS_BLOCK).rows();
    }

    MixedSRPDE<SpaceOnly,monolithic> model(problem, Sampling::pointwise);
	
	model.set_lambda_D(lambda);
	model.set_data(data);
    model.set_N(sum);

    auto start = std::chrono::high_resolution_clock::now();

    // solve smoothing problem
    std::cout << "model init" << std::endl;
    model.init();
    std::cout << "model solve" << std::endl;
    model.solve();

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;

    std::cout << "duration: " << duration.count() << std::endl;

    std::ofstream output1("model_f.csv");  // Primo file di output
    output1 << "model_f\n";  // Intestazione con il nome del file
    DMatrix<double> data1 = model.f();
    for(std::size_t i = 0; i < data1.size(); ++i) {
        output1 << data1(i) << "\n";  // Scrivi nel primo file
    }
    output1.close();  // Chiudi il primo file

    std::ofstream output2("f_.csv");  // Secondo file di output
    output2 << "f_\n";  // Intestazione con il nome del file
    DMatrix<double> data2 = f_;  // Usa una variabile diversa per il secondo file
    for(std::size_t i = 0; i < data2.size(); ++i) {
        output2 << data2(i) << "\n";  // Scrivi nel secondo file
    }
    output2.close();  // Chiudi il secondo file

    std::cout << "model_f created" << std::endl;

    // std::cout << "Monolithic: " << (model.f() - model.f() ).array().abs().maxCoeff() << std::endl;
    std::cout << "beta: \n" << model.beta() << std::endl;
    std::cout << "alpha: \n" << model.alpha() << std::endl;
    //EXPECT_TRUE(  (model.f() - f_estimate ).array().abs().maxCoeff() < 1e-6 );
    // std::cout << f_.rows() << " " << f_.cols() << std::endl; 
    // std::cout << model.f().rows() << " " << model.f().cols() << std::endl; 
    
    auto differences = (model.f() - f_).array().square().mean();
    auto rmse = std::sqrt(differences);
    std::cout<< "RMSE = " << rmse <<std::endl;

    std::cout << "max model.f(): "<< model.f().array().abs().maxCoeff() << std::endl;
    std::cout << "max f_: "<< (f_).array().abs().maxCoeff() << std::endl; 

    EXPECT_TRUE(  (model.f().head(f_.rows()) - f_ ).array().abs().maxCoeff() < 1e-2 );
    EXPECT_TRUE(  (model.f() - f_ ).array().abs().maxCoeff() < 1e-2 );
    
}
*/

// test iterativo
/*
TEST(mixed_srpde_test, mille_iter_automatico) {
       
    std::size_t n_patients = 10;
    double na_percentage = 0.;  // percentuale di valori NA 
    int seed = 1234; 
    double mu = 1000.;
    bool different_num_obs = 1; // 1 means different = TRUE


    int q = 2;
    int qV = 1;

    DVector<double> beta = generate_beta_coefficients(q, seed);
    std::cout << "beta:\n"  << beta << std::endl;
    
    DMatrix<double> alpha = generate_alpha_coefficients(n_patients, qV, seed);
    std::cout << "alpha:\n" << alpha << std::endl;

    // define data
    std::vector<BlockFrame<double, int>> data;
    data.resize(n_patients);

    // define domain 
    std::string meshID = "fine_mesh";
    std::string policyID = "iterative/";
    std::string locsID = "1000/";
    MeshLoader<Mesh2D> domain(meshID);
    meshID = meshID + "/"; 

    if(!std::filesystem::create_directory("../data/models/mixed_srpde/2D_test2/" + meshID))
	 std::filesystem::create_directory("../data/models/mixed_srpde/2D_test2/" + meshID);
	
	if(!std::filesystem::create_directory("../data/models/mixed_srpde/2D_test2/" + meshID + policyID))
	 std::filesystem::create_directory("../data/models/mixed_srpde/2D_test2/" + meshID + policyID);
	
    // Output directory
    std::string output_dir = "../data/models/mixed_srpde/2D_test2/" + meshID + policyID + locsID;
	if(!std::filesystem::create_directory(output_dir)) std::filesystem::create_directory(output_dir);

    // Genera dati per tutti i pazienti
    generate_data_for_all_patients(n_patients, beta, alpha, output_dir, seed, na_percentage, mu, different_num_obs);

    // import data from files
    for(std::size_t i = 0; i<n_patients; i++){
        std::cout<< "----- i = " << i << " -----" << std::endl;
        std::string Wname = "W_" + std::to_string(i);
        std::string Vname = "V_" + std::to_string(i);
        std::string locsname = "locs_" + std::to_string(i);
        std::string yname = "observations_" + std::to_string(i);
        data[i].read_csv<double>(W_BLOCK, "../data/models/mixed_srpde/2D_test2/" + meshID + policyID + locsID + Wname + ".csv");
        data[i].read_csv<double>(V_BLOCK, "../data/models/mixed_srpde/2D_test2/" + meshID + policyID + locsID + Vname + ".csv");
        data[i].read_csv<double>(Y_BLOCK, "../data/models/mixed_srpde/2D_test2/" + meshID + policyID + locsID + yname + ".csv");
        data[i].read_csv<double>(LOCS_BLOCK, "../data/models/mixed_srpde/2D_test2/" + meshID + policyID + locsID + locsname + ".csv");
    }

    // f_hat
    DMatrix<double> f_ = DMatrix<double>::Zero(n_patients*domain.mesh.nodes().rows(),1);
    for (std::size_t j = 0; j < n_patients; j++){
    	for (std::size_t i = 0; i < domain.mesh.nodes().rows(); i++){
         	f_(i + j*domain.mesh.nodes().rows(),0) = lambda_np(domain.mesh.nodes()(i,0), domain.mesh.nodes()(i,1),j);
    	}
    }
    std::cout << "f_ done" << std::endl;

    // define regularizing PDE
    auto L = -laplacian<FEM>();
    DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.n_elements() * n_patients, 1);
    PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> problem(domain.mesh, L, u);

    // define model
    double lambda = 1;

    // getting dimension of the data (number of total observations N)
    std::size_t sum = 0;
    for(std::size_t i=0; i<data.size(); i++){
        sum += data[i].template get<double>(LOCS_BLOCK).rows();
    }
    
    MixedSRPDE<SpaceOnly,iterative> model(problem, Sampling::pointwise);

    model.set_lambda_D(lambda);

    model.set_data(data);
    model.set_N(sum);
    
    auto start = std::chrono::high_resolution_clock::now();

    // solve smoothing problem
    model.init();
    model.solve();

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;

    std::cout << "duration: " << duration.count() << std::endl;

    std::ofstream output("model_f.csv");
    DMatrix<double> data1 = model.f();
    for(std::size_t i = 0; i < data1.size(); ++i){
        output << data1(i) << "\n";
    }
    output.close();

    std::ofstream output2("f_.csv");  // Secondo file di output
    output2 << "f_\n";  // Intestazione con il nome del file
    DMatrix<double> data2 = f_;  // Usa una variabile diversa per il secondo file
    for(std::size_t i = 0; i < data2.size(); ++i) {
        output2 << data2(i) << "\n";  // Scrivi nel secondo file
    }
    output2.close();  // Chiudi il secondo file

    std::cout << "beta: " << model.beta() << std::endl;
    std::cout << "alpha: " << model.alpha() << std::endl;

    auto differences = (model.f() - f_).array().square().mean();
    auto rmse = std::sqrt(differences);
    std::cout<< "RMSE = " << rmse <<std::endl;

    std::cout << "max model.f(): "<< model.f().array().abs().maxCoeff() << std::endl;
    std::cout << "max f_: "<< (f_).array().abs().maxCoeff() << std::endl; 

    EXPECT_TRUE(  (model.f().head(f_.rows()) - f_ ).array().abs().maxCoeff() < 1e-2 );
    EXPECT_TRUE(  (model.f() - f_ ).array().abs().maxCoeff() < 1e-2 );
}
*/
/*
// PARAMETRIC TEST 
struct TestParams {
    int patientsN;
    double mu;
    int seed;
    double na_percentage;
    std::string meshID;
    double lambda;
    bool locsy;
};

// const std::vector<TestParams> kPets = {
//     {3, 1000., 1234, 0., "unit_square_coarse", 0.1},
//     {3, 1000., 32874, 0., "unit_square_coarse", 0.1},
//     {10, 1000., 12894, 0., "unit_square_coarse", 0.1},
//     {10, 1000., 18937, 0., "unit_square_coarse", 0.1},
//     {15, 1000., 32847, 0., "unit_square_coarse", 0.1},
//     {15, 1000., 23784, 0., "unit_square_coarse", 0.1},
//     {20, 1000., 2453, 0., "unit_square_coarse", 0.1},
//     {20, 1000., 6575, 0., "unit_square_coarse", 0.1},
//     {30, 1000., 4352, 0., "unit_square_coarse", 0.1},
//     {30, 1000., 1332, 0., "unit_square_coarse", 0.1},
//     // {60, 1000., 2784, 0., "unit_square_coarse", 0.1},
//     // {60, 1000., 23784, 0., "unit_square_coarse", 0.1},
//     // {90, 1000., 32784, 0., "unit_square_coarse", 0.1},
//     // {90, 1000., 274, 0., "unit_square_coarse", 0.1},
//     // {100, 1000., 38247, 0., "unit_square_coarse", 0.1},
//     // {100, 1000., 1389, 0., "unit_square_coarse", 0.1},
//     // {500, 1000., 1389, 0., "unit_square_coarse", 0.1},
//     {3, 1000., 1234, 0., "unit_square_coarse", 0.1},
//     {6, 1000., 32874, 0., "unit_square_coarse", 0.1},
//     {10, 1000., 4783, 0., "unit_square_coarse", 0.1},
//     {12, 1000., 12894, 0., "unit_square_coarse", 0.1},
//     {18, 1000., 18937, 0., "unit_square_coarse", 0.1},
//     {20, 1000., 28394, 0., "unit_square_coarse", 0.1},
//     {24, 1000., 32847, 0., "unit_square_coarse", 0.1},
//     {28, 1000., 23784, 0., "unit_square_coarse", 0.1},
//     {30, 1000., 482947, 0., "unit_square_coarse", 0.1},
//     {32, 1000., 2453, 0., "unit_square_coarse", 0.1},
//     {34, 1000., 6575, 0., "unit_square_coarse", 0.1},
//     {36, 1000., 4352, 0., "unit_square_coarse", 0.1},
//     {40, 1000., 1332, 0., "unit_square_coarse", 0.1},
// };

std::vector<double> linspace(double start, double end, int n) {
    std::vector<double> result; result.reserve(n); 
    double step = (end - start) / (n - 1); 
    for (int i = 0; i < n; ++i) { result.push_back(start + i * step); } 
    return result; 
}

std::vector<TestParams> GenerateTestParamsConditional() {
    std::vector<TestParams> testParamsList;
    std::vector<int> patientsNs = {5, 10, 15, 20, 25, 30};
    std::vector<double> mus = {1000.0, 2500.0, 5000.0, 7500.0, 10000.0};
    std::uniform_int_distribution<> seeds(1000, 50000);
    std::mt19937 gen(std::random_device{}());
    std::vector<double> na_percentages = {0.0, 0.05, 0.1, 0.15, 0.2};
    std::string meshID = "unit_square";
    double lambda = 1.;
    // std::vector<double> lambdas = linspace(1e-5, 1, 30);
    std::vector<bool> locsyn = {1, 0};

    for (const auto& patientsN : patientsNs) {
        for (const auto& mu : mus) {
            for(const auto& na_percentage : na_percentages){
                for(const auto& locsy : locsyn){
                    TestParams params = {patientsN, mu, seeds(gen), na_percentage, meshID, lambda, locsy};
                    testParamsList.push_back(params);
                }
            }
        }
    }
    return testParamsList;
}

class MixedSRPDETest : public testing::TestWithParam<TestParams> {
  // You can implement all the usual fixture class members here.
  // To access the test parameter, call GetParam() from class TestWithParam<T>.
};

INSTANTIATE_TEST_SUITE_P(Pets, MixedSRPDETest, testing::ValuesIn(GenerateTestParamsConditional()));

TEST_P(MixedSRPDETest, monolithic) {

    TestParams params = GetParam();

    std::stringstream ss;
    ss << params.patientsN << " " << params.mu << " " << params.seed << " " << "monolithic" << " " << params.na_percentage << " " << params.meshID << " " << params.lambda;
    
    std::string filename = "prova_test.txt";
    // header:
    // patientsN mu seed policyID na_percentage meshID lambda rmse rmse_beta duration maxcoeff locs

    std::string meshID = params.meshID;
    std::string locsID = "1000";
    std::string policyID = "monolithic";
	std::size_t n_patients = params.patientsN;
    double mu = params.mu;
    int seed = params.seed;
    std::string locsyn1 = "same";
    if(params.locsy){
        locsyn1 = "diff";
    }

    auto start = std::chrono::high_resolution_clock::now();

    policyID = policyID + "/"; 
    locsID = locsID + "/"; 
	
    DVector<double> a(2);
    a(0) = -3.0;
    a(1) = 4.0; // Coefficienti per X: n_obs x 2
    DVector<double> b = generate_b_coefficients(n_patients, seed); // Coefficienti per V

    // ss << " ["; for (double value : b) { ss << value << " "; } ss << "]";
   																					
    double na_percentage = params.na_percentage;  // percentuale di valori NA 

    // define data
    std::vector<BlockFrame<double, int>> data;
    data.resize(n_patients);

    // define domain 
    MeshLoader<Mesh2D> domain(meshID);
    meshID = meshID + "/"; 

	if(!std::filesystem::create_directory("../data/models/mixed_srpde/2D_test2/" + meshID))
	 std::filesystem::create_directory("../data/models/mixed_srpde/2D_test2/" + meshID);
	
	if(!std::filesystem::create_directory("../data/models/mixed_srpde/2D_test2/" + meshID + policyID))
	 std::filesystem::create_directory("../data/models/mixed_srpde/2D_test2/" + meshID + policyID);
	
    std::string output_dir = "../data/models/mixed_srpde/2D_test2/" + meshID + policyID + locsID;
	if(!std::filesystem::create_directory(output_dir)) std::filesystem::create_directory(output_dir);
    
    generate_data_for_all_patients(n_patients, a, b, output_dir, seed, na_percentage, mu, params.locsy);

    // import data from files
    for(std::size_t i = 0; i<n_patients; i++){
        std::string Wname = "W_" + std::to_string(i);
        std::string Vname = "V_" + std::to_string(i);
        std::string locsname = "locs_" + std::to_string(i);
        std::string yname = "observations_" + std::to_string(i);
        data[i].read_csv<double>(W_BLOCK, "../data/models/mixed_srpde/2D_test2/" + meshID + policyID + locsID + Wname + ".csv");
        data[i].read_csv<double>(V_BLOCK, "../data/models/mixed_srpde/2D_test2/" + meshID + policyID + locsID + Vname + ".csv");
        data[i].read_csv<double>(Y_BLOCK, "../data/models/mixed_srpde/2D_test2/" + meshID + policyID + locsID + yname + ".csv");
        data[i].read_csv<double>(LOCS_BLOCK, "../data/models/mixed_srpde/2D_test2/" + meshID + policyID + locsID + locsname + ".csv");
    }

    // f_hat
    DMatrix<double> f_ = DMatrix<double>::Zero(n_patients*domain.mesh.nodes().rows(),1);
    for (std::size_t j = 0; j < n_patients; j++){
    	for (std::size_t i = 0; i < domain.mesh.nodes().rows(); i++){
         	f_(i + j*domain.mesh.nodes().rows(),0) = lambda_np(domain.mesh.nodes()(i,0), domain.mesh.nodes()(i,1),j);
    	}
    }

    // define regularizing PDE
    auto L = -laplacian<FEM>();
    DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.n_elements()*3, 1);
    PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> problem(domain.mesh, L, u);
	
    // define lambda
    double lambda = params.lambda; 

    // getting dimension of the data (number of total observations N)
    std::size_t sum = 0;
    for(std::size_t i=0; i<data.size(); i++){
        sum += data[i].template get<double>(LOCS_BLOCK).rows();
    }

    MixedSRPDE<SpaceOnly,monolithic> model(problem, Sampling::pointwise);
	
	model.set_lambda_D(lambda);
	model.set_data(data);
    model.set_N(sum);
    
    // solve smoothing problem
    model.init();
    model.solve();

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;

    std::ofstream output1("model_f.csv"); 
    output1 << "model_f\n";
    DMatrix<double> data1 = model.f();
    for(std::size_t i = 0; i < data1.size(); ++i) {
        output1 << data1(i) << "\n"; }
    output1.close(); 

    std::ofstream output2("f_.csv"); 
    output2 << "f_\n"; 
    DMatrix<double> data2 = f_;  
    for(std::size_t i = 0; i < data2.size(); ++i) {
        output2 << data2(i) << "\n"; }
    output2.close();  

    // std::cout << "beta: " << model.beta() << std::endl;
    // std::cout << "alpha: " << model.alpha() << std::endl;
    
    auto differences = (model.f() - f_).array().square().mean();
    auto rmse = std::sqrt(differences);
    std::cout<< "RMSE = " << rmse <<std::endl;

    auto differences_beta = (model.beta()-a).array().square().mean();
    auto rmse_beta = std::sqrt(differences_beta);
    std::cout<< "RMSE beta = " << rmse_beta <<std::endl;

    std::cout << "max model.f(): "<< model.f().array().abs().maxCoeff() << std::endl;
    std::cout << "max f_: "<< (f_).array().abs().maxCoeff() << std::endl; 

    // ss << " ["; for (double value : model.betanp()) { ss << value << " "; } ss << "] ";
    ss << " " << rmse << " " << rmse_beta << " " << duration.count() << " " << (model.f().head(f_.rows()) - f_ ).array().abs().maxCoeff() << " " << locsyn1 << std::endl;

    std::cout << "max diff: " << (model.f().head(f_.rows()) - f_ ).array().abs().maxCoeff() << std::endl;

    std::ofstream outputFile; outputFile.open(filename, std::ios_base::app);
    if (outputFile.is_open()) { outputFile << ss.str();  outputFile.close();
    } else { std::cerr << "Unable to open file for writing" << std::endl; }

    // std::stringstream ss1;
    // double nbasis_i = 0;
    // auto rmse_i = std::sqrt((model.f(0) - f_.block(nbasis_i, 0, model.f(0).rows(), 1)).array().square().mean());
    
    // // ss1 << "patients_id mu seed policyID na_percentage meshID lambda rmse rmse_beta duration maxcoeff locs rmse_i\n";
    // for (std::size_t i = 0; i<n_patients; i++){
    //     rmse_i = std::sqrt((model.f(i) - f_.block(nbasis_i, 0, model.f(i).rows(), 1)).array().square().mean());
    //     nbasis_i += model.f(i).rows();
    //     ss1 << i << " " << params.mu << " " << params.seed << " " << policyID << " " << params.na_percentage << " " << params.meshID << " " 
    //     << params.lambda << " " << rmse << " " << rmse_beta << " " << duration.count() << " " << (model.f().head(f_.rows()) - f_ ).array().abs().maxCoeff()
    //     << " diff " << rmse_i << "\n";
    // }
    // std::ofstream outputFile1; outputFile1.open("rmse_i.txt", std::ios_base::app);
    // if (outputFile1.is_open()) { outputFile1 << ss1.str();  outputFile1.close();
    // } else { std::cerr << "Unable to open file for writing" << std::endl; }

    EXPECT_TRUE(  (model.f().head(f_.rows()) - f_ ).array().abs().maxCoeff() < 0.5 );
}

TEST_P(MixedSRPDETest, iterative) {

    TestParams params = GetParam();

    std::stringstream ss;
    ss << params.patientsN << " " << params.mu << " " << params.seed << " " << "iterative" << " " << params.na_percentage << " " << params.meshID << " " << params.lambda;
    
    std::string filename = "prova_test.txt";
    // header:
    // patientsN mu seed policyID na_percentage meshID lambda rmse rmse_beta duration maxcoeff locs

    std::string meshID = params.meshID;
    std::string locsID = "1000";
    std::string policyID = "iterative";
	std::size_t n_patients = params.patientsN;
    double mu = params.mu;
    int seed = params.seed;
    std::string locsyn1 = "same";
    if(params.locsy){
        locsyn1 = "diff";
    }

    auto start = std::chrono::high_resolution_clock::now();

    policyID = policyID + "/"; 
    locsID = locsID + "/"; 
	
    DVector<double> a(2);
    a(0) = -3.0;
    a(1) = 4.0; // Coefficienti per X: n_obs x 2
    DVector<double> b = generate_b_coefficients(n_patients, params.seed); // Coefficienti per V

    // ss << " ["; for (double value : b) { ss << value << " "; } ss << "]";
   																					
    double na_percentage = params.na_percentage;  // percentuale di valori NA 

    // define data
    std::vector<BlockFrame<double, int>> data;
    data.resize(n_patients);

    // define domain 
    MeshLoader<Mesh2D> domain(meshID);
    meshID = meshID + "/"; 

	if(!std::filesystem::create_directory("../data/models/mixed_srpde/2D_test2/" + meshID))
	 std::filesystem::create_directory("../data/models/mixed_srpde/2D_test2/" + meshID);
	
	if(!std::filesystem::create_directory("../data/models/mixed_srpde/2D_test2/" + meshID + policyID))
	 std::filesystem::create_directory("../data/models/mixed_srpde/2D_test2/" + meshID + policyID);
	
    std::string output_dir = "../data/models/mixed_srpde/2D_test2/" + meshID + policyID + locsID;
	if(!std::filesystem::create_directory(output_dir)) std::filesystem::create_directory(output_dir);
    
    generate_data_for_all_patients(n_patients, a, b, output_dir, seed, na_percentage, mu, params.locsy);

    // import data from files
    for(std::size_t i = 0; i<n_patients; i++){
        std::string Wname = "W_" + std::to_string(i);
        std::string Vname = "V_" + std::to_string(i);
        std::string locsname = "locs_" + std::to_string(i);
        std::string yname = "observations_" + std::to_string(i);
        data[i].read_csv<double>(W_BLOCK, "../data/models/mixed_srpde/2D_test2/" + meshID + policyID + locsID + Wname + ".csv");
        data[i].read_csv<double>(V_BLOCK, "../data/models/mixed_srpde/2D_test2/" + meshID + policyID + locsID + Vname + ".csv");
        data[i].read_csv<double>(Y_BLOCK, "../data/models/mixed_srpde/2D_test2/" + meshID + policyID + locsID + yname + ".csv");
        data[i].read_csv<double>(LOCS_BLOCK, "../data/models/mixed_srpde/2D_test2/" + meshID + policyID + locsID + locsname + ".csv");
    }

    // f_hat
    DMatrix<double> f_ = DMatrix<double>::Zero(n_patients*domain.mesh.nodes().rows(),1);
    for (std::size_t j = 0; j < n_patients; j++){
    	for (std::size_t i = 0; i < domain.mesh.nodes().rows(); i++){
         	f_(i + j*domain.mesh.nodes().rows(),0) = lambda_np(domain.mesh.nodes()(i,0), domain.mesh.nodes()(i,1),j);
    	}
    }

    // define regularizing PDE
    auto L = -laplacian<FEM>();
    DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.n_elements()*3, 1);
    PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> problem(domain.mesh, L, u);
	
    // define lambda
    double lambda = params.lambda; 

    // getting dimension of the data (number of total observations N)
    std::size_t sum = 0;
    for(std::size_t i=0; i<data.size(); i++){
        sum += data[i].template get<double>(LOCS_BLOCK).rows();
    }

    MixedSRPDE<SpaceOnly,iterative> model(problem, Sampling::pointwise);
	
	model.set_lambda_D(lambda);
	model.set_data(data);
    model.set_N(sum);
    
    // solve smoothing problem
    model.init();
    model.solve();

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;

    std::ofstream output1("model_f.csv"); 
    output1 << "model_f\n";
    DMatrix<double> data1 = model.f();
    for(std::size_t i = 0; i < data1.size(); ++i) {
        output1 << data1(i) << "\n"; }
    output1.close(); 

    std::ofstream output2("f_.csv"); 
    output2 << "f_\n"; 
    DMatrix<double> data2 = f_;  
    for(std::size_t i = 0; i < data2.size(); ++i) {
        output2 << data2(i) << "\n"; }
    output2.close();  

    // std::cout << "beta: " << model.beta() << std::endl;
    // std::cout << "alpha: " << model.alpha() << std::endl;
    
    auto differences = (model.f() - f_).array().square().mean();
    auto rmse = std::sqrt(differences);
    std::cout<< "RMSE = " << rmse <<std::endl;

    auto differences_beta = (model.beta()-a).array().square().mean();
    auto rmse_beta = std::sqrt(differences_beta);
    std::cout<< "RMSE beta = " << rmse_beta <<std::endl;

    std::cout << "max model.f(): "<< model.f().array().abs().maxCoeff() << std::endl;
    std::cout << "max f_: "<< (f_).array().abs().maxCoeff() << std::endl; 

    // ss << " ["; for (double value : model.betanp()) { ss << value << " "; } ss << "] ";
    ss << " " << rmse << " " << rmse_beta << " " << duration.count() << " " <<  (model.f().head(f_.rows()) - f_ ).array().abs().maxCoeff() << " " << locsyn1 << std::endl;

    std::cout << "max diff: " << (model.f().head(f_.rows()) - f_ ).array().abs().maxCoeff() << std::endl;

    std::ofstream outputFile; outputFile.open(filename, std::ios_base::app);
    if (outputFile.is_open()) { outputFile << ss.str();  outputFile.close();
    } else { std::cerr << "Unable to open file for writing" << std::endl; }

    // std::stringstream ss1;
    // double nbasis_i = 0;
    // auto rmse_i = std::sqrt((model.f(0) - f_.block(nbasis_i, 0, model.f(0).rows(), 1)).array().square().mean());
    
    // // ss1 << "patients_id mu seed policyID na_percentage meshID lambda rmse rmse_beta duration maxcoeff locs rmse_i\n";
    // for (std::size_t i = 0; i<n_patients; i++){
    //     rmse_i = std::sqrt((model.f(i) - f_.block(nbasis_i, 0, model.f(i).rows(), 1)).array().square().mean());
    //     nbasis_i += model.f(i).rows();
    //     ss1 << i << " " << params.mu << " " << params.seed << " " << policyID << " " << params.na_percentage << " " << params.meshID << " " 
    //     << params.lambda << " " << rmse << " " << rmse_beta << " " << duration.count() << " " << (model.f().head(f_.rows()) - f_ ).array().abs().maxCoeff()
    //     << " diff " << rmse_i << "\n";
    // }
    // std::ofstream outputFile1; outputFile1.open("rmse_i.txt", std::ios_base::app);
    // if (outputFile1.is_open()) { outputFile1 << ss1.str();  outputFile1.close();
    // } else { std::cerr << "Unable to open file for writing" << std::endl; }

    EXPECT_TRUE(  (model.f().head(f_.rows()) - f_ ).array().abs().maxCoeff() < 0.5 );
}
*/
/*
// test iterativo LOCAZIONI DIFFERENTI
TEST(mixed_srpde_test, mille_iter) {

    // define domain 
    std::size_t n_patients = 3;
    std::string meshID = "c_shaped_1";
    std::string policyID = "richardson/";
    std::string locsID = "1000/";
    MeshLoader<Mesh2D> domain(meshID);
    meshID = meshID + "/"; 
    std::vector<BlockFrame<double, int>> data;
    
    // resize del vector di BlockFrame -> un BlockFrame per ogni paziente
    // questa informazione andrebbe passata in qualche modo
    data.resize(n_patients);

    // import data from files
    for(std::size_t i = 0; i<n_patients; i++){
        std::string Wname = "W_" + std::to_string(i+1) + ".csv";
        std::string Vname = "V_" + std::to_string(i+1);
        std::string locsname = "locs_" + std::to_string(i+1);
        std::string yname = "observations_" + std::to_string(i+1);
        data[i].read_csv<double>(W_BLOCK, "../data/models/mixed_srpde/2D_test1/" + meshID + policyID + locsID + Wname);
        data[i].read_csv<double>(V_BLOCK, "../data/models/mixed_srpde/2D_test1/" + meshID + policyID + locsID + Vname + ".csv");
        data[i].read_csv<double>(Y_BLOCK, "../data/models/mixed_srpde/2D_test1/" + meshID + policyID + locsID + yname + ".csv");
        data[i].read_csv<double>(LOCS_BLOCK, "../data/models/mixed_srpde/2D_test1/" + meshID + policyID + locsID + locsname + ".csv");
    }

    // getting dimension of the data (number of total observations N)
    std::size_t sum = 0;
    for(std::size_t i=0; i<data.size(); i++){
        sum += data[i].template get<double>(LOCS_BLOCK).rows();
    }

    // define regularizing PDE
    auto L = -laplacian<FEM>();
    DMatrix<double> u =  DMatrix<double>::Zero(domain.mesh.n_elements() * n_patients, 1); //DMatrix<double>::Zero(sum, 1);
    PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> problem(domain.mesh, L, u);
    
    // define model
    double lambda = 1;
    MixedSRPDE<SpaceOnly,iterative> model(problem, Sampling::pointwise);
    
    // setting model parameters
    model.set_lambda_D(lambda);
    model.set_data(data);
    model.set_N(sum);
    
    // solve smoothing problem
    model.init();
    model.solve();
    
    // importing the estimation of f made with the other library
    DMatrix<double> f_estimate = read_csv<double>("../data/models/mixed_srpde/2D_test1/" + 
    												 meshID + policyID + locsID + "f_hat.csv");
    
    std::cout << "Iterative: " << (model.f() - f_estimate ).array().abs().maxCoeff() << std::endl;
    EXPECT_TRUE(  (model.f() - f_estimate ).array().abs().maxCoeff() < 1e-6 );
}
*/

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
