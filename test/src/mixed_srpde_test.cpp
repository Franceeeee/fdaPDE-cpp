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
// along with this program.  If not, see <http://www.gnu.org/licenses>.

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

void write_table_noHeaders(const DMatrix<double>& M, const std::string& filename = "data.txt") {
    std::ofstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Error: Unable to open file " << filename << std::endl;
        return;
    }

    eigen2txt<double>(M, filename, false);  // Directly write the matrix without any headers
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


auto uniform_locs(std::size_t n, std::mt19937 gen) {
    std::uniform_real_distribution<> dis(0.0, 1.0);
    DMatrix<double> locs = DMatrix<double>::Zero(n,2);
    for (std::size_t i = 0; i < n; ++i) {
        locs(i,0) = dis(gen);  // x
        locs(i,1) = dis(gen);  // y
    }
    return locs;
}


auto f(DMatrix<double> locs, int id = 0) { 
    std::mt19937 gen(id);
    DMatrix<double> res = DMatrix<double>::Zero(locs.rows(),1);
    std::uniform_int_distribution<> dis(-1,1);
    double a = dis(gen);
    double phi = dis(gen)*2;
        for(std::size_t i = 0; i < locs.rows(); ++i){
            if(id == 0)
                res(i,0) = std::sin(2*fdapde::testing::pi*locs(i,0))*
                                std::sin(2*fdapde::testing::pi*locs(i,1));
            else if (id == 1)
                res(i,0) = 1.0 - locs(i,0) - locs(i,1);
            else if (id == 2)
                res(i,0) = 1-std::sin(fdapde::testing::pi*locs(i,0))*
                                std::cos(fdapde::testing::pi*locs(i,1));
            else
                res(i,0) = a*std::cos(fdapde::testing::pi*locs(i,0)+phi)*std::cos(fdapde::testing::pi*locs(i,1));
    }
        return res;
}


auto noise(std::size_t n, double sigma, std::mt19937 gen){
    DMatrix<double> res = DMatrix<double>::Zero(n,1);
    std::normal_distribution<> __noise(0.0, sigma);
    for(std::size_t i = 0; i < n; ++i){
        res(i,0) = __noise(gen);
    }
    return res;
}


auto x1_(DMatrix<double> locs){
    DMatrix<double> res = DMatrix<double>::Zero(locs.rows(),1);
    for(std::size_t i = 0; i < locs.rows(); ++i){
                res(i,0) = 1-(locs(i,0)-0.5)*(locs(i,0)-0.5) -(locs(i,1)-0.5)*(locs(i,1)-0.5); 
    }   
    return res;
}

// maschera di NA
auto create_na_mask(int size, double na_percentage, std::mt19937 gen) {
    std::vector<bool> mask(size, false);  
    int num_na = static_cast<int>(std::round(size * na_percentage));
    std::uniform_int_distribution<> dis(0, size - 1);
    for (int i = 0; i < num_na; ++i) {
        int index;
        do { index = dis(gen); } while (mask[index]); 
        mask[index] = true;
    }
    return mask;
};

DMatrix<double> generateAlpha(int m, int seed) {
    
    std::mt19937 gen(seed);
    
    DMatrix<double> alpha = DMatrix<double>::Zero(m, 1);
        std::uniform_real_distribution<double> distribution(-1.0, 1.0);

    double sum = 0.0;
    for (int i = 0; i < m; ++i) {
        alpha(i, 0) = distribution(gen);
        sum += alpha(i, 0);
    }

    // Normalize 
    for (int i = 0; i < m; ++i) {
        alpha(i, 0) /= sum;
    }

    return alpha;
}

void appendFileContent(const std::string& sourceFile, const std::string& targetFile) {
    std::ifstream source(sourceFile);  // Open source file for reading
    std::ofstream target(targetFile, std::ios::app);  // Open target file in append mode

    if (!source.is_open() || !target.is_open()) {
        std::cerr << "Error: Unable to open file(s)." << std::endl;
        return;
    }

    std::string line;
    while (std::getline(source, line)) {
        target << line << '\n';  // Append each line from source file to target file
    }

    source.close();
    target.close();
}


// TEST(mixed_srpde_test, same_locations_diff_NA) {

//     DMatrix<double> na_percentage_vec = DMatrix<double>::Zero(5,1);
//     na_percentage_vec(0,0) = 0.0; na_percentage_vec(1,0) = 0.05; na_percentage_vec(2,0) = 0.1; 
//     na_percentage_vec(3,0) = 0.15; na_percentage_vec(4,0) = 0.2;

//     // -------------- PARAMETERS ------------------------------------------------

//     std::string test_name = "same_locations_diff_NA/";
//     int seed = 23872; 
//     double lambda = 1e-3; 
//     int memory = 3;             // GMRES param
//     bool same_locs = 1;
//     std::string meshID = "unit_square";
//     std::size_t m = 3;
//     std::size_t n_sim = 100;     
    
//     DMatrix<double> beta = DMatrix<double>::Zero(2,1);
//     beta(0,0) = -2.; beta(1,0) = 1.;
//     DMatrix<double> alpha = DMatrix<double>::Zero(3,1);
//     alpha(0,0) = -0.5; alpha(1,0) = 0.; alpha(2,0) = 0.5;
//     DMatrix<int> n_obs = DMatrix<int>::Zero(5,m);
//     n_obs(0,0) = 500; n_obs(1,0) = 1000; n_obs(2,0) = 2000; 
//     n_obs(3,0) = 4000; n_obs(4,0) = 8000;


//     // ---------------------------------------------------------------------------

//     MeshLoader<Mesh2D> domain(meshID);
//     meshID = meshID + "/"; 
//     std::string name_dir = "../data/models/mixed_srpde/" + meshID;
// 	if(!std::filesystem::exists(std::filesystem::path(name_dir))) std::filesystem::create_directory(name_dir);
    
//     name_dir += test_name;
//     if(!std::filesystem::exists(std::filesystem::path(name_dir))) std::filesystem::create_directory(name_dir);
	
//     // input data 
//     std::string input_dir = name_dir  + "input/";

//     if(!std::filesystem::exists(std::filesystem::path(input_dir))) {

//         std::filesystem::create_directory(input_dir);
//             // Parametri della distribuzione gaussiana (media e varianza per riga)
//         std::vector<double> means = {500, 1000, 2000, 4000, 8000};
//         std::vector<double> stddevs = {50, 100, 200, 400, 800}; // Deviazioni standard

//         // std::cout << "\t --- generating data --- " << std::endl;

//         Eigen::saveMarket(beta, input_dir + "beta.mtx");
//         Eigen::saveMarket(alpha, input_dir + "alpha.mtx");
//         Eigen::saveMarket(n_obs, input_dir + "n_obs.mtx");

//         eigen2txt<double>(beta, input_dir + "beta.txt");
//         eigen2txt<double>(alpha, input_dir + "alpha.txt");
//         eigen2txt<int>(n_obs, input_dir + "n_obs.txt");

//         Eigen::saveMarket(x1_(domain.mesh.nodes()), input_dir + "cov_1.mtx");
//         eigen2txt<double>(x1_(domain.mesh.nodes()), input_dir + "cov_1.txt");
//         for( std::size_t j=0; j < m; ++j){
//             DMatrix<double> f_ = f(domain.mesh.nodes(), j);
//             Eigen::saveMarket(f_, input_dir + "f_" + std::to_string(j) + ".mtx");
//             eigen2txt<double>(f_, input_dir + "f_" + std::to_string(j) + ".txt");
//         }

//         for(std::size_t n = 0; n < n_obs.rows(); ++n){
            
//             // generete data
//             std::string data_dir = input_dir + std::to_string(n_obs(n,0)) + "/";
//             std::filesystem::create_directory(data_dir);
                
//             for(std::size_t sim=0; sim<n_sim; ++sim){

//                 double na_percentage = na_percentage_vec(static_cast<int>(sim/20));
//                 // std::cout<< "NA perc:" << na_percentage <<std::endl;
//                 std::mt19937 gen(seed+ sim);
                
//                 // Ciclo per riempire la matrice
//                 for (int i = 0; i < n_obs.rows(); ++i) {
//                     std::normal_distribution<> dist(means[i], stddevs[i]);
//                     for (int j = 1; j < n_obs.cols(); ++j) {
//                         n_obs(i, j) = static_cast<int>(dist(gen)); // Cast a int per valori interi
//                     }
//                 }
//                 std::string simul_dir = data_dir + std::to_string(sim) + "/"; 
//                 std::filesystem::create_directory(simul_dir);

//                 for(std::size_t j = 0; j < m; ++j){
//                     DMatrix<double> locs = uniform_locs(n_obs(n,j), gen);
//                     DMatrix<double> DesignMatrix = DMatrix<double>::Zero(n_obs(n,j),2);
                
//                     DesignMatrix.col(0) = x1_(locs); // va in V
//                     DesignMatrix.col(1) = noise(n_obs(n,j), 1.0, gen);

//                     DMatrix<double> f_ = f(locs, j);
//                     double sigma = 0.05*std::abs(f_.array().maxCoeff() - f_.array().minCoeff()); 
//                     auto eps_ = noise(n_obs(n,j), sigma, gen);
//                     eigen2txt<double>(eps_, simul_dir + "noise_" + std::to_string(j) + ".txt");
                
//                     DMatrix<double> obs = DesignMatrix * beta + DesignMatrix.col(0)*alpha(j,0)  + f_ + eps_; 
                    
//                     auto na_mask = create_na_mask(n_obs(n,j), na_percentage, gen); 
//                     for (int i = 0; i < n_obs(n,j); ++i) { 
//                         if (na_mask[i]) {
//                             obs(i, 0) = std::numeric_limits<double>::quiet_NaN();  
//                         }
//                     }
                
//                     Eigen::saveMarket(locs, simul_dir + "locs_" + std::to_string(j) + ".mtx");
//                     Eigen::saveMarket(DesignMatrix, simul_dir + "DesignMatrix_" + std::to_string(j) + ".mtx");
//                     Eigen::saveMarket(DesignMatrix.col(1), simul_dir + "W_" + std::to_string(j) + ".mtx");
//                     Eigen::saveMarket(DesignMatrix.col(0), simul_dir + "V_" + std::to_string(j) + ".mtx");
//                     Eigen::saveMarket(obs, simul_dir + "obs_" + std::to_string(j) + ".mtx");

//                     eigen2txt<double>(locs, simul_dir + "locs_" + std::to_string(j) + ".txt");
//                     eigen2txt<double>(DesignMatrix, simul_dir + "DesignMatrix_" + std::to_string(j) + ".txt");
//                     eigen2txt<double>(DesignMatrix.col(1), simul_dir + "W_" + std::to_string(j) + ".txt");
//                     eigen2txt<double>(DesignMatrix.col(0), simul_dir + "V_" + std::to_string(j) + ".txt");
//                     eigen2txt<double>(obs, simul_dir + "obs_" + std::to_string(j) + ".txt");
//                 }
//             }
//         }
//     }
     
//     // Output directory
// 	std::string output_dir = name_dir + "output/";
//     if(!std::filesystem::exists(std::filesystem::path(output_dir))){ 
//         std::filesystem::create_directory(output_dir);
//     }
    
//     std::vector<std::string> solution_policy = {"monolithic", "richardson"};

//     // import data from files
//     std::vector<std::string> header = {"time_init", "time_solve", "time",
//                                        "rmse_f","rmse_f_1", "rmse_f_2","rmse_f_3", 
//                                        "rmse_beta","rmse_alpha","n_obs","na_perc"};

//     DMatrix<double> results_mono = DMatrix<double>::Zero( n_sim*n_obs.rows(), header.size());
//     DMatrix<double> results_rich = DMatrix<double>::Zero( n_sim*n_obs.rows(), header.size());
//     DMatrix<double> results_gmres = DMatrix<double>::Zero( n_sim*n_obs.rows(), header.size());

//     for(std::size_t n = 0; n < n_obs.rows(); ++n){ 
//         output_dir = name_dir + "output/"; // + "monolithic/";
//         output_dir += std::to_string(n_obs(n,0)) + "/" ;
        
//         std::string data_dir = input_dir + std::to_string(n_obs(n,0)) + "/";
//         if(!std::filesystem::exists(std::filesystem::path(output_dir))) std::filesystem::create_directory(output_dir);
        
//     for(std::size_t sim = 0; sim < n_sim; ++sim){

//         std::string simul_dir = data_dir + std::to_string(sim) + "/"; 
//         std::string result_dir = output_dir + std::to_string(sim) + "/";
//         if(!std::filesystem::exists(std::filesystem::path(result_dir))) std::filesystem::create_directory(result_dir);

//         std::string output_monolithic = result_dir + "monolithic/";
//         std::string output_richardson = result_dir + "richardson/";
//         if(!std::filesystem::exists(std::filesystem::path(output_monolithic))) std::filesystem::create_directory(output_monolithic);
//         if(!std::filesystem::exists(std::filesystem::path(output_richardson))) std::filesystem::create_directory(output_richardson);

//         std::vector<BlockFrame<double, int>> data;
//         data.resize(m);
        
//         for(std::size_t j = 0; j<m; j++){
//             std::string Wname = simul_dir + "W_" + std::to_string(j) + ".mtx";
//             std::string Vname = simul_dir + "V_" + std::to_string(j) + ".mtx";
//             std::string locsname = simul_dir + "locs_" + std::to_string(j) + ".mtx";
//             std::string yname = simul_dir +  "obs_" + std::to_string(j) + ".mtx";
//             auto W = read_mtx<double>(Wname);
//             auto V = read_mtx<double>(Vname);
//             auto locs = read_mtx<double>(locsname);
//             auto obs = read_mtx<double>(yname);

//             // --- NaN handling ---
//             // before passing data to the model, we need to:
//             // - remove NaN from y 
//             // - remove the corresponding rows in W_BLOCK,V_BLOCK,LOCS_BLOCK
//             // - update N (number of observations)

//             // save position of non-na values
//             std::vector<int> validIndices;
//             for (int i = 0; i < obs.size(); ++i) {
//                 if (!std::isnan(obs(i,0))) {
//                     validIndices.push_back(i);
//                 }
//             }
//             // std::cout << "valid_indices: " << validIndices.size() << std::endl;
//             // std::cout << "obs_rows: " << obs.rows() << std::endl;
            
//             if(validIndices.size() != obs.rows()){
//                 // create new y_ with only valid entries
//                 DVector<double> y_new(validIndices.size());
//                 for (size_t i = 0; i < validIndices.size(); ++i) {
//                     y_new(i) = obs(validIndices[i]);
//                 }
//                 obs = y_new; 

//                 // delete rows from W,V,locs
//                 DMatrix<double> W_new(validIndices.size(), W.cols());
//                 DMatrix<double> V_new(validIndices.size(), V.cols());
//                 DMatrix<double> locs_new(validIndices.size(), locs.cols());
//                 for (size_t i = 0; i < validIndices.size(); ++i) {
//                     W_new.row(i) = W.row(validIndices[i]);
//                     V_new.row(i) = V.row(validIndices[i]);
//                     locs_new.row(i) = locs.row(validIndices[i]);
//                 }
//                 W = W_new;
//                 V = V_new;
//                 locs = locs_new;
//                 same_locs = 0;
//             }
            
//             data[j].insert(W_BLOCK, W);
//             data[j].insert(V_BLOCK, V);
//             data[j].insert(Y_BLOCK, obs);
//             data[j].insert(LOCS_BLOCK, locs);      
//         }
    
//         DMatrix<double> f_ = DMatrix<double>::Zero(m*domain.mesh.nodes().rows(),1);
//         for(std::size_t j = 0; j < m; ++j){
//             f_.block(j*domain.mesh.nodes().rows(),0, domain.mesh.nodes().rows(),1) =   f(domain.mesh.nodes(),j);
//         }
    
//         // define regularizing PDE
//         auto L = -laplacian<FEM>();
//         DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.n_elements()*3, 1);
//         PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> problem(domain.mesh, L, u);

//         // monolithic 
//         MixedSRPDE<monolithic> monolithic_(problem, Sampling::pointwise, same_locs);
//         monolithic_.set_lambda_D(lambda);
// 	    monolithic_.set_data(data);
        
//         auto start = std::chrono::high_resolution_clock::now();
//         monolithic_.init();
//         std::chrono::duration<double> duration = std::chrono::high_resolution_clock::now() - start;
//         results_mono(sim + n_sim*n, 0) = duration.count();

//         start = std::chrono::high_resolution_clock::now();
//         monolithic_.solve();
//         duration = std::chrono::high_resolution_clock::now() - start;
//         results_mono(sim + n_sim*n, 1) = duration.count();
//         results_mono(sim + n_sim*n, 2) = results_mono(sim + n_sim*n, 0) + results_mono(sim + n_sim*n, 1);

//         // iterative
//         MixedSRPDE<iterative> richardson_(problem, Sampling::pointwise, same_locs);
//         richardson_.set_lambda_D(lambda);
// 	    richardson_.set_data(data);
//         // richardson_.set_GMRES_params(0);

//         start = std::chrono::high_resolution_clock::now();
//         richardson_.init();
//         duration = std::chrono::high_resolution_clock::now() - start;
//         results_rich(sim + n_sim*n, 0) = duration.count();

//         start = std::chrono::high_resolution_clock::now();
//         richardson_.solve();
//         duration = std::chrono::high_resolution_clock::now() - start;
//         results_rich(sim + n_sim*n, 1) = duration.count();
//         results_rich(sim + n_sim*n, 2) = results_rich(sim + n_sim*n, 0) + results_rich(sim + n_sim*n, 1);
        
//         // iterative GMRES
//         MixedSRPDE<iterative> gmres_(problem, Sampling::pointwise, same_locs);
//         gmres_.set_lambda_D(lambda);
// 	    gmres_.set_data(data);
//         gmres_.set_GMRES_params(memory);

//         start = std::chrono::high_resolution_clock::now();
//         gmres_.init();
//         duration = std::chrono::high_resolution_clock::now() - start;
//         results_gmres(sim + n_sim*n, 0) = duration.count();

//         start = std::chrono::high_resolution_clock::now();
//         gmres_.solve();
//         duration = std::chrono::high_resolution_clock::now() - start;
//         results_gmres(sim + n_sim*n, 1) = duration.count();
//         results_gmres(sim + n_sim*n, 2) = results_gmres(sim + n_sim*n, 0) + results_gmres(sim + n_sim*n, 1);
        

//         // RMSEs
//         results_mono(sim + n_sim*n, 3) = (monolithic_.f() - f_).array().square().mean();
//         results_rich(sim + n_sim*n, 3) = (richardson_.f() - f_).array().square().mean();
//         results_gmres(sim + n_sim*n, 3) = (gmres_.f() - f_).array().square().mean();
        

//         for(std::size_t j = 0; j < m; ++j){
//             Eigen::saveMarket(monolithic_.f().block(j*domain.mesh.nodes().rows(),0, domain.mesh.nodes().rows(),1),
//                           output_monolithic + "estimate_f_" + std::to_string(j) + ".mtx");
//             eigen2txt<double>(monolithic_.f().block(j*domain.mesh.nodes().rows(),0, domain.mesh.nodes().rows(),1),
//                           output_monolithic + "estimate_f_" + std::to_string(j) + ".txt");

//             results_mono(sim + n_sim*n, 4+j) = (monolithic_.f().block(j*domain.mesh.nodes().rows(),0, domain.mesh.nodes().rows(),1) -
//                                                 f_.block(j*domain.mesh.nodes().rows(),0, domain.mesh.nodes().rows(),1)).array().square().mean();

//             Eigen::saveMarket(richardson_.f().block(j*domain.mesh.nodes().rows(),0, domain.mesh.nodes().rows(),1),
//                           output_richardson + "estimate_f_" + std::to_string(j) + ".mtx");
//             eigen2txt<double>(richardson_.f().block(j*domain.mesh.nodes().rows(),0, domain.mesh.nodes().rows(),1),
//                           output_richardson + "estimate_f_" + std::to_string(j) + ".txt");

//             results_rich(sim + n_sim*n, 4+j) = (richardson_.f().block(j*domain.mesh.nodes().rows(),0, domain.mesh.nodes().rows(),1) -
//                                                 f_.block(j*domain.mesh.nodes().rows(),0, domain.mesh.nodes().rows(),1)).array().square().mean();
//             results_gmres(sim + n_sim*n, 4+j) = (gmres_.f().block(j*domain.mesh.nodes().rows(),0, domain.mesh.nodes().rows(),1) -
//                                                 f_.block(j*domain.mesh.nodes().rows(),0, domain.mesh.nodes().rows(),1)).array().square().mean();
//         }

//         Eigen::saveMarket(monolithic_.f(), output_monolithic + "estimate_f.mtx");
//         eigen2txt<double>(monolithic_.f(), output_monolithic + "estimate_f.txt");
//         Eigen::saveMarket(richardson_.f(), output_richardson + "estimate_f.mtx");
//         eigen2txt<double>(richardson_.f(), output_richardson + "estimate_f.txt");
    
//         Eigen::saveMarket(monolithic_.beta(), output_monolithic + "beta.mtx");
//         eigen2txt<double>(monolithic_.beta(), output_monolithic + "beta.txt");
//         Eigen::saveMarket(richardson_.beta(), output_richardson + "beta.mtx");
//         eigen2txt<double>(richardson_.beta(), output_richardson + "beta.txt");
    
//         Eigen::saveMarket(monolithic_.alpha(), output_monolithic + "beta.mtx");
//         eigen2txt<double>(monolithic_.alpha(), output_monolithic + "alpha.txt");
//         Eigen::saveMarket(richardson_.alpha(), output_richardson + "beta.mtx");
//         eigen2txt<double>(richardson_.alpha(), output_richardson + "alpha.txt");

//         results_mono(sim + n_sim*n, 7) =  (monolithic_.beta() - beta).array().square().mean();
//         results_mono(sim + n_sim*n, 8) =  (monolithic_.alpha() - alpha).array().square().mean();
        
//         results_rich(sim + n_sim*n, 7) =  (richardson_.beta() - beta).array().square().mean();
//         results_rich(sim + n_sim*n, 8) =  (richardson_.alpha() - alpha).array().square().mean();
        
//         results_gmres(sim + n_sim*n, 7) =  (gmres_.beta() - beta).array().square().mean();
//         results_gmres(sim + n_sim*n, 8) =  (gmres_.alpha() - alpha).array().square().mean();

//         results_mono(sim + n_sim*n,9) = n_obs(n, 0);
//         results_rich(sim + n_sim*n,9) = n_obs(n, 0);
//         results_gmres(sim + n_sim*n,9) = n_obs(n, 0);

//         double na = na_percentage_vec(static_cast<int>(sim/20));

//         results_mono(sim + n_sim*n,10) = na;
//         results_rich(sim + n_sim*n,10) = na;
//         results_gmres(sim + n_sim*n,10) = na;

//         EXPECT_TRUE(  (monolithic_.beta() - beta).array().square().mean() < 1e-2 );
//         EXPECT_TRUE(  (monolithic_.alpha() - alpha).array().square().mean() < 1e-2 );

//         EXPECT_TRUE(  (richardson_.beta() - beta).array().square().mean() < 1e-2 );
//         EXPECT_TRUE(  (richardson_.alpha() - alpha).array().square().mean() < 1e-2 );
//     }
//     }

//     write_table(results_mono, header, name_dir + "output/" + solution_policy[0] + ".txt");
//     write_table(results_rich, header, name_dir + "output/" + solution_policy[1] + ".txt");
//     write_table(results_gmres, header, name_dir + "output/" + solution_policy[1] + "_gmres.txt");
// }

// TEST(mixed_srpde_test, diff_locations_diff_NA) {

//     DMatrix<double> na_percentage_vec = DMatrix<double>::Zero(5,1);
//     na_percentage_vec(0,0) = 0.0; na_percentage_vec(1,0) = 0.05; na_percentage_vec(2,0) = 0.1; 
//     na_percentage_vec(3,0) = 0.15; na_percentage_vec(4,0) = 0.2;

//     // -------------- PARAMETERS ------------------------------------------------

//     std::string test_name = "diff_locations_diff_NA/";
//     int seed = 23872; 
//     double lambda = 1e-3; 
//     int memory = 3;             // GMRES param
//     bool same_locs = 0;
//     std::string meshID = "unit_square";
//     std::size_t m = 3;
//     std::size_t n_sim = 100;     
    
//     DMatrix<double> beta = DMatrix<double>::Zero(2,1);
//     beta(0,0) = -2.; beta(1,0) = 1.;
//     DMatrix<double> alpha = DMatrix<double>::Zero(3,1);
//     alpha(0,0) = -0.5; alpha(1,0) = 0.; alpha(2,0) = 0.5;
//     DMatrix<int> n_obs = DMatrix<int>::Zero(5,m);
//     n_obs(0,0) = 500; n_obs(1,0) = 1000; n_obs(2,0) = 2000; 
//     n_obs(3,0) = 4000; n_obs(4,0) = 8000;


//     // ---------------------------------------------------------------------------

//     MeshLoader<Mesh2D> domain(meshID);
//     meshID = meshID + "/"; 
//     std::string name_dir = "../data/models/mixed_srpde/" + meshID;
// 	if(!std::filesystem::exists(std::filesystem::path(name_dir))) std::filesystem::create_directory(name_dir);
    
//     name_dir += test_name;
//     if(!std::filesystem::exists(std::filesystem::path(name_dir))) std::filesystem::create_directory(name_dir);
	
//     // input data 
//     std::string input_dir = name_dir  + "input/";

//     if(!std::filesystem::exists(std::filesystem::path(input_dir))) {

//         std::filesystem::create_directory(input_dir);
//             // Parametri della distribuzione gaussiana (media e varianza per riga)
//         std::vector<double> means = {500, 1000, 2000, 4000, 8000};
//         std::vector<double> stddevs = {50, 100, 200, 400, 800}; // Deviazioni standard

//         // std::cout << "\t --- generating data --- " << std::endl;

//         Eigen::saveMarket(beta, input_dir + "beta.mtx");
//         Eigen::saveMarket(alpha, input_dir + "alpha.mtx");
//         Eigen::saveMarket(n_obs, input_dir + "n_obs.mtx");

//         eigen2txt<double>(beta, input_dir + "beta.txt");
//         eigen2txt<double>(alpha, input_dir + "alpha.txt");
//         eigen2txt<int>(n_obs, input_dir + "n_obs.txt");

//         Eigen::saveMarket(x1_(domain.mesh.nodes()), input_dir + "cov_1.mtx");
//         eigen2txt<double>(x1_(domain.mesh.nodes()), input_dir + "cov_1.txt");
//         for( std::size_t j=0; j < m; ++j){
//             DMatrix<double> f_ = f(domain.mesh.nodes(), j);
//             Eigen::saveMarket(f_, input_dir + "f_" + std::to_string(j) + ".mtx");
//             eigen2txt<double>(f_, input_dir + "f_" + std::to_string(j) + ".txt");
//         }

//         for(std::size_t n = 0; n < n_obs.rows(); ++n){
            
//             // generete data
//             std::string data_dir = input_dir + std::to_string(n_obs(n,0)) + "/";
//             std::filesystem::create_directory(data_dir);
                
//             for(std::size_t sim=0; sim<n_sim; ++sim){

//                 double na_percentage = na_percentage_vec(static_cast<int>(sim/20));
//                 // std::cout<< "NA perc:" << na_percentage <<std::endl;
//                 std::mt19937 gen(seed+ sim);
                
//                 // Ciclo per riempire la matrice
//                 for (int i = 0; i < n_obs.rows(); ++i) {
//                     std::normal_distribution<> dist(means[i], stddevs[i]);
//                     for (int j = 1; j < n_obs.cols(); ++j) {
//                         n_obs(i, j) = static_cast<int>(dist(gen)); // Cast a int per valori interi
//                     }
//                 }
//                 std::string simul_dir = data_dir + std::to_string(sim) + "/"; 
//                 std::filesystem::create_directory(simul_dir);

//                 for(std::size_t j = 0; j < m; ++j){
//                     DMatrix<double> locs = uniform_locs(n_obs(n,j), gen);
//                     DMatrix<double> DesignMatrix = DMatrix<double>::Zero(n_obs(n,j),2);
                
//                     DesignMatrix.col(0) = x1_(locs); // va in V
//                     DesignMatrix.col(1) = noise(n_obs(n,j), 1.0, gen);

//                     DMatrix<double> f_ = f(locs, j);
//                     double sigma = 0.05*std::abs(f_.array().maxCoeff() - f_.array().minCoeff()); 
//                     auto eps_ = noise(n_obs(n,j), sigma, gen);
//                     eigen2txt<double>(eps_, simul_dir + "noise_" + std::to_string(j) + ".txt");
                
//                     DMatrix<double> obs = DesignMatrix * beta + DesignMatrix.col(0)*alpha(j,0)  + f_ + eps_; 
                    
//                     auto na_mask = create_na_mask(n_obs(n,j), na_percentage, gen); 
//                     for (int i = 0; i < n_obs(n,j); ++i) { 
//                         if (na_mask[i]) {
//                             obs(i, 0) = std::numeric_limits<double>::quiet_NaN();  
//                         }
//                     }
                
//                     Eigen::saveMarket(locs, simul_dir + "locs_" + std::to_string(j) + ".mtx");
//                     Eigen::saveMarket(DesignMatrix, simul_dir + "DesignMatrix_" + std::to_string(j) + ".mtx");
//                     Eigen::saveMarket(DesignMatrix.col(1), simul_dir + "W_" + std::to_string(j) + ".mtx");
//                     Eigen::saveMarket(DesignMatrix.col(0), simul_dir + "V_" + std::to_string(j) + ".mtx");
//                     Eigen::saveMarket(obs, simul_dir + "obs_" + std::to_string(j) + ".mtx");

//                     eigen2txt<double>(locs, simul_dir + "locs_" + std::to_string(j) + ".txt");
//                     eigen2txt<double>(DesignMatrix, simul_dir + "DesignMatrix_" + std::to_string(j) + ".txt");
//                     eigen2txt<double>(DesignMatrix.col(1), simul_dir + "W_" + std::to_string(j) + ".txt");
//                     eigen2txt<double>(DesignMatrix.col(0), simul_dir + "V_" + std::to_string(j) + ".txt");
//                     eigen2txt<double>(obs, simul_dir + "obs_" + std::to_string(j) + ".txt");
//                 }
//             }
//         }
//     }
     
//     // Output directory
// 	std::string output_dir = name_dir + "output/";
//     if(!std::filesystem::exists(std::filesystem::path(output_dir))){ 
//         std::filesystem::create_directory(output_dir);
//     }
    
//     std::vector<std::string> solution_policy = {"monolithic", "richardson"};

//     // import data from files
//     std::vector<std::string> header = {"time_init", "time_solve", "time",
//                                        "rmse_f","rmse_f_1", "rmse_f_2","rmse_f_3", 
//                                        "rmse_beta","rmse_alpha","n_obs","na_perc"};

//     DMatrix<double> results_mono = DMatrix<double>::Zero( n_sim*n_obs.rows(), header.size());
//     DMatrix<double> results_rich = DMatrix<double>::Zero( n_sim*n_obs.rows(), header.size());
//     DMatrix<double> results_gmres = DMatrix<double>::Zero( n_sim*n_obs.rows(), header.size());

//     for(std::size_t n = 0; n < n_obs.rows(); ++n){ 
//         output_dir = name_dir + "output/"; // + "monolithic/";
//         output_dir += std::to_string(n_obs(n,0)) + "/" ;
        
//         std::string data_dir = input_dir + std::to_string(n_obs(n,0)) + "/";
//         if(!std::filesystem::exists(std::filesystem::path(output_dir))) std::filesystem::create_directory(output_dir);
        
//     for(std::size_t sim = 0; sim < n_sim; ++sim){

//         std::string simul_dir = data_dir + std::to_string(sim) + "/"; 
//         std::string result_dir = output_dir + std::to_string(sim) + "/";
//         if(!std::filesystem::exists(std::filesystem::path(result_dir))) std::filesystem::create_directory(result_dir);

//         std::string output_monolithic = result_dir + "monolithic/";
//         std::string output_richardson = result_dir + "richardson/";
//         if(!std::filesystem::exists(std::filesystem::path(output_monolithic))) std::filesystem::create_directory(output_monolithic);
//         if(!std::filesystem::exists(std::filesystem::path(output_richardson))) std::filesystem::create_directory(output_richardson);

//         std::vector<BlockFrame<double, int>> data;
//         data.resize(m);
        
//         for(std::size_t j = 0; j<m; j++){
//             std::string Wname = simul_dir + "W_" + std::to_string(j) + ".mtx";
//             std::string Vname = simul_dir + "V_" + std::to_string(j) + ".mtx";
//             std::string locsname = simul_dir + "locs_" + std::to_string(j) + ".mtx";
//             std::string yname = simul_dir +  "obs_" + std::to_string(j) + ".mtx";
//             auto W = read_mtx<double>(Wname);
//             auto V = read_mtx<double>(Vname);
//             auto locs = read_mtx<double>(locsname);
//             auto obs = read_mtx<double>(yname);

//             // --- NaN handling ---
//             // before passing data to the model, we need to:
//             // - remove NaN from y 
//             // - remove the corresponding rows in W_BLOCK,V_BLOCK,LOCS_BLOCK
//             // - update N (number of observations)

//             // save position of non-na values
//             std::vector<int> validIndices;
//             for (int i = 0; i < obs.size(); ++i) {
//                 if (!std::isnan(obs(i,0))) {
//                     validIndices.push_back(i);
//                 }
//             }
//             // std::cout << "valid_indices: " << validIndices.size() << std::endl;
//             // std::cout << "obs_rows: " << obs.rows() << std::endl;
            
//             if(validIndices.size() != obs.rows()){
//                 // create new y_ with only valid entries
//                 DVector<double> y_new(validIndices.size());
//                 for (size_t i = 0; i < validIndices.size(); ++i) {
//                     y_new(i) = obs(validIndices[i]);
//                 }
//                 obs = y_new; 

//                 // delete rows from W,V,locs
//                 DMatrix<double> W_new(validIndices.size(), W.cols());
//                 DMatrix<double> V_new(validIndices.size(), V.cols());
//                 DMatrix<double> locs_new(validIndices.size(), locs.cols());
//                 for (size_t i = 0; i < validIndices.size(); ++i) {
//                     W_new.row(i) = W.row(validIndices[i]);
//                     V_new.row(i) = V.row(validIndices[i]);
//                     locs_new.row(i) = locs.row(validIndices[i]);
//                 }
//                 W = W_new;
//                 V = V_new;
//                 locs = locs_new;
//                 same_locs = 0;
//             }
            
//             data[j].insert(W_BLOCK, W);
//             data[j].insert(V_BLOCK, V);
//             data[j].insert(Y_BLOCK, obs);
//             data[j].insert(LOCS_BLOCK, locs);      
//         }
    
//         DMatrix<double> f_ = DMatrix<double>::Zero(m*domain.mesh.nodes().rows(),1);
//         for(std::size_t j = 0; j < m; ++j){
//             f_.block(j*domain.mesh.nodes().rows(),0, domain.mesh.nodes().rows(),1) =   f(domain.mesh.nodes(),j);
//         }
    
//         // define regularizing PDE
//         auto L = -laplacian<FEM>();
//         DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.n_elements()*3, 1);
//         PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> problem(domain.mesh, L, u);

//         // monolithic 
//         MixedSRPDE<monolithic> monolithic_(problem, Sampling::pointwise, same_locs);
//         monolithic_.set_lambda_D(lambda);
// 	    monolithic_.set_data(data);
        
//         auto start = std::chrono::high_resolution_clock::now();
//         monolithic_.init();
//         std::chrono::duration<double> duration = std::chrono::high_resolution_clock::now() - start;
//         results_mono(sim + n_sim*n, 0) = duration.count();

//         start = std::chrono::high_resolution_clock::now();
//         monolithic_.solve();
//         duration = std::chrono::high_resolution_clock::now() - start;
//         results_mono(sim + n_sim*n, 1) = duration.count();
//         results_mono(sim + n_sim*n, 2) = results_mono(sim + n_sim*n, 0) + results_mono(sim + n_sim*n, 1);

//         // iterative
//         MixedSRPDE<iterative> richardson_(problem, Sampling::pointwise, same_locs);
//         richardson_.set_lambda_D(lambda);
// 	    richardson_.set_data(data);
//         // richardson_.set_GMRES_params(0);

//         start = std::chrono::high_resolution_clock::now();
//         richardson_.init();
//         duration = std::chrono::high_resolution_clock::now() - start;
//         results_rich(sim + n_sim*n, 0) = duration.count();

//         start = std::chrono::high_resolution_clock::now();
//         richardson_.solve();
//         duration = std::chrono::high_resolution_clock::now() - start;
//         results_rich(sim + n_sim*n, 1) = duration.count();
//         results_rich(sim + n_sim*n, 2) = results_rich(sim + n_sim*n, 0) + results_rich(sim + n_sim*n, 1);
        
//         // iterative GMRES
//         MixedSRPDE<iterative> gmres_(problem, Sampling::pointwise, same_locs);
//         gmres_.set_lambda_D(lambda);
// 	    gmres_.set_data(data);
//         gmres_.set_GMRES_params(memory);

//         start = std::chrono::high_resolution_clock::now();
//         gmres_.init();
//         duration = std::chrono::high_resolution_clock::now() - start;
//         results_gmres(sim + n_sim*n, 0) = duration.count();

//         start = std::chrono::high_resolution_clock::now();
//         gmres_.solve();
//         duration = std::chrono::high_resolution_clock::now() - start;
//         results_gmres(sim + n_sim*n, 1) = duration.count();
//         results_gmres(sim + n_sim*n, 2) = results_gmres(sim + n_sim*n, 0) + results_gmres(sim + n_sim*n, 1);
        

//         // RMSEs
//         results_mono(sim + n_sim*n, 3) = (monolithic_.f() - f_).array().square().mean();
//         results_rich(sim + n_sim*n, 3) = (richardson_.f() - f_).array().square().mean();
//         results_gmres(sim + n_sim*n, 3) = (gmres_.f() - f_).array().square().mean();
        

//         for(std::size_t j = 0; j < m; ++j){
//             Eigen::saveMarket(monolithic_.f().block(j*domain.mesh.nodes().rows(),0, domain.mesh.nodes().rows(),1),
//                           output_monolithic + "estimate_f_" + std::to_string(j) + ".mtx");
//             eigen2txt<double>(monolithic_.f().block(j*domain.mesh.nodes().rows(),0, domain.mesh.nodes().rows(),1),
//                           output_monolithic + "estimate_f_" + std::to_string(j) + ".txt");

//             results_mono(sim + n_sim*n, 4+j) = (monolithic_.f().block(j*domain.mesh.nodes().rows(),0, domain.mesh.nodes().rows(),1) -
//                                                 f_.block(j*domain.mesh.nodes().rows(),0, domain.mesh.nodes().rows(),1)).array().square().mean();

//             Eigen::saveMarket(richardson_.f().block(j*domain.mesh.nodes().rows(),0, domain.mesh.nodes().rows(),1),
//                           output_richardson + "estimate_f_" + std::to_string(j) + ".mtx");
//             eigen2txt<double>(richardson_.f().block(j*domain.mesh.nodes().rows(),0, domain.mesh.nodes().rows(),1),
//                           output_richardson + "estimate_f_" + std::to_string(j) + ".txt");

//             results_rich(sim + n_sim*n, 4+j) = (richardson_.f().block(j*domain.mesh.nodes().rows(),0, domain.mesh.nodes().rows(),1) -
//                                                 f_.block(j*domain.mesh.nodes().rows(),0, domain.mesh.nodes().rows(),1)).array().square().mean();
//             results_gmres(sim + n_sim*n, 4+j) = (gmres_.f().block(j*domain.mesh.nodes().rows(),0, domain.mesh.nodes().rows(),1) -
//                                                 f_.block(j*domain.mesh.nodes().rows(),0, domain.mesh.nodes().rows(),1)).array().square().mean();
//         }

//         Eigen::saveMarket(monolithic_.f(), output_monolithic + "estimate_f.mtx");
//         eigen2txt<double>(monolithic_.f(), output_monolithic + "estimate_f.txt");
//         Eigen::saveMarket(richardson_.f(), output_richardson + "estimate_f.mtx");
//         eigen2txt<double>(richardson_.f(), output_richardson + "estimate_f.txt");
    
//         Eigen::saveMarket(monolithic_.beta(), output_monolithic + "beta.mtx");
//         eigen2txt<double>(monolithic_.beta(), output_monolithic + "beta.txt");
//         Eigen::saveMarket(richardson_.beta(), output_richardson + "beta.mtx");
//         eigen2txt<double>(richardson_.beta(), output_richardson + "beta.txt");
    
//         Eigen::saveMarket(monolithic_.alpha(), output_monolithic + "beta.mtx");
//         eigen2txt<double>(monolithic_.alpha(), output_monolithic + "alpha.txt");
//         Eigen::saveMarket(richardson_.alpha(), output_richardson + "beta.mtx");
//         eigen2txt<double>(richardson_.alpha(), output_richardson + "alpha.txt");

//         results_mono(sim + n_sim*n, 7) =  (monolithic_.beta() - beta).array().square().mean();
//         results_mono(sim + n_sim*n, 8) =  (monolithic_.alpha() - alpha).array().square().mean();
        
//         results_rich(sim + n_sim*n, 7) =  (richardson_.beta() - beta).array().square().mean();
//         results_rich(sim + n_sim*n, 8) =  (richardson_.alpha() - alpha).array().square().mean();
        
//         results_gmres(sim + n_sim*n, 7) =  (gmres_.beta() - beta).array().square().mean();
//         results_gmres(sim + n_sim*n, 8) =  (gmres_.alpha() - alpha).array().square().mean();

//         results_mono(sim + n_sim*n,9) = n_obs(n, 0);
//         results_rich(sim + n_sim*n,9) = n_obs(n, 0);
//         results_gmres(sim + n_sim*n,9) = n_obs(n, 0);

//         double na = na_percentage_vec(static_cast<int>(sim/20));

//         results_mono(sim + n_sim*n,10) = na;
//         results_rich(sim + n_sim*n,10) = na;
//         results_gmres(sim + n_sim*n,10) = na;

//         EXPECT_TRUE(  (monolithic_.beta() - beta).array().square().mean() < 1e-2 );
//         EXPECT_TRUE(  (monolithic_.alpha() - alpha).array().square().mean() < 1e-2 );

//         EXPECT_TRUE(  (richardson_.beta() - beta).array().square().mean() < 1e-2 );
//         EXPECT_TRUE(  (richardson_.alpha() - alpha).array().square().mean() < 1e-2 );
//     }
//     }

//     write_table(results_mono, header, name_dir + "output/" + solution_policy[0] + ".txt");
//     write_table(results_rich, header, name_dir + "output/" + solution_policy[1] + ".txt");
//     write_table(results_gmres, header, name_dir + "output/" + solution_policy[1] + "_gmres.txt");
// }

// TEST(mixed_srpde_test, diff_locs_large_n) {

//     DMatrix<double> na_percentage_vec = DMatrix<double>::Zero(2,1);
//     na_percentage_vec(0,0) = 0.0; na_percentage_vec(1,0) = 0.1; 

//     // -------------- PARAMETERS ------------------------------------------------

//     std::string test_name = "diff_locs_large_n/";
//     int seed = 23872; 
//     double lambda = 1e-3; 
//     int memory = 3;             // GMRES param
//     bool same_locs = 0;
//     std::string meshID = "unit_square_coarse";
//     std::size_t m = 3;
//     std::size_t n_sim = 10;     
    
//     DMatrix<double> beta = DMatrix<double>::Zero(2,1);
//     beta(0,0) = -2.; beta(1,0) = 1.;
//     DMatrix<double> alpha = DMatrix<double>::Zero(3,1);
//     alpha(0,0) = -0.5; alpha(1,0) = 0.; alpha(2,0) = 0.5;
//     DMatrix<int> n_obs = DMatrix<int>::Zero(4,m);
//     n_obs(0,0) = 1e5; n_obs(1,0) = 1e6; n_obs(2,0) = 1e7; 
//     n_obs(3,0) = 1e8;


//     // ---------------------------------------------------------------------------

//     MeshLoader<Mesh2D> domain(meshID);
//     meshID = meshID + "/"; 
//     std::string name_dir = "../data/models/mixed_srpde/" + meshID;
// 	if(!std::filesystem::exists(std::filesystem::path(name_dir))) std::filesystem::create_directory(name_dir);
    
//     name_dir += test_name;
//     if(!std::filesystem::exists(std::filesystem::path(name_dir))) std::filesystem::create_directory(name_dir);
	
//     // input data 
//     std::string input_dir = name_dir  + "input/";

//     if(!std::filesystem::exists(std::filesystem::path(input_dir))) {

//         std::filesystem::create_directory(input_dir);
//             // Parametri della distribuzione gaussiana (media e varianza per riga)
//         std::vector<double> means = {500, 1000, 2000, 4000, 8000};
//         std::vector<double> stddevs = {50, 100, 200, 400, 800}; // Deviazioni standard

//         // std::cout << "\t --- generating data --- " << std::endl;

//         Eigen::saveMarket(beta, input_dir + "beta.mtx");
//         Eigen::saveMarket(alpha, input_dir + "alpha.mtx");
//         Eigen::saveMarket(n_obs, input_dir + "n_obs.mtx");

//         eigen2txt<double>(beta, input_dir + "beta.txt");
//         eigen2txt<double>(alpha, input_dir + "alpha.txt");
//         eigen2txt<int>(n_obs, input_dir + "n_obs.txt");

//         Eigen::saveMarket(x1_(domain.mesh.nodes()), input_dir + "cov_1.mtx");
//         eigen2txt<double>(x1_(domain.mesh.nodes()), input_dir + "cov_1.txt");
//         for( std::size_t j=0; j < m; ++j){
//             DMatrix<double> f_ = f(domain.mesh.nodes(), j);
//             Eigen::saveMarket(f_, input_dir + "f_" + std::to_string(j) + ".mtx");
//             eigen2txt<double>(f_, input_dir + "f_" + std::to_string(j) + ".txt");
//         }

//         for(std::size_t n = 0; n < n_obs.rows(); ++n){
            
//             // generete data
//             std::string data_dir = input_dir + std::to_string(n_obs(n,0)) + "/";
//             std::filesystem::create_directory(data_dir);
                
//             for(std::size_t sim=0; sim<n_sim; ++sim){

//                 double na_percentage = na_percentage_vec(static_cast<int>(sim/20));
//                 // std::cout<< "NA perc:" << na_percentage <<std::endl;
//                 std::mt19937 gen(seed+ sim);
                
//                 // Ciclo per riempire la matrice
//                 for (int i = 0; i < n_obs.rows(); ++i) {
//                     std::normal_distribution<> dist(means[i], stddevs[i]);
//                     for (int j = 1; j < n_obs.cols(); ++j) {
//                         n_obs(i, j) = static_cast<int>(dist(gen)); // Cast a int per valori interi
//                     }
//                 }
//                 std::string simul_dir = data_dir + std::to_string(sim) + "/"; 
//                 std::filesystem::create_directory(simul_dir);

//                 for(std::size_t j = 0; j < m; ++j){
//                     DMatrix<double> locs = uniform_locs(n_obs(n,j), gen);
//                     DMatrix<double> DesignMatrix = DMatrix<double>::Zero(n_obs(n,j),2);
                
//                     DesignMatrix.col(0) = x1_(locs); // va in V
//                     DesignMatrix.col(1) = noise(n_obs(n,j), 1.0, gen);

//                     DMatrix<double> f_ = f(locs, j);
//                     double sigma = 0.05*std::abs(f_.array().maxCoeff() - f_.array().minCoeff()); 
//                     auto eps_ = noise(n_obs(n,j), sigma, gen);
//                     eigen2txt<double>(eps_, simul_dir + "noise_" + std::to_string(j) + ".txt");
                
//                     DMatrix<double> obs = DesignMatrix * beta + DesignMatrix.col(0)*alpha(j,0)  + f_ + eps_; 
                    
//                     auto na_mask = create_na_mask(n_obs(n,j), na_percentage, gen); 
//                     for (int i = 0; i < n_obs(n,j); ++i) { 
//                         if (na_mask[i]) {
//                             obs(i, 0) = std::numeric_limits<double>::quiet_NaN();  
//                         }
//                     }
                
//                     Eigen::saveMarket(locs, simul_dir + "locs_" + std::to_string(j) + ".mtx");
//                     Eigen::saveMarket(DesignMatrix, simul_dir + "DesignMatrix_" + std::to_string(j) + ".mtx");
//                     Eigen::saveMarket(DesignMatrix.col(1), simul_dir + "W_" + std::to_string(j) + ".mtx");
//                     Eigen::saveMarket(DesignMatrix.col(0), simul_dir + "V_" + std::to_string(j) + ".mtx");
//                     Eigen::saveMarket(obs, simul_dir + "obs_" + std::to_string(j) + ".mtx");

//                     eigen2txt<double>(locs, simul_dir + "locs_" + std::to_string(j) + ".txt");
//                     eigen2txt<double>(DesignMatrix, simul_dir + "DesignMatrix_" + std::to_string(j) + ".txt");
//                     eigen2txt<double>(DesignMatrix.col(1), simul_dir + "W_" + std::to_string(j) + ".txt");
//                     eigen2txt<double>(DesignMatrix.col(0), simul_dir + "V_" + std::to_string(j) + ".txt");
//                     eigen2txt<double>(obs, simul_dir + "obs_" + std::to_string(j) + ".txt");
//                 }
//             }
//         }
//     }
     
//     // Output directory
// 	std::string output_dir = name_dir + "output/";
//     if(!std::filesystem::exists(std::filesystem::path(output_dir))){ 
//         std::filesystem::create_directory(output_dir);
//     }
    
//     std::vector<std::string> solution_policy = {"monolithic", "richardson"};

//     // import data from files
//     std::vector<std::string> header = {"time_init", "time_solve", "time",
//                                        "rmse_f","rmse_f_1", "rmse_f_2","rmse_f_3", 
//                                        "rmse_beta","rmse_alpha","n_obs","na_perc"};

//     DMatrix<double> results_mono = DMatrix<double>::Zero( n_sim*n_obs.rows(), header.size());
//     DMatrix<double> results_rich = DMatrix<double>::Zero( n_sim*n_obs.rows(), header.size());
//     DMatrix<double> results_gmres = DMatrix<double>::Zero( n_sim*n_obs.rows(), header.size());

//     for(std::size_t n = 0; n < n_obs.rows(); ++n){ 
//         output_dir = name_dir + "output/"; // + "monolithic/";
//         output_dir += std::to_string(n_obs(n,0)) + "/" ;
        
//         std::string data_dir = input_dir + std::to_string(n_obs(n,0)) + "/";
//         if(!std::filesystem::exists(std::filesystem::path(output_dir))) std::filesystem::create_directory(output_dir);
        
//     for(std::size_t sim = 0; sim < n_sim; ++sim){

//         std::string simul_dir = data_dir + std::to_string(sim) + "/"; 
//         std::string result_dir = output_dir + std::to_string(sim) + "/";
//         if(!std::filesystem::exists(std::filesystem::path(result_dir))) std::filesystem::create_directory(result_dir);

//         std::string output_monolithic = result_dir + "monolithic/";
//         std::string output_richardson = result_dir + "richardson/";
//         if(!std::filesystem::exists(std::filesystem::path(output_monolithic))) std::filesystem::create_directory(output_monolithic);
//         if(!std::filesystem::exists(std::filesystem::path(output_richardson))) std::filesystem::create_directory(output_richardson);

//         std::vector<BlockFrame<double, int>> data;
//         data.resize(m);
        
//         for(std::size_t j = 0; j<m; j++){
//             std::string Wname = simul_dir + "W_" + std::to_string(j) + ".mtx";
//             std::string Vname = simul_dir + "V_" + std::to_string(j) + ".mtx";
//             std::string locsname = simul_dir + "locs_" + std::to_string(j) + ".mtx";
//             std::string yname = simul_dir +  "obs_" + std::to_string(j) + ".mtx";
//             auto W = read_mtx<double>(Wname);
//             auto V = read_mtx<double>(Vname);
//             auto locs = read_mtx<double>(locsname);
//             auto obs = read_mtx<double>(yname);

//             // --- NaN handling ---
//             // before passing data to the model, we need to:
//             // - remove NaN from y 
//             // - remove the corresponding rows in W_BLOCK,V_BLOCK,LOCS_BLOCK
//             // - update N (number of observations)

//             // save position of non-na values
//             std::vector<int> validIndices;
//             for (int i = 0; i < obs.size(); ++i) {
//                 if (!std::isnan(obs(i,0))) {
//                     validIndices.push_back(i);
//                 }
//             }
//             // std::cout << "valid_indices: " << validIndices.size() << std::endl;
//             // std::cout << "obs_rows: " << obs.rows() << std::endl;
            
//             if(validIndices.size() != obs.rows()){
//                 // create new y_ with only valid entries
//                 DVector<double> y_new(validIndices.size());
//                 for (size_t i = 0; i < validIndices.size(); ++i) {
//                     y_new(i) = obs(validIndices[i]);
//                 }
//                 obs = y_new; 

//                 // delete rows from W,V,locs
//                 DMatrix<double> W_new(validIndices.size(), W.cols());
//                 DMatrix<double> V_new(validIndices.size(), V.cols());
//                 DMatrix<double> locs_new(validIndices.size(), locs.cols());
//                 for (size_t i = 0; i < validIndices.size(); ++i) {
//                     W_new.row(i) = W.row(validIndices[i]);
//                     V_new.row(i) = V.row(validIndices[i]);
//                     locs_new.row(i) = locs.row(validIndices[i]);
//                 }
//                 W = W_new;
//                 V = V_new;
//                 locs = locs_new;
//                 same_locs = 0;
//             }
            
//             data[j].insert(W_BLOCK, W);
//             data[j].insert(V_BLOCK, V);
//             data[j].insert(Y_BLOCK, obs);
//             data[j].insert(LOCS_BLOCK, locs);      
//         }
    
//         DMatrix<double> f_ = DMatrix<double>::Zero(m*domain.mesh.nodes().rows(),1);
//         for(std::size_t j = 0; j < m; ++j){
//             f_.block(j*domain.mesh.nodes().rows(),0, domain.mesh.nodes().rows(),1) =   f(domain.mesh.nodes(),j);
//         }
    
//         // define regularizing PDE
//         auto L = -laplacian<FEM>();
//         DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.n_elements()*3, 1);
//         PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> problem(domain.mesh, L, u);

//         // monolithic 
//         MixedSRPDE<monolithic> monolithic_(problem, Sampling::pointwise, same_locs);
//         monolithic_.set_lambda_D(lambda);
// 	    monolithic_.set_data(data);
        
//         auto start = std::chrono::high_resolution_clock::now();
//         monolithic_.init();
//         std::chrono::duration<double> duration = std::chrono::high_resolution_clock::now() - start;
//         results_mono(sim + n_sim*n, 0) = duration.count();

//         start = std::chrono::high_resolution_clock::now();
//         monolithic_.solve();
//         duration = std::chrono::high_resolution_clock::now() - start;
//         results_mono(sim + n_sim*n, 1) = duration.count();
//         results_mono(sim + n_sim*n, 2) = results_mono(sim + n_sim*n, 0) + results_mono(sim + n_sim*n, 1);

//         // iterative
//         MixedSRPDE<iterative> richardson_(problem, Sampling::pointwise, same_locs);
//         richardson_.set_lambda_D(lambda);
// 	    richardson_.set_data(data);
//         // richardson_.set_GMRES_params(0);

//         start = std::chrono::high_resolution_clock::now();
//         richardson_.init();
//         duration = std::chrono::high_resolution_clock::now() - start;
//         results_rich(sim + n_sim*n, 0) = duration.count();

//         start = std::chrono::high_resolution_clock::now();
//         richardson_.solve();
//         duration = std::chrono::high_resolution_clock::now() - start;
//         results_rich(sim + n_sim*n, 1) = duration.count();
//         results_rich(sim + n_sim*n, 2) = results_rich(sim + n_sim*n, 0) + results_rich(sim + n_sim*n, 1);
        
//         // iterative GMRES
//         MixedSRPDE<iterative> gmres_(problem, Sampling::pointwise, same_locs);
//         gmres_.set_lambda_D(lambda);
// 	    gmres_.set_data(data);
//         gmres_.set_GMRES_params(memory);

//         start = std::chrono::high_resolution_clock::now();
//         gmres_.init();
//         duration = std::chrono::high_resolution_clock::now() - start;
//         results_gmres(sim + n_sim*n, 0) = duration.count();

//         start = std::chrono::high_resolution_clock::now();
//         gmres_.solve();
//         duration = std::chrono::high_resolution_clock::now() - start;
//         results_gmres(sim + n_sim*n, 1) = duration.count();
//         results_gmres(sim + n_sim*n, 2) = results_gmres(sim + n_sim*n, 0) + results_gmres(sim + n_sim*n, 1);
        

//         // RMSEs
//         results_mono(sim + n_sim*n, 3) = (monolithic_.f() - f_).array().square().mean();
//         results_rich(sim + n_sim*n, 3) = (richardson_.f() - f_).array().square().mean();
//         results_gmres(sim + n_sim*n, 3) = (gmres_.f() - f_).array().square().mean();
        

//         for(std::size_t j = 0; j < m; ++j){
//             Eigen::saveMarket(monolithic_.f().block(j*domain.mesh.nodes().rows(),0, domain.mesh.nodes().rows(),1),
//                           output_monolithic + "estimate_f_" + std::to_string(j) + ".mtx");
//             eigen2txt<double>(monolithic_.f().block(j*domain.mesh.nodes().rows(),0, domain.mesh.nodes().rows(),1),
//                           output_monolithic + "estimate_f_" + std::to_string(j) + ".txt");

//             results_mono(sim + n_sim*n, 4+j) = (monolithic_.f().block(j*domain.mesh.nodes().rows(),0, domain.mesh.nodes().rows(),1) -
//                                                 f_.block(j*domain.mesh.nodes().rows(),0, domain.mesh.nodes().rows(),1)).array().square().mean();

//             Eigen::saveMarket(richardson_.f().block(j*domain.mesh.nodes().rows(),0, domain.mesh.nodes().rows(),1),
//                           output_richardson + "estimate_f_" + std::to_string(j) + ".mtx");
//             eigen2txt<double>(richardson_.f().block(j*domain.mesh.nodes().rows(),0, domain.mesh.nodes().rows(),1),
//                           output_richardson + "estimate_f_" + std::to_string(j) + ".txt");

//             results_rich(sim + n_sim*n, 4+j) = (richardson_.f().block(j*domain.mesh.nodes().rows(),0, domain.mesh.nodes().rows(),1) -
//                                                 f_.block(j*domain.mesh.nodes().rows(),0, domain.mesh.nodes().rows(),1)).array().square().mean();
//             results_gmres(sim + n_sim*n, 4+j) = (gmres_.f().block(j*domain.mesh.nodes().rows(),0, domain.mesh.nodes().rows(),1) -
//                                                 f_.block(j*domain.mesh.nodes().rows(),0, domain.mesh.nodes().rows(),1)).array().square().mean();
//         }

//         Eigen::saveMarket(monolithic_.f(), output_monolithic + "estimate_f.mtx");
//         eigen2txt<double>(monolithic_.f(), output_monolithic + "estimate_f.txt");
//         Eigen::saveMarket(richardson_.f(), output_richardson + "estimate_f.mtx");
//         eigen2txt<double>(richardson_.f(), output_richardson + "estimate_f.txt");
    
//         Eigen::saveMarket(monolithic_.beta(), output_monolithic + "beta.mtx");
//         eigen2txt<double>(monolithic_.beta(), output_monolithic + "beta.txt");
//         Eigen::saveMarket(richardson_.beta(), output_richardson + "beta.mtx");
//         eigen2txt<double>(richardson_.beta(), output_richardson + "beta.txt");
    
//         Eigen::saveMarket(monolithic_.alpha(), output_monolithic + "beta.mtx");
//         eigen2txt<double>(monolithic_.alpha(), output_monolithic + "alpha.txt");
//         Eigen::saveMarket(richardson_.alpha(), output_richardson + "beta.mtx");
//         eigen2txt<double>(richardson_.alpha(), output_richardson + "alpha.txt");

//         results_mono(sim + n_sim*n, 7) =  (monolithic_.beta() - beta).array().square().mean();
//         results_mono(sim + n_sim*n, 8) =  (monolithic_.alpha() - alpha).array().square().mean();
        
//         results_rich(sim + n_sim*n, 7) =  (richardson_.beta() - beta).array().square().mean();
//         results_rich(sim + n_sim*n, 8) =  (richardson_.alpha() - alpha).array().square().mean();
        
//         results_gmres(sim + n_sim*n, 7) =  (gmres_.beta() - beta).array().square().mean();
//         results_gmres(sim + n_sim*n, 8) =  (gmres_.alpha() - alpha).array().square().mean();

//         results_mono(sim + n_sim*n,9) = n_obs(n, 0);
//         results_rich(sim + n_sim*n,9) = n_obs(n, 0);
//         results_gmres(sim + n_sim*n,9) = n_obs(n, 0);

//         double na = na_percentage_vec(static_cast<int>(sim/20));

//         results_mono(sim + n_sim*n,10) = na;
//         results_rich(sim + n_sim*n,10) = na;
//         results_gmres(sim + n_sim*n,10) = na;

//         EXPECT_TRUE(  (monolithic_.beta() - beta).array().square().mean() < 1e-2 );
//         EXPECT_TRUE(  (monolithic_.alpha() - alpha).array().square().mean() < 1e-2 );

//         EXPECT_TRUE(  (richardson_.beta() - beta).array().square().mean() < 1e-2 );
//         EXPECT_TRUE(  (richardson_.alpha() - alpha).array().square().mean() < 1e-2 );
//     }
//     }

//     write_table(results_mono, header, name_dir + "output/" + solution_policy[0] + ".txt");
//     write_table(results_rich, header, name_dir + "output/" + solution_policy[1] + ".txt");
//     write_table(results_gmres, header, name_dir + "output/" + solution_policy[1] + "_gmres.txt");
// }

TEST(mixed_srpde_test, same_locs_diff_m) {

    std::string test_name = "same_locs_diff_m";
    std::size_t n_sim = 100;

    DMatrix<double> beta = DMatrix<double>::Zero(2,1);
    beta(0,0) = -2.; beta(1,0) = 1.;

    bool same_locs = 1;
    int memory = 3;  
    double lambda = 1e-3;

    DMatrix<int> m = DVector<int>::Zero(8);
    m(0) = 2;
    m(1) = 3; 
    m(2) = 4; 
    m(3) = 5; 
    m(4) = 10;
    m(5) = 15;
    m(6) = 20;
    m(7) = 25;

    DMatrix<int> n_obs = DMatrix<int>::Zero(1,1);
    n_obs(0,0) = 2000;

    int seed = 473865; 

    //std::string meshID = "unit_square";
    std::string meshID = "unit_square_coarse";
    MeshLoader<Mesh2D> domain(meshID);
    meshID = meshID + "/"; 
    std::string name_dir = "../data/models/mixed_srpde/" + meshID;
	if(!std::filesystem::create_directory(name_dir)) std::filesystem::create_directory(name_dir);
    
    name_dir += test_name;
    if(!std::filesystem::create_directory(name_dir)) std::filesystem::create_directory(name_dir);
	
    // input
    std::string input_dir = name_dir  + "input/";

    for(size_t i; i < m.rows(); ++i){

        std::cout << "\t --- i = " << i << std::endl;

        DMatrix<double> alpha = generateAlpha(m(i), seed);

        if(!std::filesystem::exists(std::filesystem::path(input_dir))) {
            std::filesystem::create_directory(input_dir);
        }

        std::cout << "\t --- generating data --- " << std::endl;
        Eigen::saveMarket(beta, input_dir + "beta.mtx");
        Eigen::saveMarket(alpha, input_dir + "alpha.mtx");
        Eigen::saveMarket(n_obs, input_dir + "n_obs.mtx");
    
        eigen2txt<double>(beta, input_dir + "beta.txt");
        eigen2txt<double>(alpha, input_dir + "alpha.txt");
        eigen2txt<int>(n_obs, input_dir + "n_obs.txt");

        Eigen::saveMarket(x1_(domain.mesh.nodes()), input_dir + "cov_1.mtx");
        eigen2txt<double>(x1_(domain.mesh.nodes()), input_dir + "cov_1.txt");

        for( std::size_t j=0; j < m(i); ++j){
            DMatrix<double> f_ = f(domain.mesh.nodes(), j);
            Eigen::saveMarket(f_, input_dir + "f_" + std::to_string(j) + ".mtx");
            eigen2txt<double>(f_, input_dir + "f_" + std::to_string(j) + ".txt");
        }

        for(std::size_t n = 0; n < n_obs.rows(); ++n){
            
            // generete data
            std::string data_dir = input_dir + std::to_string(n_obs(n)) + "/";
            std::filesystem::create_directory(data_dir);
                
            for(std::size_t sim=0; sim<n_sim; ++sim){
                std::mt19937 gen(seed+ sim); 
                std::string simul_dir = data_dir + std::to_string(sim) + "/"; 
                std::filesystem::create_directory(simul_dir);

                DMatrix<double> locs = uniform_locs(n_obs(n), gen);
                
                for(std::size_t j = 0; j < m(i); ++j){
                    DMatrix<double> DesignMatrix = DMatrix<double>::Zero(n_obs(n),2);
                
                    DesignMatrix.col(0) = x1_(locs); // va in V
                    DesignMatrix.col(1) = noise(n_obs(n), 1.0, gen);

                    DMatrix<double> f_ = f(locs, j);
                    double sigma = 0.05*std::abs(f_.array().maxCoeff() - f_.array().minCoeff()); 
                    auto eps_ = noise(n_obs(n), sigma, gen);
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
        
            // Output directory
            std::string output_dir = name_dir + "output/";
            if(!std::filesystem::exists(std::filesystem::path(output_dir))){ 
                std::filesystem::create_directory(output_dir);
            }
        
            std::vector<std::string> solution_policy = {"monolithic", "richardson"};

            // import data from files
            std::vector<std::string> header = {"time_init", "time_solve", "time", "rmse_f", "rmse_beta", "rmse_alpha", "n_obs", "m"};
            
            for (int j = 1; j <= m.maxCoeff(); ++j) {
                header.push_back("rmse_f_" + std::to_string(j));
            }

            DMatrix<double> results_mono = DMatrix<double>::Zero( n_sim*n_obs.size(), header.size());
            DMatrix<double> results_rich = DMatrix<double>::Zero( n_sim*n_obs.size(), header.size());
            DMatrix<double> results_gmres = DMatrix<double>::Zero( n_sim*n_obs.rows(), header.size());

        for(std::size_t n = 0; n < n_obs.rows(); ++n){ 
            output_dir = name_dir + "output/"; // + "monolithic/";
            output_dir += std::to_string(n_obs(n,0)) + "/" ;
            
            std::string data_dir = input_dir + std::to_string(n_obs(n,0)) + "/";
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
                data.resize(m(i));
                
                for(std::size_t j = 0; j<m(i); j++){
                    std::cout << "---- j = " << j << std::endl;
                    std::string Wname = simul_dir + "W_" + std::to_string(j) + ".mtx";
                    std::string Vname = simul_dir + "V_" + std::to_string(j) + ".mtx";
                    std::string locsname = simul_dir + "locs_" + std::to_string(j) + ".mtx";
                    std::string yname = simul_dir +  "obs_" + std::to_string(j) + ".mtx";
                    auto W = read_mtx<double>(Wname);
                    auto V = read_mtx<double>(Vname);
                    auto locs = read_mtx<double>(locsname);
                    auto obs = read_mtx<double>(yname);

                    // save position of non-na values
                    std::vector<int> validIndices;
                    for (int i = 0; i < obs.size(); ++i) {
                        if (!std::isnan(obs(i,0))) {
                            validIndices.push_back(i);
                        }
                    }
                    // std::cout << "valid_indices: " << validIndices.size() << std::endl;
                    // std::cout << "obs_rows: " << obs.rows() << std::endl;
                    
                    if(validIndices.size() != obs.rows()){
                        // create new y_ with only valid entries
                        DVector<double> y_new(validIndices.size());
                        for (size_t i = 0; i < validIndices.size(); ++i) {
                            y_new(i) = obs(validIndices[i]);
                        }
                        obs = y_new; 

                        // delete rows from W,V,locs
                        DMatrix<double> W_new(validIndices.size(), W.cols());
                        DMatrix<double> V_new(validIndices.size(), V.cols());
                        DMatrix<double> locs_new(validIndices.size(), locs.cols());
                        for (size_t i = 0; i < validIndices.size(); ++i) {
                            W_new.row(i) = W.row(validIndices[i]);
                            V_new.row(i) = V.row(validIndices[i]);
                            locs_new.row(i) = locs.row(validIndices[i]);
                        }
                        W = W_new;
                        V = V_new;
                        locs = locs_new;
                        same_locs = 0;
                    }

                    data[j].insert(W_BLOCK, W);
                    data[j].insert(V_BLOCK, V);
                    data[j].insert(Y_BLOCK, obs);
                    data[j].insert(LOCS_BLOCK, locs);      
                }
            
                DMatrix<double> f_ = DMatrix<double>::Zero(m(i)*domain.mesh.nodes().rows(),1);
                
                for(std::size_t j = 0; j < m(i); ++j){
                    f_.block(j*domain.mesh.nodes().rows(),0, domain.mesh.nodes().rows(),1) =   f(domain.mesh.nodes(),j);
                }
            
                // define regularizing PDE
                auto L = -laplacian<FEM>();
                DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.n_elements()*3, 1);
                PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> problem(domain.mesh, L, u);

                // monolithic 
                MixedSRPDE<monolithic> monolithic_(problem, Sampling::pointwise, same_locs);
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
                MixedSRPDE<iterative> richardson_(problem, Sampling::pointwise, same_locs);
                richardson_.set_lambda_D(lambda);
                richardson_.set_data(data);
                // richardson_.set_GMRES_params(memory);

                start = std::chrono::high_resolution_clock::now();
                richardson_.init();
                duration = std::chrono::high_resolution_clock::now() - start;
                results_rich(sim + n_sim*n, 0) = duration.count();

                start = std::chrono::high_resolution_clock::now();
                richardson_.solve();
                duration = std::chrono::high_resolution_clock::now() - start;
                results_rich(sim + n_sim*n, 1) = duration.count();
                results_rich(sim + n_sim*n, 2) = results_rich(sim + n_sim*n, 0) + results_rich(sim + n_sim*n, 1);
                
                // iterative GMRES
                MixedSRPDE<iterative> gmres_(problem, Sampling::pointwise, same_locs);
                gmres_.set_lambda_D(lambda);
                gmres_.set_data(data);
                gmres_.set_GMRES_params(memory);

                start = std::chrono::high_resolution_clock::now();
                gmres_.init();
                duration = std::chrono::high_resolution_clock::now() - start;
                results_gmres(sim + n_sim*n, 0) = duration.count();

                start = std::chrono::high_resolution_clock::now();
                gmres_.solve();
                duration = std::chrono::high_resolution_clock::now() - start;
                results_gmres(sim + n_sim*n, 1) = duration.count();
                results_gmres(sim + n_sim*n, 2) = results_gmres(sim + n_sim*n, 0) + results_gmres(sim + n_sim*n, 1);

                // RMSEs
                results_mono(sim + n_sim*n, 3) = (monolithic_.f() - f_).array().square().mean();
                results_rich(sim + n_sim*n, 3) = (richardson_.f() - f_).array().square().mean();
                results_gmres(sim + n_sim*n, 3) = (gmres_.f() - f_).array().square().mean();

                int index = 4;

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

                results_mono(sim + n_sim*n, index) =  (monolithic_.beta() - beta).array().square().mean();
                results_mono(sim + n_sim*n, index+1) =  (monolithic_.alpha() - alpha).array().square().mean();
                
                results_rich(sim + n_sim*n, index) =  (richardson_.beta() - beta).array().square().mean();
                results_rich(sim + n_sim*n, index+1) =  (richardson_.alpha() - alpha).array().square().mean();
                
                results_gmres(sim + n_sim*n, index) =  (gmres_.beta() - beta).array().square().mean();
                results_gmres(sim + n_sim*n, index+1) =  (gmres_.alpha() - alpha).array().square().mean();
                
                results_mono(sim + n_sim*n,index+2) = n_obs(n);
                results_rich(sim + n_sim*n,index+2) = n_obs(n);
                results_gmres(sim + n_sim*n,index+2) = n_obs(n);

                results_mono(sim + n_sim*n,index+3) = m(i);
                results_rich(sim + n_sim*n,index+3) = m(i);
                results_gmres(sim + n_sim*n,index+3) = m(i);

                // rmse_f_i
                for(std::size_t j = 0; j < m(i); ++j){
                    Eigen::saveMarket(monolithic_.f().block(j*domain.mesh.nodes().rows(),0, domain.mesh.nodes().rows(),1),
                                output_monolithic + "estimate_f_" + std::to_string(j) + ".mtx");
                    eigen2txt<double>(monolithic_.f().block(j*domain.mesh.nodes().rows(),0, domain.mesh.nodes().rows(),1),
                                output_monolithic + "estimate_f_" + std::to_string(j) + ".txt");

                    results_mono(sim + n_sim*n, index+4+j) = (monolithic_.f().block(j*domain.mesh.nodes().rows(),0, domain.mesh.nodes().rows(),1) -
                                                        f_.block(j*domain.mesh.nodes().rows(),0, domain.mesh.nodes().rows(),1)).array().square().mean();

                    Eigen::saveMarket(richardson_.f().block(j*domain.mesh.nodes().rows(),0, domain.mesh.nodes().rows(),1),
                                output_richardson + "estimate_f_" + std::to_string(j) + ".mtx");
                    eigen2txt<double>(richardson_.f().block(j*domain.mesh.nodes().rows(),0, domain.mesh.nodes().rows(),1),
                                output_richardson + "estimate_f_" + std::to_string(j) + ".txt");

                    results_rich(sim + n_sim*n, index+4+j) = (richardson_.f().block(j*domain.mesh.nodes().rows(),0, domain.mesh.nodes().rows(),1) -
                                                        f_.block(j*domain.mesh.nodes().rows(),0, domain.mesh.nodes().rows(),1)).array().square().mean();
                
                    results_gmres(sim + n_sim*n, index+4+j) = (gmres_.f().block(j*domain.mesh.nodes().rows(),0, domain.mesh.nodes().rows(),1) -
                                                        f_.block(j*domain.mesh.nodes().rows(),0, domain.mesh.nodes().rows(),1)).array().square().mean();                
                
                }


                EXPECT_TRUE(  (monolithic_.beta() - beta).array().square().mean() < 1e-2 );
                EXPECT_TRUE(  (monolithic_.alpha() - alpha).array().square().mean() < 1e-2 );

                EXPECT_TRUE(  (richardson_.beta() - beta).array().square().mean() < 1e-2 );
                EXPECT_TRUE(  (richardson_.alpha() - alpha).array().square().mean() < 1e-2 );
            }
        }

        
        std::string fileName_mono_general = name_dir + "output/" + solution_policy[0] + "_gen.txt";
        std::string fileName_iter_general = name_dir + "output/" + solution_policy[1] + "_gen.txt";
        std::string fileName_gmres_general = name_dir + "output/" + solution_policy[1] + "_gmres_gen.txt";

        if(i){   
            write_table_noHeaders(results_mono, name_dir + "output/" + solution_policy[0] + ".txt");
            write_table_noHeaders(results_rich, name_dir + "output/" + solution_policy[1] + ".txt");
            write_table_noHeaders(results_gmres, name_dir + "output/" + solution_policy[1] + "_gmres.txt");

            std::string fileName_mono = name_dir + "output/" + solution_policy[0] + ".txt";
            std::string fileName_iter = name_dir + "output/" + solution_policy[1] + ".txt";
            std::string fileName_gmres = name_dir + "output/" + solution_policy[1] + "_gmres.txt";

            appendFileContent(fileName_mono, fileName_mono_general);
            appendFileContent(fileName_iter, fileName_iter_general);
            appendFileContent(fileName_gmres, fileName_gmres_general);
        } else {
            write_table(results_mono, header, name_dir + "output/" + solution_policy[0] + "_gen.txt");
            write_table(results_rich, header, name_dir + "output/" + solution_policy[1] + "_gen.txt");
            write_table(results_gmres, header, name_dir + "output/" + solution_policy[1] + "_gmres_gen.txt");
        }
    }
}