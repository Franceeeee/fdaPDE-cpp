#include <iostream>
#include <vector>
#include <string>
#include <random>
#include <cmath>

#include <cstddef>
// #include <gtest/gtest.h>   // testing framework
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

#include "../../fdaPDE/models/regression/fanova.h"
#include "../../fdaPDE/models/sampling_design.h"
using fdapde::models::fANOVA;
using fdapde::models::Sampling;
using fdapde::models::SpaceOnly;
using fdapde::monolithic;
using fdapde::iterative;

#include "../test/src/utils/constants.h"
#include "../test/src/utils/mesh_loader.h"
#include "../test/src/utils/utils.h"
#include<filesystem>
using fdapde::testing::almost_equal;
using fdapde::testing::MeshLoader;
using fdapde::testing::read_csv;




template <typename T> Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> 
read_mtx(const std::string& file_name) {
    Eigen::SparseMatrix<T> buff;
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

// NA mask
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

    // normalize 
    for (int i = 0; i < m; ++i) {
        alpha(i, 0) /= sum;
    }

    return alpha;
}

void appendFileContent(const std::string& sourceFile, const std::string& targetFile) {
    std::ifstream source(sourceFile); 
    std::ofstream target(targetFile, std::ios::app); 

    if (!source.is_open() || !target.is_open()) {
        std::cerr << "Error: Unable to open file(s)." << std::endl;
        return;
    }

    std::string line;
    while (std::getline(source, line)) {
        target << line << '\n';  
    }

    source.close();
    target.close();
}

int main(){

    // TEST 1: mono&iter 2000 obs - unit square coarse
    // -- test parameters --
	std::string test_name = "small_obs_mesh_coarse/";
	int seed = 23872; 
    double lambda = 1e-3; 
    bool same_locs = 0;
    std::string meshID = "unit_square_coarse";
    std::size_t m = 15;
	double na_percentage = 0.0;
	DMatrix<double> beta = DMatrix<double>::Zero(2,1);
    beta(0,0) = -2.; beta(1,0) = 1.;
    DMatrix<double> alpha = DMatrix<double>::Zero(3,1);
    alpha(0,0) = -0.5; alpha(1,0) = 0.; alpha(2,0) = 0.5;
    DMatrix<int> n_obs = DMatrix<int>::Zero(1,m);
    n_obs(0,0) = 2000;

    // ---
	MeshLoader<Mesh2D> domain(meshID);
    meshID = meshID + "/"; 

    std::string name_dir = "../data/models/fanova/";
	if(!std::filesystem::create_directory(name_dir)) std::filesystem::create_directory(name_dir);

    name_dir += meshID;
	if(!std::filesystem::create_directory(name_dir)) std::filesystem::create_directory(name_dir);

    name_dir += test_name;
    if(!std::filesystem::exists(std::filesystem::path(name_dir))) std::filesystem::create_directory(name_dir);
	
	// input data 
    std::string input_dir = name_dir  + "input/";

    if(!std::filesystem::exists(std::filesystem::path(input_dir))) {

        std::filesystem::create_directory(input_dir);
        std::vector<double> means = {500, 1000, 2000, 4000, 8000};
        std::vector<double> stddevs = {50, 100, 200, 400, 800}; 

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
            
        // generate data
        std::string data_dir = input_dir + std::to_string(n_obs) + "/";
        std::filesystem::create_directory(data_dir);
            
        std::mt19937 gen(seed);
        
        std::string simul_dir = data_dir + "/"; 
        std::filesystem::create_directory(simul_dir);

        for(std::size_t j = 0; j < m; ++j){
            DMatrix<double> DesignMatrix = DMatrix<double>::Zero(n_obs,2);
        
            DesignMatrix.col(0) = x1_(locs); // va in V
            DesignMatrix.col(1) = noise(n_obs, 1.0, gen);

            DMatrix<double> f_ = f(locs, j);
            double sigma = 0.05*std::abs(f_.array().maxCoeff() - f_.array().minCoeff()); 
            auto eps_ = noise(n_obs, sigma, gen);
            eigen2txt<double>(eps_, simul_dir + "noise_" + std::to_string(j) + ".txt");
        
            DMatrix<double> obs = DesignMatrix * beta + DesignMatrix.col(0)*alpha(j,0)  + f_ + eps_; 
            
            auto na_mask = create_na_mask(n_obs, na_percentage, gen); 
            for (int i = 0; i < n_obs; ++i) { 
                if (na_mask[i]) {
                    obs(i, 0) = std::numeric_limits<double>::quiet_NaN();  
                }
            }
        
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

	// output directory
	std::string output_dir = name_dir + "output/";
    if(!std::filesystem::exists(std::filesystem::path(output_dir))){ 
        std::filesystem::create_directory(output_dir);
    }
	
    std::vector<std::string> solution_policy = {"monolithic", "richardson"};

    // import data from files
    std::vector<std::string> header = {"time_init", "time_solve", "time",
                                       "rmse_f","rmse_f_1", "rmse_f_2","rmse_f_3", 
                                       "rmse_beta","rmse_alpha","n_obs","na_perc", "meshID"};

    int n = n_obs.rows();

    DMatrix<double> results_mono = DMatrix<double>::Zero(n, header.size());
    DMatrix<double> results_rich = DMatrix<double>::Zero(n, header.size());
    DMatrix<double> results_gmres = DMatrix<double>::Zero(n, header.size());

    output_dir = name_dir + "output/"; 
    output_dir += std::to_string(n_obs) + "/" ;
    
    std::string data_dir = input_dir + std::to_string(n_obs) + "/";
    if(!std::filesystem::exists(std::filesystem::path(output_dir))) std::filesystem::create_directory(output_dir);

    std::string simul_dir = data_dir + "/"; 
    std::string result_dir = output_dir + "/";
    if(!std::filesystem::exists(std::filesystem::path(result_dir))) std::filesystem::create_directory(result_dir);

    std::string output_monolithic = result_dir + "monolithic/";
    std::string output_richardson = result_dir + "richardson/";
    if(!std::filesystem::exists(std::filesystem::path(output_monolithic))) std::filesystem::create_directory(output_monolithic);
    if(!std::filesystem::exists(std::filesystem::path(output_richardson))) std::filesystem::create_directory(output_richardson);

    std::vector<BlockFrame<double, int>> data;
    data.resize(m);
    
    // insert data in blockframes
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

    // monolithic 
    fANOVA<monolithic> monolithic_(problem, Sampling::pointwise, same_locs);
    monolithic_.set_lambda_D(lambda);
    monolithic_.set_data(data);
    
    auto start = std::chrono::high_resolution_clock::now();
    monolithic_.init();
    std::chrono::duration<double> duration = std::chrono::high_resolution_clock::now() - start;
    results_mono(n, 0) = duration.count();

    start = std::chrono::high_resolution_clock::now();
    monolithic_.solve();
    duration = std::chrono::high_resolution_clock::now() - start;
    results_mono(n, 1) = duration.count();
    results_mono(n, 2) = results_mono(n, 0) + results_mono(n, 1);

    // iterative
    fANOVA<iterative> richardson_(problem, Sampling::pointwise, same_locs);
    richardson_.set_lambda_D(lambda);
    richardson_.set_data(data);

    start = std::chrono::high_resolution_clock::now();
    richardson_.init();
    duration = std::chrono::high_resolution_clock::now() - start;
    results_rich(n, 0) = duration.count();

    start = std::chrono::high_resolution_clock::now();
    richardson_.solve();
    duration = std::chrono::high_resolution_clock::now() - start;
    results_rich(n, 1) = duration.count();
    results_rich(n, 2) = results_rich(n, 0) + results_rich(n, 1);

    // RMSEs
    results_mono(n, 3) = (monolithic_.f() - f_).array().square().mean();
    results_rich(n, 3) = (richardson_.f() - f_).array().square().mean();
    
    for(std::size_t j = 0; j < m; ++j){
        Eigen::saveMarket(monolithic_.f().block(j*domain.mesh.nodes().rows(),0, domain.mesh.nodes().rows(),1),
                        output_monolithic + "estimate_f_" + std::to_string(j) + ".mtx");
        eigen2txt<double>(monolithic_.f().block(j*domain.mesh.nodes().rows(),0, domain.mesh.nodes().rows(),1),
                        output_monolithic + "estimate_f_" + std::to_string(j) + ".txt");

        results_mono(n, 4+j) = (monolithic_.f().block(j*domain.mesh.nodes().rows(),0, domain.mesh.nodes().rows(),1) -
                                            f_.block(j*domain.mesh.nodes().rows(),0, domain.mesh.nodes().rows(),1)).array().square().mean();

        Eigen::saveMarket(richardson_.f().block(j*domain.mesh.nodes().rows(),0, domain.mesh.nodes().rows(),1),
                        output_richardson + "estimate_f_" + std::to_string(j) + ".mtx");
        eigen2txt<double>(richardson_.f().block(j*domain.mesh.nodes().rows(),0, domain.mesh.nodes().rows(),1),
                        output_richardson + "estimate_f_" + std::to_string(j) + ".txt");

        results_rich(n, 4+j) = (richardson_.f().block(j*domain.mesh.nodes().rows(),0, domain.mesh.nodes().rows(),1) -
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

    results_mono(n, 7) = (monolithic_.beta() - beta).array().square().mean();
    results_mono(n, 8) = (monolithic_.alpha() - alpha).array().square().mean();
    
    results_rich(n, 7) = (richardson_.beta() - beta).array().square().mean();
    results_rich(n, 8) = (richardson_.alpha() - alpha).array().square().mean();

    results_mono(n,9) = n_obs;
    results_rich(n,9) = n_obs;

    results_mono(n,10) = na_percentage;
    results_rich(n,10) = na_percentage;

    results_mono(n,11) = meshID;
    results_rich(n,11) = meshID;

    EXPECT_TRUE(  (monolithic_.beta() - beta).array().square().mean() < 1e-2 );
    EXPECT_TRUE(  (monolithic_.alpha() - alpha).array().square().mean() < 1e-2 );

    EXPECT_TRUE(  (richardson_.beta() - beta).array().square().mean() < 1e-2 );
    EXPECT_TRUE(  (richardson_.alpha() - alpha).array().square().mean() < 1e-2 );
    
    write_table(results_mono, header, name_dir + "output/" + solution_policy[0] + ".txt");
    write_table(results_rich, header, name_dir + "output/" + solution_policy[1] + ".txt");
	

    // TEST 2: mono&iter 20000 obs - unit square
    // -- test parameters --
    std::string test_name = "large_obs_mesh_fine/";
	int seed = 23872; 
    double lambda = 1e-3; 
    bool same_locs = 0;
    std::string meshID = "unit_square";
    std::size_t m = 15;
	double na_percentage = 0.0;
	DMatrix<double> beta = DMatrix<double>::Zero(2,1);
    beta(0,0) = -2.; beta(1,0) = 1.;
    DMatrix<double> alpha = DMatrix<double>::Zero(3,1);
    alpha(0,0) = -0.5; alpha(1,0) = 0.; alpha(2,0) = 0.5;
    DMatrix<int> n_obs = DMatrix<int>::Zero(1,m);
    n_obs(0,0) = 20000;

	// ---
	MeshLoader<Mesh2D> domain(meshID);
    meshID = meshID + "/"; 

    std::string name_dir = "../data/models/fanova/";
	if(!std::filesystem::create_directory(name_dir)) std::filesystem::create_directory(name_dir);

    name_dir += meshID;
	if(!std::filesystem::create_directory(name_dir)) std::filesystem::create_directory(name_dir);

    name_dir += test_name;
    if(!std::filesystem::exists(std::filesystem::path(name_dir))) std::filesystem::create_directory(name_dir);
	
	// input data 
    std::string input_dir = name_dir  + "input/";

    if(!std::filesystem::exists(std::filesystem::path(input_dir))) {

        std::filesystem::create_directory(input_dir);
        std::vector<double> means = {500, 1000, 2000, 4000, 8000};
        std::vector<double> stddevs = {50, 100, 200, 400, 800}; 

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
            
        // generate data
        std::string data_dir = input_dir + std::to_string(n_obs) + "/";
        std::filesystem::create_directory(data_dir);
            
        std::mt19937 gen(seed);
        
        std::string simul_dir = data_dir + "/"; 
        std::filesystem::create_directory(simul_dir);

        for(std::size_t j = 0; j < m; ++j){
            DMatrix<double> DesignMatrix = DMatrix<double>::Zero(n_obs,2);
        
            DesignMatrix.col(0) = x1_(locs); // va in V
            DesignMatrix.col(1) = noise(n_obs, 1.0, gen);

            DMatrix<double> f_ = f(locs, j);
            double sigma = 0.05*std::abs(f_.array().maxCoeff() - f_.array().minCoeff()); 
            auto eps_ = noise(n_obs, sigma, gen);
            eigen2txt<double>(eps_, simul_dir + "noise_" + std::to_string(j) + ".txt");
        
            DMatrix<double> obs = DesignMatrix * beta + DesignMatrix.col(0)*alpha(j,0)  + f_ + eps_; 
            
            auto na_mask = create_na_mask(n_obs, na_percentage, gen); 
            for (int i = 0; i < n_obs; ++i) { 
                if (na_mask[i]) {
                    obs(i, 0) = std::numeric_limits<double>::quiet_NaN();  
                }
            }
        
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

	// output directory
	std::string output_dir = name_dir + "output/";
    if(!std::filesystem::exists(std::filesystem::path(output_dir))){ 
        std::filesystem::create_directory(output_dir);
    }
	
    std::vector<std::string> solution_policy = {"monolithic", "richardson"};

    // import data from files
    std::vector<std::string> header = {"time_init", "time_solve", "time",
                                       "rmse_f","rmse_f_1", "rmse_f_2","rmse_f_3", 
                                       "rmse_beta","rmse_alpha","n_obs","na_perc", "meshID"};

    int n = n_obs.rows();

    DMatrix<double> results_mono = DMatrix<double>::Zero(n, header.size());
    DMatrix<double> results_rich = DMatrix<double>::Zero(n, header.size());
    DMatrix<double> results_gmres = DMatrix<double>::Zero(n, header.size());

    output_dir = name_dir + "output/"; 
    output_dir += std::to_string(n_obs) + "/" ;
    
    std::string data_dir = input_dir + std::to_string(n_obs) + "/";
    if(!std::filesystem::exists(std::filesystem::path(output_dir))) std::filesystem::create_directory(output_dir);

    std::string simul_dir = data_dir + "/"; 
    std::string result_dir = output_dir + "/";
    if(!std::filesystem::exists(std::filesystem::path(result_dir))) std::filesystem::create_directory(result_dir);

    std::string output_monolithic = result_dir + "monolithic/";
    std::string output_richardson = result_dir + "richardson/";
    if(!std::filesystem::exists(std::filesystem::path(output_monolithic))) std::filesystem::create_directory(output_monolithic);
    if(!std::filesystem::exists(std::filesystem::path(output_richardson))) std::filesystem::create_directory(output_richardson);

    std::vector<BlockFrame<double, int>> data;
    data.resize(m);
    
    // insert data in blockframes
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

    // monolithic 
    fANOVA<monolithic> monolithic_(problem, Sampling::pointwise, same_locs);
    monolithic_.set_lambda_D(lambda);
    monolithic_.set_data(data);
    
    auto start = std::chrono::high_resolution_clock::now();
    monolithic_.init();
    std::chrono::duration<double> duration = std::chrono::high_resolution_clock::now() - start;
    results_mono(n, 0) = duration.count();

    start = std::chrono::high_resolution_clock::now();
    monolithic_.solve();
    duration = std::chrono::high_resolution_clock::now() - start;
    results_mono(n, 1) = duration.count();
    results_mono(n, 2) = results_mono(n, 0) + results_mono(n, 1);

    // iterative
    fANOVA<iterative> richardson_(problem, Sampling::pointwise, same_locs);
    richardson_.set_lambda_D(lambda);
    richardson_.set_data(data);

    start = std::chrono::high_resolution_clock::now();
    richardson_.init();
    duration = std::chrono::high_resolution_clock::now() - start;
    results_rich(n, 0) = duration.count();

    start = std::chrono::high_resolution_clock::now();
    richardson_.solve();
    duration = std::chrono::high_resolution_clock::now() - start;
    results_rich(n, 1) = duration.count();
    results_rich(n, 2) = results_rich(n, 0) + results_rich(n, 1);

    // RMSEs
    results_mono(n, 3) = (monolithic_.f() - f_).array().square().mean();
    results_rich(n, 3) = (richardson_.f() - f_).array().square().mean();
    

    for(std::size_t j = 0; j < m; ++j){
        Eigen::saveMarket(monolithic_.f().block(j*domain.mesh.nodes().rows(),0, domain.mesh.nodes().rows(),1),
                        output_monolithic + "estimate_f_" + std::to_string(j) + ".mtx");
        eigen2txt<double>(monolithic_.f().block(j*domain.mesh.nodes().rows(),0, domain.mesh.nodes().rows(),1),
                        output_monolithic + "estimate_f_" + std::to_string(j) + ".txt");

        results_mono(n, 4+j) = (monolithic_.f().block(j*domain.mesh.nodes().rows(),0, domain.mesh.nodes().rows(),1) -
                                            f_.block(j*domain.mesh.nodes().rows(),0, domain.mesh.nodes().rows(),1)).array().square().mean();

        Eigen::saveMarket(richardson_.f().block(j*domain.mesh.nodes().rows(),0, domain.mesh.nodes().rows(),1),
                        output_richardson + "estimate_f_" + std::to_string(j) + ".mtx");
        eigen2txt<double>(richardson_.f().block(j*domain.mesh.nodes().rows(),0, domain.mesh.nodes().rows(),1),
                        output_richardson + "estimate_f_" + std::to_string(j) + ".txt");

        results_rich(n, 4+j) = (richardson_.f().block(j*domain.mesh.nodes().rows(),0, domain.mesh.nodes().rows(),1) -
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

    results_mono(n, 7) = (monolithic_.beta() - beta).array().square().mean();
    results_mono(n, 8) = (monolithic_.alpha() - alpha).array().square().mean();
    
    results_rich(n, 7) = (richardson_.beta() - beta).array().square().mean();
    results_rich(n, 8) = (richardson_.alpha() - alpha).array().square().mean();

    results_mono(n,9) = n_obs;
    results_rich(n,9) = n_obs;

    results_mono(n,10) = na_percentage;
    results_rich(n,10) = na_percentage;

    results_mono(n,11) = meshID;
    results_rich(n,11) = meshID;

    EXPECT_TRUE(  (monolithic_.beta() - beta).array().square().mean() < 1e-2 );
    EXPECT_TRUE(  (monolithic_.alpha() - alpha).array().square().mean() < 1e-2 );

    EXPECT_TRUE(  (richardson_.beta() - beta).array().square().mean() < 1e-2 );
    EXPECT_TRUE(  (richardson_.alpha() - alpha).array().square().mean() < 1e-2 );

    write_table(results_mono, header, name_dir + "output/" + solution_policy[0] + ".txt");
    write_table(results_rich, header, name_dir + "output/" + solution_policy[1] + ".txt");
	
	return 0;
}
