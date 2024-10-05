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

// TEST CON GENERAZIONE INTERNA DEI DATI
std::vector<double> generate_b_coefficients(int n_patients, int seed) {
    std::mt19937 gen(seed);

    if (n_patients < 1) throw std::invalid_argument("Number of patients must be at least 1");
    
    std::vector<double> b(n_patients);

    if (n_patients == 1){
        b[0] = 0; 
        return b;
    }

    // Uniform distribution between -1 and 1
    std::uniform_real_distribution<> dis(-1.0, 1.0);

    // Generate n_patients - 1 random coefficients in the range [-1, 1]
    for (int i = 0; i < n_patients - 1; ++i) {
        b[i] = dis(gen);
    }

    // Calculate the sum of the first n_patients - 1 elements
    double sum = std::accumulate(b.begin(), b.end() - 1, 0.0);

    // Set the last element to make the sum of the vector 0
    b[n_patients - 1] = -sum;

    return b;
}

auto lambda_cov = [](double x, double y) { // valutazione delle covariate
    return std::sin(2*fdapde::testing::pi*x)*std::sin(2*fdapde::testing::pi*y); };

auto lambda_np = [](double x, double y, int id = 0) { // lambda function per la parte non parametrica
	// if(id == 0)
    // 	return std::sin(fdapde::testing::pi*x)*std::sin(fdapde::testing::pi*y); 
	// else if(id == 1)
	// 	return std::sin(fdapde::testing::pi*x)*std::cos(fdapde::testing::pi*y);
	// else if (id == 2)
	// 	return 1.-x-y;
		
	// return 0.;

    std::mt19937 gen(id);  // Initialize a generator with the patient ID as the seed
    std::uniform_real_distribution<> dis(-5.0, 5.0);  // Uniform distribution

    double a_sin = dis(gen);  // Generate a0 based on id
    double a_cos = dis(gen);

    return (a_sin * std::sin(fdapde::testing::pi * x) + a_cos * std::cos(fdapde::testing::pi * y))*x*y*(1-x)*(1-y);
};
  
void write_to_csv(const std::string& filename, const std::vector<std::vector<double>>& data) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Unable to open file: " << filename << std::endl;
        return;
    }
    for (std::size_t row_index = 0; row_index < data.size(); ++row_index) {
        file << "\"" << row_index << "\"";  // Write row index with quotes
        for (std::size_t col_index = 0; col_index < data[row_index].size(); col_index++) {
            file << ",\"" << data[row_index][col_index] << "\"";  // Write each value with quotes
        }
        file << "\n";  // Newline after each row
    }
    file.close();
}

// generare punti casuali nel quadrato unitario [0, 1] x [0, 1]
std::vector<std::vector<double>> generate_random_points(int n_points, std::mt19937& gen) {
    std::uniform_real_distribution<> dis(0.0, 1.0);
    std::vector<std::vector<double>> points(n_points, std::vector<double>(2));
    for (int i = 0; i < n_points; ++i) {
        points[i][0] = dis(gen);  // x
        points[i][1] = dis(gen);  // y
    }
    return points;
}

// maschera di NA
std::vector<bool> create_na_mask(int size, double na_percentage, std::mt19937& gen) {
    std::vector<bool> mask(size, false);  
    int num_na = static_cast<int>(std::round(size * na_percentage));
    std::uniform_int_distribution<> dis(0, size - 1);
    for (int i = 0; i < num_na; ++i) {
        int index;
        do { index = dis(gen); } while (mask[index]); 
        mask[index] = true;
    }
    return mask;
}

// generare dati per un singolo paziente
void generate_data_for_patient(int patient_id, const std::vector<double>& a, const std::vector<double>& b, int n_obs, std::mt19937& gen, const std::string& output_dir, double na_percentage) {
    std::vector<std::vector<double>> locs = generate_random_points(n_obs, gen);
    std::vector<std::vector<double>> X(n_obs, std::vector<double>(a.size()));
    std::vector<std::vector<double>> W(n_obs, std::vector<double>(a.size()-1));
    std::vector<std::vector<double>> V(n_obs, std::vector<double>(a.size()-1)); 
    std::vector<std::vector<double>> observations(n_obs, std::vector<double>(1));

    // distribuzione per generare le covariate
    // std::uniform_real_distribution<> dis(0.0, 1.0);
    std::normal_distribution<> noise_dist(0.0, 2.0);
    std::normal_distribution<> noise_obs(0.0, 1.0);
    
    // matrice disegno
    for (int i=0; i<n_obs; i++){
        X[i][0] = lambda_cov(locs[i][0], locs[i][1]);
        X[i][1] = noise_dist(gen);
    }

    for (int i = 0; i < n_obs; ++i) {
        W[i][0] = X[i][1];
        V[i][0] = X[i][0];

        double f_value = lambda_np(locs[i][0], locs[i][1], patient_id);

        observations[i][0] = a[0]*X[i][0] + a[1]*X[i][1] + b[patient_id]*V[i][0] + f_value;
        observations[i][0] += noise_obs(gen);
    }

 
    std::vector<bool> na_mask = create_na_mask(n_obs, na_percentage, gen);
    for (int i = 0; i < n_obs; ++i) {
        if (na_mask[i]) {
            observations[i][0] = std::numeric_limits<double>::quiet_NaN();  
        }
    }

    write_to_csv(output_dir + "cov_" + std::to_string(patient_id) + ".csv", X);
    write_to_csv(output_dir + "W_" + std::to_string(patient_id) + ".csv", W);
    write_to_csv(output_dir + "V_" + std::to_string(patient_id) + ".csv", V);
    write_to_csv(output_dir + "locs_" + std::to_string(patient_id) + ".csv", locs);
    write_to_csv(output_dir + "observations_" + std::to_string(patient_id) + ".csv", observations);
    // std::cout << "Generati dati per il paziente " << patient_id << " con " << n_obs << " osservazioni.\n";
}

void generate_data_for_all_patients(int num_patients, const std::vector<double>& a, const std::vector<double>& b,
const std::string& output_dir, int seed, double na_percentage, double mu) {
    std::mt19937 gen(seed);
    std::normal_distribution<> obs_dist(mu, mu/4);

    for (int patient_id = 0; patient_id < num_patients; ++patient_id) {
        int n_obs = std::round(obs_dist(gen));
        // std::cout << "Generando dati per il paziente " << patient_id << " con " << n_obs << " osservazioni.\n";
        generate_data_for_patient(patient_id, a, b, n_obs, gen, output_dir, na_percentage);
    }
}

/*
// test monolitico
TEST(mixed_srpde_test, mille_mono_automatico) {
    int seed = 1234; 
	std::size_t n_patients = 3; //4
    double mu = 1000.;
	
    std::vector<double> a = {-3.0, 4.0}; // Coefficienti per X: n_obs x 2
    // std::vector<double> b = {0.5,  0., -0.5}; // Coefficienti per V
    std::vector<double> b = generate_b_coefficients(n_patients, seed);

    std::cout << "b:" << std::endl;
    for (double value : b) {
        std::cout << value << " ";
    }
    std::cout << std::endl;
   																					
    double na_percentage = 0.;  // percentuale di valori NA 

    // define data
    std::vector<BlockFrame<double, int>> data;
    data.resize(n_patients);

    // define domain 
    std::string meshID = "unit_square_coarse";
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
    generate_data_for_all_patients(n_patients, a, b, output_dir, seed, na_percentage, mu);

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
    double lambda = 0.1; 

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
    std::cout << "beta: " << model.beta() << std::endl;
    std::cout << "alpha: " << model.alpha() << std::endl;
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

// test iterativo
TEST(mixed_srpde_test, mille_iter_automatico) {
       
    std::size_t n_patients = 3;
    double na_percentage = 0.;  // percentuale di valori NA 
    int seed = 1234; 
    double mu = 1000.;

    std::vector<double> a = {2.0, -1.5};  // Coefficienti per W
    // std::vector<double> b = {0.5, 0., -0.5};   // Coefficienti per V
    std::vector<double> b = generate_b_coefficients(n_patients, seed);

    // define data
    std::vector<BlockFrame<double, int>> data;
    data.resize(n_patients);

    // define domain 
    std::string meshID = "unit_square_coarse";
    std::string policyID = "richardson/";
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
    generate_data_for_all_patients(n_patients, a, b, output_dir, seed, na_percentage, mu);

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
    DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.n_elements() * n_patients, 1);
    PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> problem(domain.mesh, L, u);

    // define model
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
    
    // solve smoothing problem
    model.init();
    model.solve();

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
}*/

// PARAMETRIC TEST 
struct TestParams {
    int patientsN;
    double mu;
    int seed;
    double na_percentage;
    std::string meshID;
    double lambda;
};

const std::vector<TestParams> kPets = {
    // {3, 1000., 1234, 0., "unit_square_coarse", 0.1},
    // {3, 1000., 32874, 0., "unit_square_coarse", 0.1},
    // {10, 1000., 12894, 0., "unit_square_coarse", 0.1},
    // {10, 1000., 18937, 0., "unit_square_coarse", 0.1},
    // {15, 1000., 32847, 0., "unit_square_coarse", 0.1},
    // {15, 1000., 23784, 0., "unit_square_coarse", 0.1},
    // {20, 1000., 2453, 0., "unit_square_coarse", 0.1},
    // {20, 1000., 6575, 0., "unit_square_coarse", 0.1},
    // {30, 1000., 4352, 0., "unit_square_coarse", 0.1},
    // {30, 1000., 1332, 0., "unit_square_coarse", 0.1},
    // {60, 1000., 2784, 0., "unit_square_coarse", 0.1},
    // {60, 1000., 23784, 0., "unit_square_coarse", 0.1},
    // {90, 1000., 32784, 0., "unit_square_coarse", 0.1},
    // {90, 1000., 274, 0., "unit_square_coarse", 0.1},
    // {100, 1000., 38247, 0., "unit_square_coarse", 0.1},
    // {100, 1000., 1389, 0., "unit_square_coarse", 0.1},
    {500, 1000., 1389, 0., "unit_square_coarse", 0.1},
};

class MixedSRPDETest : public testing::TestWithParam<TestParams> {
  // You can implement all the usual fixture class members here.
  // To access the test parameter, call GetParam() from class TestWithParam<T>.
};

INSTANTIATE_TEST_SUITE_P(Pets, MixedSRPDETest, testing::ValuesIn(kPets));

TEST_P(MixedSRPDETest, monolithic) {

    TestParams params = GetParam();

    std::stringstream ss;
    ss << params.patientsN << " " << params.mu << " " << params.seed << " " << "monolithic" << " " << params.na_percentage << " " << params.meshID << " " << params.lambda;
    
    std::string filename = "prova_test.txt";
    // header:
    // patientsN mu seed policyID na_percentage meshID lambda [b] [beta] rmse duration maxcoeff

    std::string meshID = params.meshID;
    std::string locsID = "1000";
    std::string policyID = "monolithic";
	std::size_t n_patients = params.patientsN;
    double mu = params.mu;
    int seed = params.seed;

    auto start = std::chrono::high_resolution_clock::now();

    policyID = policyID + "/"; 
    locsID = locsID + "/"; 
	
    std::vector<double> a = {-3.0, 4.0}; // Coefficienti per X: n_obs x 2
    std::vector<double> b = generate_b_coefficients(n_patients, params.seed); // Coefficienti per V

    ss << " ["; for (double value : b) { ss << value << " "; } ss << "]";
   																					
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
    
    generate_data_for_all_patients(n_patients, a, b, output_dir, seed, na_percentage, mu);

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

    std::cout << "beta: " << model.beta() << std::endl;
    std::cout << "alpha: " << model.alpha() << std::endl;
    
    auto differences = (model.f() - f_).array().square().mean();
    auto rmse = std::sqrt(differences);
    std::cout<< "RMSE = " << rmse <<std::endl;

    std::cout << "max model.f(): "<< model.f().array().abs().maxCoeff() << std::endl;
    std::cout << "max f_: "<< (f_).array().abs().maxCoeff() << std::endl; 

    ss << " ["; for (double value : model.betanp()) { ss << value << " "; } ss << "] ";
    ss << rmse << " " << duration.count() << " " <<  (model.f().head(f_.rows()) - f_ ).array().abs().maxCoeff() << std::endl;

    std::cout << "max diff: " << (model.f().head(f_.rows()) - f_ ).array().abs().maxCoeff() << std::endl;

    std::ofstream outputFile; outputFile.open(filename, std::ios_base::app);
    if (outputFile.is_open()) { outputFile << ss.str();  outputFile.close();
    } else { std::cerr << "Unable to open file for writing" << std::endl; }

    EXPECT_TRUE(  (model.f().head(f_.rows()) - f_ ).array().abs().maxCoeff() < 1e-2 );
}

TEST_P(MixedSRPDETest, iterative) {

    TestParams params = GetParam();

    std::stringstream ss;
    ss << params.patientsN << " " << params.mu << " " << params.seed << " " << "iterative" << " " << params.na_percentage << " " << params.meshID << " " << params.lambda;
    
    std::string filename = "prova_test.txt";
    // header:
    // patientsN mu seed policyID na_percentage meshID lambda [b] [beta] rmse duration maxcoeff

    std::string meshID = params.meshID;
    std::string locsID = "1000";
    std::string policyID = "iterative";
	std::size_t n_patients = params.patientsN;
    double mu = params.mu;
    int seed = params.seed;

    auto start = std::chrono::high_resolution_clock::now();

    policyID = policyID + "/"; 
    locsID = locsID + "/"; 
	
    std::vector<double> a = {-3.0, 4.0}; // Coefficienti per X: n_obs x 2
    std::vector<double> b = generate_b_coefficients(n_patients, params.seed); // Coefficienti per V

    ss << " ["; for (double value : b) { ss << value << " "; } ss << "]";
   																					
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
    
    generate_data_for_all_patients(n_patients, a, b, output_dir, seed, na_percentage, mu);

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

    std::cout << "beta: " << model.beta() << std::endl;
    std::cout << "alpha: " << model.alpha() << std::endl;
    
    auto differences = (model.f() - f_).array().square().mean();
    auto rmse = std::sqrt(differences);
    std::cout<< "RMSE = " << rmse <<std::endl;

    std::cout << "max model.f(): "<< model.f().array().abs().maxCoeff() << std::endl;
    std::cout << "max f_: "<< (f_).array().abs().maxCoeff() << std::endl; 

    ss << " ["; for (double value : model.betanp()) { ss << value << " "; } ss << "] ";
    ss << rmse << " " << duration.count() << " " <<  (model.f().head(f_.rows()) - f_ ).array().abs().maxCoeff() << std::endl;

    std::cout << "max diff: " << (model.f().head(f_.rows()) - f_ ).array().abs().maxCoeff() << std::endl;

    std::ofstream outputFile; outputFile.open(filename, std::ios_base::app);
    if (outputFile.is_open()) { outputFile << ss.str();  outputFile.close();
    } else { std::cerr << "Unable to open file for writing" << std::endl; }

    EXPECT_TRUE(  (model.f().head(f_.rows()) - f_ ).array().abs().maxCoeff() < 1e-2 );
}

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
