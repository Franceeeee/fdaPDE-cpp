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
using fdapde::testing::almost_equal;
using fdapde::testing::MeshLoader;
using fdapde::testing::read_csv;

// Lambda function per la valutazione delle covariate
auto lambda_cov = [](double x, double y) {
    return 2.0 * x + 3.0 * y;  // Dummy function for covariate
};

// Lambda function per la parte non parametrica
auto lambda_np = [](double x, double y) {
    return std::sin(x * y);  // Dummy non-parametric function
};

// Funzione per scrivere in CSV
void write_to_csv(const std::string& filename, const std::vector<std::vector<double>>& data) {
    std::ofstream file(filename);
    for (const auto& row : data) {
        for (size_t i = 0; i < row.size(); ++i) {
            if (std::isnan(row[i])) {
                file << "NaN";  // Scrive "NA" nel file se il valore è NaN
            } else {
                file << row[i];
            }
            if (i < row.size() - 1) {
                file << ",";
            }
        }
        file << "\n";
    }
    file.close();
}

// Funzione per generare punti casuali nel quadrato unitario [0, 1] x [0, 1]
std::vector<std::vector<double>> generate_random_points(int n_points, std::mt19937& gen) {
    std::uniform_real_distribution<> dis(0.0, 1.0);
    std::vector<std::vector<double>> points(n_points, std::vector<double>(2));

    for (int i = 0; i < n_points; ++i) {
        points[i][0] = dis(gen);  // x
        points[i][1] = dis(gen);  // y
    }

    return points;
}

// Funzione per creare una maschera di NA basata sulla percentuale
std::vector<bool> create_na_mask(int size, double na_percentage, std::mt19937& gen) {
    std::vector<bool> mask(size, false);  // Inizializza la maschera senza NA (false)
    
    int num_na = static_cast<int>(std::round(size * na_percentage));  // Numero di NA da inserire
    std::uniform_int_distribution<> dis(0, size - 1);
    
    // Imposta randomicamente i valori della maschera a true per indicare NA
    for (int i = 0; i < num_na; ++i) {
        int index;
        do {
            index = dis(gen);  // Genera un indice casuale
        } while (mask[index]);  // Assicura che non venga scelto due volte lo stesso indice
        mask[index] = true;
    }
    
    return mask;
}

// Funzione per generare dati per un singolo paziente
void generate_data_for_patient(int patient_id, const std::vector<double>& a, const std::vector<double>& b, int n_obs, std::mt19937& gen, const std::string& output_dir, double na_percentage) {
    // Genera i punti casuali nel quadrato unitario
    std::vector<std::vector<double>> locs = generate_random_points(n_obs, gen);
    
    // Matrici W, V e vettore delle osservazioni y
    std::vector<std::vector<double>> W(n_obs, std::vector<double>(a.size()));
    std::vector<std::vector<double>> V(n_obs, std::vector<double>(b.size()));
    std::vector<std::vector<double>> observations(n_obs, std::vector<double>(1));
    std::vector<std::vector<double>> true_values(n_obs, std::vector<double>(1)); // Salviamo le osservazioni corrette (senza rumore)
    
    // Distribuzione per generare le covariate
    std::uniform_real_distribution<> dis(0.0, 1.0);
    std::normal_distribution<> noise_dist(0.0, 0.1); // Rumore gaussiano con media 0 e deviazione standard 0.1
    
    // Per ogni osservazione, calcoliamo W, V e y
    for (int i = 0; i < n_obs; ++i) {
        double y = 0.0;

        // Genera W e V tramite sampling
        for (size_t j = 0; j < a.size(); ++j) {
            double cov_value = lambda_cov(locs[i][0], locs[i][1]); // Usa le coordinate delle locazioni
            W[i][j] = cov_value; 
            y += W[i][j] * a[j];
        }
        for (size_t j = 0; j < b.size(); ++j) {
            double cov_value = lambda_cov(locs[i][0], locs[i][1]); // Usa le coordinate delle locazioni
            V[i][j] = cov_value;
            y += V[i][j] * b[j];
        }

        // Aggiungi la parte non parametrica f(p)
        double f_value = lambda_np(locs[i][0], locs[i][1]);
        y += f_value;

        // Salva l'osservazione corretta
        true_values[i][0] = y;

        // Aggiungi rumore all'osservazione
        observations[i][0] = y + noise_dist(gen);
    }

    // Crea una maschera di NA
    std::vector<bool> na_mask = create_na_mask(n_obs, na_percentage, gen);

    // Applica la maschera di NA a observations e true_values
    for (int i = 0; i < n_obs; ++i) {
        if (na_mask[i]) {
            observations[i][0] = std::numeric_limits<double>::quiet_NaN();  // Imposta il valore a NaN
            true_values[i][0] = std::numeric_limits<double>::quiet_NaN();   // Imposta il valore a NaN anche in f_hat
        }
    }


    // Salva W, V, locazioni, osservazioni e f_hat in CSV
    write_to_csv(output_dir + "W_" + std::to_string(patient_id) + ".csv", W);
    write_to_csv(output_dir + "V_" + std::to_string(patient_id) + ".csv", V);
    write_to_csv(output_dir + "locs_" + std::to_string(patient_id) + ".csv", locs);
    write_to_csv(output_dir + "observations_" + std::to_string(patient_id) + ".csv", observations);
    write_to_csv(output_dir + "f_hat" + std::to_string(patient_id) + ".csv", {true_values}); // Salva le osservazioni corrette
}

// Funzione principale per gestire più pazienti
void generate_data_for_all_patients(int num_patients, const std::vector<double>& a, const std::vector<double>& b, const std::string& output_dir, int seed, double na_percentage) {
    // Imposta il generatore casuale con il seed specificato
    std::mt19937 gen(seed);
    
    // Distribuzione normale per il numero di osservazioni per ciascun paziente
    std::normal_distribution<> obs_dist(1000.0, 300.0);
    
    for (int patient_id = 1; patient_id <= num_patients; ++patient_id) {
        // Campiona il numero di osservazioni per questo paziente
        int n_obs = std::round(obs_dist(gen));
        
        // Assicuriamoci che n_obs sia almeno 1
        if (n_obs < 1) n_obs = 1;

        std::cout << "Generando dati per il paziente " << patient_id << " con " << n_obs << " osservazioni.\n";
        
        // Genera i dati per il paziente corrente
        generate_data_for_patient(patient_id, a, b, n_obs, gen, output_dir, na_percentage);
    }
}

// test monolitico
TEST(mixed_srpde_test, mille_mono) {

    // Definisci i coefficienti
    std::vector<double> a = {2.0, -1.5};  // Coefficienti per W
    std::vector<double> b = {0.5, 1.2};   // Coefficienti per V
    
    // Numero di pazienti
    int num_patients = 2;

    // Percentuale di NA
    double na_percentage = 0.1;  // Percentuale di valori NA (10%)

    // Seed casuale
    int seed = 1234;  // Specifica il seed

    std::size_t n_patients = num_patients;
    std::vector<BlockFrame<double, int>> data;
    data.resize(n_patients);

    // define domain 
    std::string meshID = "unit_square";
    std::string policyID = "monolithic/";
    std::string locsID = "1000/";
    MeshLoader<Mesh2D> domain(meshID);
    meshID = meshID + "/"; 

    // Output directory
    std::string output_dir = "../data/models/mixed_srpde/2D_test2/" + meshID + policyID + locsID;

    // Genera dati per tutti i pazienti
    generate_data_for_all_patients(num_patients, a, b, output_dir, seed, na_percentage);

    // import data from files
    for(std::size_t i = 0; i<n_patients; i++){
        std::string Wname = "W_" + std::to_string(i+1);
        std::string Vname = "V_" + std::to_string(i+1);
        std::string locsname = "locs_" + std::to_string(i+1);
        std::string yname = "observations_" + std::to_string(i+1);
        data[i].read_csv<double>(W_BLOCK, "../data/models/mixed_srpde/2D_test2/" + meshID + policyID + locsID + Wname + ".csv");
        data[i].read_csv<double>(V_BLOCK, "../data/models/mixed_srpde/2D_test2/" + meshID + policyID + locsID + Vname + ".csv");
        data[i].read_csv<double>(Y_BLOCK, "../data/models/mixed_srpde/2D_test2/" + meshID + policyID + locsID + yname + ".csv");
        data[i].read_csv<double>(LOCS_BLOCK, "../data/models/mixed_srpde/2D_test2/" + meshID + policyID + locsID + locsname + ".csv");
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
    
    DMatrix<double> f_estimate = read_csv<double>("../data/models/mixed_srpde/2D_test2/" + 
    												 meshID + policyID + locsID + "f_hat.csv");

    std::cout << "Monolithic: " << (model.f() - f_estimate ).array().abs().maxCoeff() << std::endl;
    EXPECT_TRUE(  (model.f() - f_estimate ).array().abs().maxCoeff() < 1e-6 );
}


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
