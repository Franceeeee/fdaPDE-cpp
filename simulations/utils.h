#include <iostream>
#include <vector>
#include <string>
#include <random>
#include <cmath>

#include <cstddef>
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
using fdapde::core::Mesh;
#include "../test/src/utils/constants.h"
#include "../test/src/utils/mesh_loader.h"
#include "../test/src/utils/utils.h"
using fdapde::testing::MeshLoader;
using fdapde::testing::read_csv;
using fdapde::core::CSVReader;

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

