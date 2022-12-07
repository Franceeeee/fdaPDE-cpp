#include <cstddef>
#include <gtest/gtest-typed-test.h>
#include <gtest/gtest.h> // testing framework

#include "../utils/Constants.h"
#include "core/utils/Symbols.h"
using fdaPDE::testing::DOUBLE_TOLERANCE;
#include "../../fdaPDE/core/NLA/KroneckerProduct.h"
using fdaPDE::core::NLA::Kronecker;

TEST(KroneckerProductTest, DenseSquareOperands) {
  // define dense operands (square matrices)
  DMatrix<double> m1;
  m1.resize(2,2);
  m1 << 1,2,3,4;
  DMatrix<double> n1 = DMatrix<double>::Identity(2,2);
  // evaluate kronecker product n1 \kron m1
  DMatrix<double> kron1 = Kronecker(n1, m1);
  // expected result
  DMatrix<double> expected1;
  expected1.resize(4,4);
  expected1 <<
    1,2,0,0,
    3,4,0,0,
    0,0,1,2,
    0,0,3,4;

  // check equality
  for(std::size_t i = 0; i < 4; ++i){
    for(std::size_t j = 0; j < 4; ++j){
      EXPECT_NEAR(kron1(i,j), expected1(i,j), DOUBLE_TOLERANCE);
    }
  }
}

TEST(KroneckerProductTest, DenseRectangularOperands){
  // define dense operands
  DMatrix<double> m1;
  m1.resize(2,3);
  m1 << 1,2,0,4,0,1;
  DMatrix<double> n1;
  n1.resize(2,2);
  n1 << 1,1,0,1;
  // evaluate kronecker product m1 \kron n1
  DMatrix<double> kron1 = Kronecker(m1, n1);
  // expected result
  DMatrix<double> expected1;
  expected1.resize(4,6);
  expected1 <<
    1,1,2,2,0,0,
    0,1,0,2,0,0,
    4,4,0,0,1,1,
    0,4,0,0,0,1;

  // check equality
  for(std::size_t i = 0; i < 4; ++i){
    for(std::size_t j = 0; j < 6; ++j){
      EXPECT_NEAR(kron1(i,j), expected1(i,j), DOUBLE_TOLERANCE);
    }
  }
}

TEST(KroneckerProductTest, SparseSquareOperands){
  std::vector<Eigen::Triplet<double>> coeff;
  coeff.push_back(Eigen::Triplet<double>(0,0, 2.0));
  coeff.push_back(Eigen::Triplet<double>(1,1, 3.0));
  coeff.push_back(Eigen::Triplet<double>(1,0, 1.0));
  coeff.push_back(Eigen::Triplet<double>(2,2, 1.0));
  SpMatrix<double> m1(3,3);
  m1.setFromTriplets(coeff.begin(), coeff.end());
  m1.makeCompressed();
  SpMatrix<double> n1(3,3);
  n1.setIdentity();
  
  // evaluate kronecker product n1 \kron m1
  SpMatrix<double> kron1 = Kronecker(n1, m1);
  // expected result
  DMatrix<double> expected1;
  expected1.resize(9,9);
  expected1 <<
    2,0,0,0,0,0,0,0,0,
    1,3,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,
    0,0,0,2,0,0,0,0,0,
    0,0,0,1,3,0,0,0,0,
    0,0,0,0,0,1,0,0,0,
    0,0,0,0,0,0,2,0,0,
    0,0,0,0,0,0,1,3,0,
    0,0,0,0,0,0,0,0,1;
  
  // check equality
  for(int k = 0; k < kron1.outerSize(); ++k) {
    for(SpMatrix<double>::InnerIterator it(kron1,k); it; ++it){
      EXPECT_NEAR(expected1(it.row(),it.col()), it.value(), DOUBLE_TOLERANCE);
    }
  }
}

TEST(KroneckerProductTest, SparseRectangularOperands){
  std::vector<Eigen::Triplet<double>> coeff;
  coeff.push_back(Eigen::Triplet<double>(0,0, 2.0));
  coeff.push_back(Eigen::Triplet<double>(1,1, 3.0));
  coeff.push_back(Eigen::Triplet<double>(1,0, 1.0));
  coeff.push_back(Eigen::Triplet<double>(1,2, 1.0));
  SpMatrix<double> m1(2,3);
  m1.setFromTriplets(coeff.begin(), coeff.end());
  m1.makeCompressed();

  std::vector<Eigen::Triplet<double>> coeff2;
  coeff2.push_back(Eigen::Triplet<double>(0,0, 5.0));
  coeff2.push_back(Eigen::Triplet<double>(0,1, 6.0));
  coeff2.push_back(Eigen::Triplet<double>(1,1, 1.0));
  SpMatrix<double> n1(3,2);
  n1.setFromTriplets(coeff2.begin(), coeff2.end());
  n1.makeCompressed();
  
  // evaluate kronecker product m1 \kron n1
  SpMatrix<double> kron1 = Kronecker(m1, n1);
  // expected result
  DMatrix<double> expected1;
  expected1.resize(6,6);
  expected1 <<
    10, 12, 0,  0, 0, 0,
    0,  2,  0,  0, 0, 0,
    0,  0,  0,  0, 0, 0,
    5,  6, 15, 18, 5, 6,
    0,  1,  0,  3, 0, 1,
    0,  0,  0,  0, 0, 0;

  // check equality
  for(int k = 0; k < kron1.outerSize(); ++k) {
    for(SpMatrix<double>::InnerIterator it(kron1,k); it; ++it){
      EXPECT_NEAR(expected1(it.row(),it.col()), it.value(), DOUBLE_TOLERANCE);
    }
  }
}