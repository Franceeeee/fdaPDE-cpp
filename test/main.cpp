#include <gtest/gtest.h> // testing framework
// include eigen now to avoid possible linking errors
#include <Eigen/Dense>
#include <Eigen/Sparse>

// regression test suite
/*#include "src/srpde_test.cpp"
#include "src/strpde_test.cpp"
#include "src/gsrpde_test.cpp"
#include "src/qsrpde_test.cpp"
#include "src/gcv_srpde_test.cpp"
#include "src/gcv_qsrpde_test.cpp"
#include "src/gcv_srpde_newton_test.cpp"*/
// #include "src/kcv_srpde_test.cpp"
// functional test suite
//#include "src/fpca_test.cpp"
//#include "src/fpls_test.cpp"
//#include "src/centering_test.cpp"
#include "src/mixed_srpde_test.cpp"


//int main(int argc, char **argv){
  // start testing
  //testing::InitGoogleTest(&argc, argv);
  
  //return RUN_ALL_TESTS();
//}

// cyclic execution
int main(int argc, char **argv){
  // start testing
  testing::InitGoogleTest(&argc, argv);
    int numRuns = 30; 
    for (int i = 0; i < numRuns; ++i) {
        std::cout << "===== Run " << i+1 << " =====" << std::endl;
        int result = RUN_ALL_TESTS();
        if (result != 0) {
            std::cerr << "Google Test returned non-zero exit code: " << result << std::endl;
            return result;
        }
    }
}
