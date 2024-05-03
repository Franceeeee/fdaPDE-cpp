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

#ifndef __MIXED_SRPDE_H__
#define __MIXED_SRPDE_H__

#include <fdaPDE/linear_algebra.h>
//#include <fdaPDE/linear_algebra/kronecker_product.h>
#include <fdaPDE/pde.h>
#include <fdaPDE/utils.h>
#include <Eigen/Dense> // for leftCols(p); p # of cols

#include <memory>
#include <type_traits>

#include "../model_base.h"
#include "../model_macros.h"
#include "../sampling_design.h"
#include "regression_base.h"
using fdapde::core::SMW;
using fdapde::core::SparseBlockMatrix;
using fdapde::core::pde_ptr;
using fdapde::core::Kronecker;

namespace fdapde {
namespace models {

// MixedSRPDE model signature
template <typename RegularizationType, typename SolutionPolicy> class MixedSRPDE;

// implementation of MixedSRPDE for space-only regularization
template <>
class MixedSRPDE<SpaceOnly,monolithic> : public RegressionBase<MixedSRPDE<SpaceOnly,monolithic>, SpaceOnly> {
   private:
    typedef RegressionBase<MixedSRPDE, SpaceOnly> Base;
    SparseBlockMatrix<double, 2, 2> A_ {};         // system matrix of non-parametric problem (2N x 2N matrix)
    fdapde::SparseLU<SpMatrix<double>> invA_ {};   // factorization of matrix A
    DVector<double> b_ {};                         // right hand side of problem's linear system (1 x 2N vector)
    
    // using SamplingBase<Model>::Psi;
    // using Base::R0;
    // using Base::R1;

    DMatrix<double> X_ {};      // dimensione: N osservazioni totali * p covariate gruppo specifiche
	
    int N;
    int n;
    int L;
    int qV;
    int p;
    
    SpMatrix<double> I_;   // N x N sparse identity matrix 
    
    SpMatrix<double> mPsi_;
    SpMatrix<double> mPsiTD_;
    
    void init_X() {
    
    N = Wg().rows(); // n*L
    std::cout << "N: " << N << std::endl;
    n = n_locs(); //?? Aggiunto manualmente
    std::cout << "n: " << n << std::endl;
    L = N/n;
    std::cout << "L: " << L << std::endl;
    qV = Vp().cols();
    std::cout << "qV: " << qV << std::endl;
    p = Wg().cols();
    std::cout << "p: " << p << std::endl;
     
 	X_ = DMatrix<double>::Zero(N, q());
    X_.leftCols(p) = Wg(); // matrix W: first column of X
    
    for ( int i = 0; i < L ; i++ ) { // j parte da p (coariate gruppo specifiche) e finisce a q*L (ogni volta aggiunge q covariate paziente specifico)
        X_.block( i*n_locs(), p+i*qV, n_locs(), qV ) = Vp().middleRows(i*n_locs(), n_locs()); // matrix V: block of matrices V1,V2,...,Vn (n: numero di pazienti))
    }
    
    std::cout << "X_ rows: " << X_.rows() << std::endl;
    std::cout << "X_ cols: " << X_.cols() << std::endl;

   }
   
   public:
    IMPORT_REGRESSION_SYMBOLS;

    using Base::lambda_D;   // smoothing parameter in space
    using Base::n_basis;    // number of spatial basis
    using Base::runtime;    // runtime model status
    using RegularizationType = SpaceOnly;
    static constexpr int n_lambda = 1;
    // constructor
    MixedSRPDE() = default;
    MixedSRPDE(const pde_ptr& pde, Sampling s) : Base(pde, s) { };
    
    void analyze_data() {
    	// build multi-domain model design matrix 
    	init_X();
    	std::cout << "init_X()" << std::endl;
        // initialize empty masks
        if (!y_mask_.size()) y_mask_.resize(Base::n_locs()); // da sostituire N 
        if (!nan_mask_.size()) nan_mask_.resize(Base::n_locs()); // da sostiture N
        // compute q x q dense matrix X^\top*W*X and its factorization
        if (has_weights() && df_.is_dirty(WEIGHTS_BLK)) {
            W_ = df_.template get<double>(WEIGHTS_BLK).col(0).asDiagonal(); // !!! da modificare !!!
            model().runtime().set(runtime_status::require_W_update);
        } else if (is_empty(W_)) {
            W_ = DVector<double>::Ones(N).asDiagonal(); // W_ in R^{N times N}
        }
        
        std::cout << "W_ " << W_.rows()<< " " << W_.cols() << std::endl;
    	
        // compute q x q dense matrix X^\top*W*X and its factorization
        if (has_covariates() && (df_.is_dirty(DESIGN_MATRIX_BLK) || df_.is_dirty(WEIGHTS_BLK))) {
            XtWX_ = X().transpose() * W_ * X(); 
            invXtWX_ = XtWX_.partialPivLu(); 
        }
        
        std::cout << "XtWX_ " << XtWX_.rows()<< " " << XtWX_.cols() << std::endl;
    	std::cout << "invXtWX_ " << invXtWX_.rows()<< " " << invXtWX_.cols() << std::endl;
    
        // derive missingness pattern from observations vector (if changed)
        if (df_.is_dirty(OBSERVATIONS_BLK)) {
            n_nan_ = 0;
            for (std::size_t i = 0; i < df_.template get<double>(OBSERVATIONS_BLK).size(); ++i) {
                if (std::isnan(y()(i, 0))) {   // requires -ffast-math compiler flag to be disabled
                    nan_mask_.set(i);
                    n_nan_++;
                    df_.template get<double>(OBSERVATIONS_BLK)(i, 0) = 0.0;   // zero out NaN
                }
            }
            if (has_nan()) model().runtime().set(runtime_status::require_psi_correction);
        }
        return;
    }
    
    void init_model() { 

    // Notazione:
    // N: numero di osservazioni totali
    // p: covariate gruppo specifiche
    // n: numero di osservazioni per paziente
    // L: numero di pazienti
    // q: covariate paziente specifiche
    
    std::cout << "init_model()" << std::endl; // the run doesn't even reach this line
    
    std::cout << "n_basis: " << n_basis() << std::endl;
    
    I_.resize(L,L);
    I_.setIdentity();
    
    mPsi_ = mPsi();
    
    mPsiTD_ = mPsiTD();
   
    std::cout << "X_ rows: " << X_.rows() << std::endl;
    std::cout << "X_ cols: " << X_.cols() << std::endl;
    // std::cout << "X_: " << X_ << std::endl;
    //XtWX_ = X().transpose() * W_ * X(); 
    //invXtWX_ = XtWX_.partialPivLu();
    
    std::cout << "XtWX_ " << XtWX_.rows()<< " " << XtWX_.cols() << std::endl;
    std::cout << "invXtWX_ " << invXtWX_.rows()<< " " << invXtWX_.cols() << std::endl;
    
    if (runtime().query(runtime_status::is_lambda_changed)) {
        std::cout << "entro dell'if" << std::endl;
        // assemble system matrix for nonparameteric part
    
        // vediamo come sono fatte ste matrici...
        std::cout << "mPsiTD_ rows:" << mPsiTD_.rows() << std::endl;
        std::cout << "mPsiTD_ cols:" << mPsiTD_.cols() << std::endl;
        
        std::cout << "W rows:" << W().rows() << std::endl;
        std::cout << "W cols:" << W().cols() << std::endl;
        
        std::cout << "mPsi_ rows:" << mPsi_.rows() << std::endl;
        std::cout << "mPsi_ cols:" << mPsi_.cols() << std::endl;
        
        std::cout << "mPsiTD_ rows:" << mPsiTD_.rows() << std::endl;
        std::cout << "mPsiTD_ cols:" << mPsiTD_.cols() << std::endl;

        std::cout << "R0 rows:" << R0().rows() << std::endl;
        std::cout << "R0 cols:" << R0().cols() << std::endl;

        std::cout << "R1 rows:" << R1().rows() << std::endl;
        std::cout << "R1 cols:" << R1().cols() << std::endl;

        std::cout << "lambda_D:" << lambda_D() << std::endl;
	std::cout << "north-west" << std::endl;
	std::cout << (- mPsiTD() * mPsi()).rows() << " "  << (- mPsiTD() * mPsi()).cols() << std::endl; //* mPsi()).rows() << " " << (-mPsiTD() * W() * mPsi()).cols() << std::endl; 
        A_ = SparseBlockMatrix<double, 2, 2>(
                -mPsiTD()  * W() * mPsi(), lambda_D() * R1().transpose(), // 
                 lambda_D() * R1(),        lambda_D() * R0()            );
        invA_.compute(A_);
        std::cout << "A assemblata" << std::endl;

        // prepare rhs of linear system (Questa parte sembra andare)
        b_.resize(A_.rows());
        std::cout << "b resized " << b_.rows() << " " << b_.cols() << std::endl;
        std::cout << "u " << u().rows() << " " << u().cols() << std::endl;
        
        for( int i=0; i < L; ++i){
        	b_.block((L+i)*n_basis(), 0, n_basis(), 1) = lambda_D() * u(); // !!!! n_basis() * L
        }
        std::cout << "b completata" << std::endl;
        return;
    }
    if (runtime().query(runtime_status::require_W_update)) {
        std::cout << "altro if" << std::endl;
        // adjust north-west block of matrix A_ only
        A_.block(0, 0) = -mPsiTD() * mPsi(); // W() problemi . 
        invA_.compute(A_);
        std::cout << "finito" << std::endl;
        return;
    }
    }

    void solve() {
        fdapde_assert(y().rows() != 0);
        DVector<double> sol;

        std::cout << "b_.block" << std::endl;
        // parametric case
        // update rhs of SR-PDE linear system
        std::cout << "y() " << y().rows() << " " << y().cols() << std::endl;
        std::cout << "lmbQ(y()) " << lmbQ(y()).rows() << " " << lmbQ(y()).cols()  << std::endl;
        b_.block(0, 0, L*n_basis(), 1) = -mPsiTD() * lmbQ(y());   // -\Psi^T*D*Q*z
        // matrices U and V for application of woodbury formula
        std::cout << "U_" << std::endl;
        U_ = DMatrix<double>::Zero(2 * L * n_basis(), q());
        std::cout << "U_.block" << std::endl;
        U_.block(0, 0, L* n_basis(), q()) = mPsiTD_  * X(); // * W() * X();
        std::cout << "V_" << std::endl;
        V_ = DMatrix<double>::Zero(q(), 2 * L * n_basis());
        std::cout << "V_.block" << std::endl;
        V_.block(0, 0, q(), L * n_basis()) = X().transpose() * mPsi(); // W() * mPsi()
        // solve system (A_ + U_*(X^T*W_*X)*V_)x = b using woodbury formula from linear_algebra module
        std::cout << "sol" << std::endl;
        sol = SMW<>().solve(invA_, U_, XtWX(), V_, b_);
        // store result of smoothing
        std::cout << "f" << std::endl;
        f_ = sol.head(L*n_basis());
        std::cout << "beta" << std::endl;
        beta_ = invXtWX().solve(X().transpose() ) * (y() - mPsi_ * f_); // X().transpose() * W()
        // store PDE misfit
        std::cout << "g" << std::endl;
        g_ = sol.tail(L*n_basis());
        return;
    }
    // GCV support
    double norm(const DMatrix<double>& op1, const DMatrix<double>& op2) const { return (op1 - op2).squaredNorm(); }

    // getters
    
    std::size_t q() const {
        return p+L*qV;
    }
    
    const DiagMatrix<double>& W() const { return W_; }
    const DMatrix<double>& X() const { return X_; }  
    const DMatrix<double>& Wg() const { return df_.template get<double>(DESIGN_MATRIX_BLK); } 
    const DMatrix<double>& Vp() const { return df_.template get<double>(MIXED_EFFECTS_BLK); } 
    const SpMatrix<double> mPsi() const { return Kronecker(I_, Psi()); }
    const SpMatrix<double> mPsiTD() const { return Kronecker(I_, PsiTD(not_nan())); }
    const SpMatrix<double> R0() const { return Kronecker(I_, pde_.mass()); }
    const SpMatrix<double> R1() const { return Kronecker(I_, pde_.stiff()); }
    
    virtual ~MixedSRPDE() = default;
};
}   // namespace models
}   // namespace fdapde

#endif   // __MIXED_SRPDE_H__
