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

   public:
    IMPORT_REGRESSION_SYMBOLS;

    using Base::lambda_D;   // smoothing parameter in space
    using Base::n_basis;    // number of spatial basis
    using Base::runtime;    // runtime model status
    using RegularizationType = SpaceOnly;
    static constexpr int n_lambda = 1;
    // constructor
    MixedSRPDE() = default;
    MixedSRPDE(const pde_ptr& pde, Sampling s) : Base(pde, s) {};

    void init_model() { 

    // Notazione:
    // N: numero di osservazioni totali
    // p: covariate gruppo specifiche
    // n: numero di osservazioni per paziente
    // L: numero di pazienti
    // q: covariate paziente specifiche
    
    std::cout << "init_model()" << std::endl; // the run doesn't even reach this line
    N = Wg().rows(); // n*L
    std::cout << "N: " << N << std::endl;
    n = 3; //n_locs(); ?? Aggiunto manualmente
    std::cout << "n: " << n << std::endl;
    L = N/n;
    std::cout << "L: " << L << std::endl;
    qV = Vp().cols();
    std::cout << "qV: " << qV << std::endl;
    p = Wg().cols();
    std::cout << "p: " << p << std::endl;
    
    I_.resize(n,n);
    //I_.resize(n,n);
    //I_.resize(n_locs(), n_locs());
    I_.setIdentity();
    std::cout << "I_: " << I_ << std::endl;
    
    mPsi_ = mPsi();
    //std::cout << "mPsi_" << sizeof(mPsi_) << std::endl; 
    
    mPsiTD_ = mPsiTD();
    //std::cout << "mPsiTD_" << sizeof(mPsiTD_) << std::endl; 
    
    // std::cout << "X_: " << X_ << std::endl; // X is empty at this point
    //std::cout << "X_ empty: " << is_empty(X_) << std::endl; // returns 1 -> enter the if 

    if (is_empty(X_)) { // computation of X
 	X_.resize(N, p+n*qV);
    	//std::cout <<  "Wg: " << Wg() << std::endl;
    	//std::cout <<  "X_: " << X_ << std::endl;
	// std::cout << "X_ leftcols before: " << X_.leftCols(p) << std::endl;
        X_.leftCols(p) = Wg(); // matrix W: first column of X
       	//std::cout << "X_ leftcols after: " << X_ << std::endl;
       	// fin qui ok -- X_ viene popolata con Wg nella prima colonna
        
        for ( int i = 0; i < n ; i++ ) { // j parte da p (coariate gruppo specifiche) e finisce a q*L (ogni volta aggiunge q covariate paziente specifico)
            // std::cout << "Vp:" << Vp().middleRows(i*L, L) << std::endl;
            X_.block( i*L, p+i*qV, L, qV ) = Vp().middleRows(i*L, L); // matrix V: block of matrices V1,V2,...,Vn (n: numero di pazienti)
        }
    }
    
    std::cout << "X_ rows: " << X_.rows() << std::endl;
    std::cout << "X_ cols: " << X_.cols() << std::endl;
    // std::cout << "X_: " << X_ << std::endl;

    if (runtime().query(runtime_status::is_lambda_changed)) {
        std::cout << "entro dell'if" << std::endl;
        // assemble system matrix for nonparameteric part
    
        // vediamo come sono fatte ste matrici...
        std::cout << "mPsiTD_ rows:" << mPsiTD_.rows() << std::endl;
        std::cout << "mPsiTD_ cols:" << mPsiTD_.cols() << std::endl;
        
        std::cout << "W rows:" << W().rows() << std::endl;
        std::cout << "W cols:" << W().cols() << std::endl;
        // LA W() DERIVANTE DA ANLYZE DATA HA DIMENSIONI 3600 X 3600 
        // QUANDO DOVREBBE ESSERE 10800 X 10800
        // 3600 X 3 = 108000 ...INDAGARE QUESTA RELAZIONE NEL CODICE!

        std::cout << "mPsi_ rows:" << mPsi_.rows() << std::endl;
        std::cout << "mPsi_ cols:" << mPsi_.cols() << std::endl;

        std::cout << "R0 rows:" << R0().rows() << std::endl;
        std::cout << "R0 cols:" << R0().cols() << std::endl;

        std::cout << "R1 rows:" << R1().rows() << std::endl;
        std::cout << "R1 cols:" << R1().cols() << std::endl;

        std::cout << "lambda_D:" << lambda_D() << std::endl;

        A_ = SparseBlockMatrix<double, 2, 2>(
                -mPsiTD_ * W() * mPsi_, lambda_D() * R1().transpose(),
                lambda_D() * R1(),      lambda_D() * R0()            );
        invA_.compute(A_);
        std::cout << "A assemblata" << std::endl;

        // prepare rhs of linear system (Questa parte sembra andare)
        b_.resize(A_.rows());
        std::cout << "b resized" << std::endl;
        b_.block(n_basis(), 0, n_basis(), 1) = lambda_D() * u();
        std::cout << "b completata" << std::endl;
        return;
    }
    if (runtime().query(runtime_status::require_W_update)) {
        std::cout << "altro if" << std::endl;
        // adjust north-west block of matrix A_ only
        A_.block(0, 0) = -mPsiTD_ * W() * mPsi_;
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
        b_.block(0, 0, n_basis(), 1) = -mPsiTD_ * lmbQ(y());   // -\Psi^T*D*Q*z
        // matrices U and V for application of woodbury formula
        std::cout << "U_" << std::endl;
        U_ = DMatrix<double>::Zero(2 * n_basis(), q());
        std::cout << "U_.block" << std::endl;
        U_.block(0, 0, n_basis(), q()) = mPsiTD_ * W() * X();
        std::cout << "V_" << std::endl;
        V_ = DMatrix<double>::Zero(q(), 2 * n_basis());
        std::cout << "V_.block" << std::endl;
        V_.block(0, 0, q(), n_basis()) = X().transpose() * W() * mPsi_;
        // solve system (A_ + U_*(X^T*W_*X)*V_)x = b using woodbury formula from linear_algebra module
        std::cout << "sol" << std::endl;
        sol = SMW<>().solve(invA_, U_, XtWX(), V_, b_);
        // store result of smoothing
        std::cout << "f" << std::endl;
        f_ = sol.head(n_basis());
        std::cout << "beta" << std::endl;
        beta_ = invXtWX().solve(X().transpose() * W()) * (y() - mPsi_ * f_);
        // store PDE misfit
        std::cout << "g" << std::endl;
        g_ = sol.tail(n_basis());
        return;
    }
    // GCV support
    double norm(const DMatrix<double>& op1, const DMatrix<double>& op2) const { return (op1 - op2).squaredNorm(); }

    // getters
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
