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
#include <fdaPDE/pde.h>
#include <fdaPDE/utils.h>

#include <memory>
#include <type_traits>

#include "../model_base.h"
#include "../model_macros.h"
#include "../sampling_design.h"
#include "regression_base.h"
using fdapde::core::SMW;
using fdapde::core::SparseBlockMatrix;

namespace fdapde {
namespace models {

// MixedSRPDE model signature
template <typename RegularizationType, typename SolutionPolicy> class MixedSRPDE;

// implementation of MixedSRPDE for space-only regularization
template <>
class MixedSRPDE<SpaceOnly,monolithic> : public RegressionBase<MixedSRPDE<SpaceOnly,monolithic>, SpaceOnly> {
   private:
    typedef RegressionBase<SRPDE, SpaceOnly> Base;
    SparseBlockMatrix<double, 2, 2> A_ {};         // system matrix of non-parametric problem (2N x 2N matrix)
    fdapde::SparseLU<SpMatrix<double>> invA_ {};   // factorization of matrix A
    DVector<double> b_ {};                         // right hand side of problem's linear system (1 x 2N vector)

    using Base::Psi_;
    using Base::R0_;
    using Base::R1_;
    // cambiare con DMatrix<double> per fare resize():
    DMatrix<double> X_ {};      // dimensione: N osservazioni totali * p covariate gruppo specifiche
    // N = X_.rows()
    // p = X_.cols()

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

    N = Wg.rows(); // n*L
    n = Base::n_locs();
    L = N/n;
    q = Vp.cols();
    p = Wg.cols();
    
    X_.resize(N, p+L*q);

    Id = DVector<double>::Ones(N).asDiagonal(); // dim: N*N

    // Psi, R0, R1 hanno dimensioni N*N
    Psi = Kronecker(Id, Psi()); // forse serve dichiarare questo membro privato
    R0_ = Kronecker(Id, space_pde_.mass());  // I \kron R0
    R1_ = Kronecker(Id, space_pde_.stiff()); // I \kron R1

    // Wg dim(N,p)
    // Vp dim(N=n*L, q)

    if (!is_empty(X_)) { // computation of X
        X_.leftCols() = Wg; // matrix W: first column of X
        for ( i = 0; i < L ; i++ ) { // j parte da p (coariate gruppo specifiche) e finisce a q*n (ogni volta aggiunge q covariate paziente specifico)
            X_.block( i*n, p+i*q, n, q ) = Vp.middleRows(i*n, n); // matrix V: block of matrices V1,V2,...,VL (L: numero di pazienti)
        }
    }

    if (runtime().query(runtime_status::is_lambda_changed)) {

        // assemble system matrix for nonparameteric part
        A_ = SparseBlockMatrix<double, 2, 2>(
                -PsiTD() * W() * Psi(), lambda_D() * R1().transpose(),
                lambda_D() * R1(),      lambda_D() * R0()            );
        invA_.compute(A_);
        
        // prepare rhs of linear system
        b_.resize(A_.rows());
        b_.block(n_basis(), 0, n_basis(), 1) = lambda_D() * u();
        return;
    }
    if (runtime().query(runtime_status::require_W_update)) {
        // adjust north-west block of matrix A_ only
        A_.block(0, 0) = -PsiTD() * W() * Psi();
        invA_.compute(A_);
        return;
    }
    }

    void solve() {
        fdapde_assert(y().rows() != 0);
        DVector<double> sol;
    
        // parametric case
        // update rhs of SR-PDE linear system
        b_.block(0, 0, n_basis(), 1) = -PsiTD() * lmbQ(y());   // -\Psi^T*D*Q*z
        // matrices U and V for application of woodbury formula
        U_ = DMatrix<double>::Zero(2 * n_basis(), q());
        U_.block(0, 0, n_basis(), q()) = PsiTD() * W() * X();
        V_ = DMatrix<double>::Zero(q(), 2 * n_basis());
        V_.block(0, 0, q(), n_basis()) = X().transpose() * W() * Psi();
        // solve system (A_ + U_*(X^T*W_*X)*V_)x = b using woodbury formula from linear_algebra module
        sol = SMW<>().solve(invA_, U_, XtWX(), V_, b_);
        // store result of smoothing
        f_ = sol.head(n_basis());
        beta_ = invXtWX().solve(X().transpose() * W()) * (y() - Psi() * f_);
        // store PDE misfit
        g_ = sol.tail(n_basis());
        return;
    }
    // GCV support
    double norm(const DMatrix<double>& op1, const DMatrix<double>& op2) const { return (op1 - op2).squaredNorm(); }

    // getters
    const SparseBlockMatrix<double, >& X() const { return X_; }
    const SparseBlockMatrix<double, >& Psi() const { return Psi_; }
    const SparseBlockMatrix<double, >& R0() const { return R0_; }
    const SparseBlockMatrix<double, >& R1() const { return R1_; }


    virtual ~MixedSRPDE() = default;
}
}
}