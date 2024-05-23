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
    n = n_locs(); 
    L = N/n;
    qV = Vp().cols();
    p = Wg().cols();
     
    X_ = DMatrix<double>::Zero(N, q());
    X_.leftCols(p) = Wg(); // matrix W: first column of X
    
    for ( int i = 0; i < L ; i++ ) { // j parte da p (coariate gruppo specifiche) e finisce a q*L (ogni volta aggiunge q covariate paziente specifico)
        X_.block( i*n_locs(), p+i*qV, n_locs(), qV ) = Vp().middleRows(i*n_locs(), n_locs()); // matrix V: block of matrices V1,V2,...,Vn (n: numero di pazienti))
    }
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
       
    	
        // compute q x q dense matrix X^\top*W*X and its factorization
        if (has_covariates() && (df_.is_dirty(DESIGN_MATRIX_BLK) || df_.is_dirty(WEIGHTS_BLK))) {
            XtWX_ = X().transpose() * W_ * X(); 
            invXtWX_ = XtWX_.partialPivLu(); 
        }
        

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
    
    I_.resize(L,L);
    I_.setIdentity();
    
    mPsi_ = mPsi();
    
    mPsiTD_ = mPsiTD();
   

    if (runtime().query(runtime_status::is_lambda_changed)) {
        
        A_ = SparseBlockMatrix<double, 2, 2>(
                -mPsiTD()  * W() * mPsi(), lambda_D() * R1().transpose(), // 
                 lambda_D() * R1(),        lambda_D() * R0()            );
        invA_.compute(A_);

        // prepare rhs of linear system 
        b_.resize(A_.rows());
        
        for( int i=0; i < L; ++i){
        	b_.block((L+i)*n_basis(), 0, n_basis(), 1) = lambda_D() * u(); // !!!! n_basis() * L
        }
        return;
    }
    if (runtime().query(runtime_status::require_W_update)) {
        // adjust north-west block of matrix A_ only
        A_.block(0, 0) = -mPsiTD() * mPsi(); // W() problemi . 
        invA_.compute(A_);
        return;
    }
    }

    void solve() {
        fdapde_assert(y().rows() != 0);
        DVector<double> sol;

        // parametric case
        // update rhs of SR-PDE linear system

        b_.block(0, 0, L*n_basis(), 1) = -mPsiTD() * lmbQ(y());   // -\Psi^T*D*Q*z
        // matrices U and V for application of woodbury formula
        U_ = DMatrix<double>::Zero(2 * L * n_basis(), q());
        U_.block(0, 0, L* n_basis(), q()) = mPsiTD_  * X(); // * W() * X();
        V_ = DMatrix<double>::Zero(q(), 2 * L * n_basis());
        V_.block(0, 0, q(), L * n_basis()) = X().transpose() * mPsi(); // W() * mPsi()
        // solve system (A_ + U_*(X^T*W_*X)*V_)x = b using woodbury formula from linear_algebra module
        sol = SMW<>().solve(invA_, U_, XtWX(), V_, b_);
        // store result of smoothing
        f_ = sol.head(L*n_basis());
        beta_ = invXtWX().solve(X().transpose() ) * (y() - mPsi_ * f_); // X().transpose() * W()
        // store PDE misfit
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
}; // monolithic

template <>
class MixedSRPDE<SpaceOnly,iterative> : public RegressionBase<MixedSRPDE<SpaceOnly,iterative>, SpaceOnly> {
   private:
    // typedef RegressionBase<MixedSRPDE, SpaceOnly> Base;
    SparseBlockMatrix<double, 2, 2> A_ {};         // system matrix of non-parametric problem (2N x 2N matrix)
    fdapde::SparseLU<SpMatrix<double>> invA_ {};   // factorization of matrix A
    DVector<double> b_ {};                         // right hand side of problem's linear system (1 x 2N vector)

    // construction of the functional minimized by the iterative scheme
    // ... eq.(4) file aldo


    // iterative scheme 
    double tol_ = 1e-4;             // tolerance (stopping criterion)
    std::size_t max_iter_ = 30;     // maximum number of iteration

   public:
    using RegularizationType = SpaceOnly
    using Base = RegressionBase<MixedSRPDE<RegularizationType, iterative>, RegularizationType>;
    IMPORT_REGRESSION_SYMBOLS;
    using Base::lambda_D;      // smoothing parameter in space
    using Base::n_basis;       // number of spatial basis
    using Base::runtime:       // runtime model status
    static constexpr int n_lambda = 1;

    // constructors
    MixedSRPDE() = default;
    MixedSRPDE(const pde_ptr& pde, Sampling s) : Base(pde, s) { };

    // check: if SpaceOnly tensorizes \Psi matrix -> if yes uncomment the following line
    // void tensorize_psi() { return; }   // avoid tensorization of \Psi matrix
    void init_regularization(){
        pde_.init();
        s_ = pde_.initial_condition();
        u_ = pde_.force()   // forcing term
        // check: I'm not sure about this -> see what is done in SpaceOnly
    }

    void init_model() { return; };// ?

    // the functional minimized by the iterative scheme
    // J(f,g) =  ...?
    double J(const DMatrix<double>& f, const DMatrix<double>& g) const {
        
        // SSE = \sum_{k=1}^m (z^k - \Psi*f^k)^T*(z^k - \Psi*f^k)
        
        return SSE + lambda_D() * g.squaredNorm();
    }
    // internal solve routine used by the iterative method
    void solve(BlockVector<double>& f_new, BlockVector<double>& g_new) const {
        DVector<double> x = invA_.solve(b_);
        f_new = x.topRows(n_spatial_basis());
        g_new = x.bottomRows(n_spatial_basis());
        // f prende le prime n righe della soluzione x di Ax=b
        // g prende le ultime n righe della soluzione 
        // è quello che vogliamo ? [controllare la dimensionalità]
        return;
    }

    // internal utilities -- non sono sicura che servano
    DMatrix<double> y(std::size_t k) const { return y().block(n_spatial_locs() * k, 0, n_spatial_locs(), 1); }
    DMatrix<double> u(std::size_t k) const { return u_.block(n_basis() * k, 0, n_basis(), 1); }

    // functional minimized by the iterative scheme
    // J(f)=norm(y-X\nu-f_n)+\lambda\sum_{i=1}^m norm(\nabla f_i)
    double J(const DMatrix<double>& f) const{
    // non mi ricordo cosa mi avevi spiegato sul laplaciano
    double term = 0; // this is the sum of the L2-squared norms of the Laplace-Beltrami operator

    for (std::size_t i = 1; i <= n_locs; i++) {
        // non sono sicura di n_locs, i = 1,...,m cicla sul numero di statistical units
        term += ;
    }
    return (norm(y()-X_*nu-f)+lambda*term) // lambda_S?
    // here I'm using X_ constructed as in the monolithic model: need to copy
    // ritorna norm()
    // qui  va completato definendo tutte le parti: nu, X, lambda
    }

    // internal solve -> useful for the iterations
    void solve(std::size_t t, BlockVector<double>& f_new, BlockVector<double>& g_new) const {
        DVector<double> x = invA_.solve(b_);
        f_new(t) = x.topRows(n_spatial_basis());
        g_new(t) = x.bottomRows(n_spatial_basis());
        return;
    } // come è definito g? da controllare questo

    void solve() { 
        fdapde_assert(y().rows() != 0); // what is this?

        // compute the starting point for iterative minimization of functional 
        // should it be something like x^0 = (f_hat^0, g^0) ...?

        // eq.(6) file aldo:
        A_ = SparseBlockMatrix<double,2,2>(
            -mPsiTD()*W()*mPsi(), lambda_D()*R1().transpose(),
            lambda_D()*R1(), lambda_D()*R0()
        ); // domanda: W() sarebbe definito come Q? cioè I-X(X^T*X)^{-1}*X^T
        invA_.compute(A_); 
        b_.resize(A_.rows());

        // the solution is in the form x^{k+1} = (f_hat^{k+1}, g^{k+1})^T
        // cosa cristo è z ?

        // qui dovremmo calcolare f e g

        
        // initialize the functional to minimize...


        // Note:
        // - f_ e g_ sono già definite in model_macros.h, vanno aggiornate a ogni iterazione
        // - serve popolare la design matrix X (?)
        // - importante: inizializzare x0 = (f_hat0, g0)
        

        std::size_t i = 1;   // iteration number

        // iterative scheme for minimization of functional 
        while (i < max_iter_ && std::abs(r_new) > tol_ /*qual è la condizione di stop ??*/) {
            
            // r_new = b_ - A_*x_old;

            
            // r_old = r_new;
            i++;
        }

        return;
    }

    // GCV support
    double norm(const DMatrix<double>& op1, const DMatrix<double>& op2) const { return (op1 - op2).squaredNorm(); }

    // setters
    void set_tolerance(double tol) { tol_ = tol; }
    void set_max_iter(std::size_t max_iter) { max_iter_ = max_iter; }

    // getters
    const DiagMatrix<double>& W() const { return W_; }
    const DMatrix<double>& X() const { return X_; }  
    const DMatrix<double>& Wg() const { return df_.template get<double>(DESIGN_MATRIX_BLK); } 
    const DMatrix<double>& Vp() const { return df_.template get<double>(MIXED_EFFECTS_BLK); } 
    const SpMatrix<double> mPsi() const { return Kronecker(I_, Psi()); }
    const SpMatrix<double> mPsiTD() const { return Kronecker(I_, PsiTD(not_nan())); }
    const SpMatrix<double> R0() const { return Kronecker(I_, pde_.mass()); }
    const SpMatrix<double> R1() const { return Kronecker(I_, pde_.stiff()); }
    // n_basis() ??? n_basis ???? in the monolithic model as well: we defined n_basis but we never use it

    virtual ~MixedSRPDE() = default;

}; // iterative


}   // namespace models
}   // namespace fdapde

#endif   // __MIXED_SRPDE_H__
