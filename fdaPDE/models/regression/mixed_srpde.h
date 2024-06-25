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
#include "../model_traits.h"
#include "../sampling_design.h"
#include "regression_base.h"
using fdapde::core::SMW;
using fdapde::core::SparseBlockMatrix;
using fdapde::core::pde_ptr;
using fdapde::core::Kronecker;
using fdapde::core::BlockVector;

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

    DMatrix<double> X_ {};      // dimensione: N osservazioni totali * p covariate gruppo specifiche
	
    int N;          // N: total observations (N=n*L)
    int n;          // n: observations for each statistical unit (patient)
    int L;          // L: number of patients
    int qV;         // qV: patient-specific covariatess
    int p;          // p: group-specific covariates
    
    SpMatrix<double> I_;   // N x N sparse identity matrix 
    
    SpMatrix<double> mPsi_;
    SpMatrix<double> mPsiTD_;
    
    void init_X() {

        N = Wg().rows();        // N: total observations (N=n*L)
        n = n_locs();           // n: observations for each statistical unit (patient)
        L = N/n;                // L: number of patients
        qV = Vp().cols();       // qV: patient-specific covariatess
        p = Wg().cols();        // p: group-specific covariates
        
        X_ = DMatrix<double>::Zero(N, q()); // q = p+L*qV
        X_.leftCols(p) = Wg(); // matrix W: first column of X
        
        for ( int i = 0; i < L ; i++ ) { // cycle over the number of patients that adds q patient-specific covariates to X_ at each loop
            X_.block( i*n_locs(), p+i*qV, n_locs(), qV ) = Vp().middleRows(i*n_locs(), n_locs()); // matrix V: diagonal block matrix with V1,V2,...,VL on the diagonal
        }

    }
   
   public:
    IMPORT_REGRESSION_SYMBOLS;

    using Base::lambda_D;   // smoothing parameter in space
    using Base::n_basis;    // number of spatial basis
    using Base::runtime;    // runtime model status
    using RegularizationType = SpaceOnly;
    static constexpr int n_lambda = 1;

    // constructors
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

        // I_ is a NxN sparse identity matrix
        I_.resize(L,L);
        I_.setIdentity();
        
        // Kronecker products with the identity
        mPsi_ = mPsi();    
        mPsiTD_ = mPsiTD();
    
        if (runtime().query(runtime_status::is_lambda_changed)) {
            
            A_ = SparseBlockMatrix<double, 2, 2>(
                    -mPsiTD()  * W() * mPsi(), lambda_D() * R1().transpose(),
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
    std::size_t q() const { return p+L*qV; }
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
    DMatrix<double> X_ {};                         // design matrix
    SpMatrix<double> Q_ {};                        // Q = I - X(X^T X)^{-1}X^T
    int N;                                         // N: total observations (N=n*L)
    int n_;                                        // n: observations for each statistical unit (patient)
    int L;                                         // L: number of patients
    int qV;                                        // qV: patient-specific covariatess
    int p;                                         // p: group-specific covariates
    int alpha_ = 1;                                // alpha: acceleration parameter
    SparseBlockMatrix<double, 2, 2> P_ {};         // preconditioning matrix of the Richardson scheme
    fdapde::SparseLU<SpMatrix<double>> invP_ {};   // factorization of matrix P
    SpMatrix<double> Gamma_ {};                    // approximation of the north-east block of the matrix A
    SpMatrix<double> I_;                           // N x N sparse identity matrix 
    DMatrix<double> s_;     // N x 1 initial condition vector
    DMatrix<double> u_;     // discretized forcing [1/DeltaT * (u_1 + R_0*s) \ldots u_n]
    SpMatrix<double> mPsi_;
    SpMatrix<double> mPsiTD_;

    // construction of the design matrix X_
    void init_X() {

        N = Wg().rows();          // N: total observations (N=n*L)
        n_ = n_locs();            // n: observations for each statistical unit (patient)
        L = N/n_;                 // L: number of patients
        qV = Vp().cols();         // qV: patient-specific covariatess
        p = Wg().cols();          // p: group-specific covariates

        // I_ is a NxN sparse identity matrix
        I_.resize(L,L);
        I_.setIdentity();

        // Kronecker products with the identity
        mPsi_ = mPsi();
        mPsiTD_ = mPsiTD();
        
        X_ = DMatrix<double>::Zero(N, q());
        X_.leftCols(p) = Wg(); // matrix W: first column of X
        for ( int i = 0; i < L ; i++ ) { // cycle over the number of patients that adds q patient-specific covariates to X_ at each loop
            X_.block( i*n_locs(), p+i*qV, n_locs(), qV ) = Vp().middleRows(i*n_locs(), n_locs()); // matrix V: diagonal block matrix with V1,V2,...,VL on the diagonal
        }
    }

    // construction of Gamma (approxiamtion of \Psi^T Q \Psi)
    void init_Gamma() {
        DMatrix<double> Qconstr;
        Qconstr.resize(N,N);
        for(std::size_t i = 0; i < L; i++){
            DMatrix<double> I = {};
            I.resize(n(i),n(i));
            I.setIdentity();
            DMatrix<double> XtWXi = X().transpose() * X();   // qui sarebbe X().transpose() * W() * X() // ma bisogna controllare le dimensioni di W
            DMatrix<double> invXtWXi = XtWXi.inverse(); 
            DMatrix<double> XiT = X(i).transpose();
            DMatrix<double> Qi = I - X(i)*invXtWXi*XiT;       // dimension of Qi: n*n
            
            int sum_n = 0;
            for (std::size_t k = 0; k < i; k++){
                sum_n += n(k);
            }

            Qconstr.block(sum_n,sum_n,n(i),n(i)) = Qi; // oppure Qi.sparseView();         // dimension of Q: N*N
        }
        
        Q_ = Qconstr.sparseView();
        Gamma_ = -mPsiTD_*Q()*mPsi_;   // dimension of Gamma: NL*NL
    }

    // mask for Q
    SpMatrix<double> Q(std::size_t i) const { 
        int sum_n = 0;
        for (std::size_t k = 0; k < i; k++){
            sum_n += n(k);
        }
        return Q().block(sum_n,sum_n,n(i),n(i));
    }; 

    // construction of the preconditioning matrix P_
    void init_P() {
        P_ = SparseBlockMatrix<double, 2, 2>(
            Gamma(),                  lambda_D() * R1().transpose(),
            lambda_D() * R1(),        lambda_D() * R0()                 );
    }

    // iterative scheme 
    double tol_ = 1e-4;             // tolerance (stopping criterion)
    std::size_t max_iter_ = 30;     // maximum number of iteration

   public:
    using RegularizationType = SpaceOnly;
    using Base = RegressionBase<MixedSRPDE<RegularizationType, iterative>, RegularizationType>;
    IMPORT_REGRESSION_SYMBOLS;
    using Base::lambda_D;      // smoothing parameter in space
    using Base::n_basis;       // number of spatial basis
    using Base::runtime;       // runtime model status
    using Base::pde_;              // parabolic differential operator df/dt + Lf - u
    static constexpr int n_lambda = 1;

    // constructors
    MixedSRPDE() = default;
    MixedSRPDE(const pde_ptr& pde, Sampling s) : Base(pde, s) { pde_.init(); };

    // check: if SpaceOnly tensorizes \Psi matrix -> if yes uncomment the following line
    // void tensorize_psi() { return; }   // avoid tensorization of \Psi matrix
    void init_regularization(){
        pde_.init();
        s_ = pde_.initial_condition();
        u_ = pde_.force();   // forcing term... questo è definito in (3.5) thesis Ischia pag. 26
        // u = \Psi^T*Q*z        // check: I'm not sure about this 
    }

    void analyze_data() { 

        // build multi-domain model design matrix 
        // std::cout << "- INIT X" << std::endl;
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

        // build Gamma
        init_Gamma();

        // build preconditioning matrix 
        init_P();
        invP_.compute(P_);

        // build A_ and initialize b_   
        A_ = SparseBlockMatrix<double,2,2>(
            -mPsiTD()*W()*mPsi(), lambda_D()*R1().transpose(),
            lambda_D()*R1(), lambda_D()*R0()
        ); 
        invA_.compute(A_); 
        b_.resize(A_.rows());
        b_ = DMatrix<double>::Zero(A_.rows(), 1);

        return; 
    }

    // internal utilities
    int n(std::size_t k) const { return k == 0 ? 0 : n_; } // at this point n is assumed to be the same for everyone (?)
    int n_basis(std::size_t k) const { return k == 0 ? 0 : n_basis(); }
    DMatrix<double> y(std::size_t k) const { return y().block(n_spatial_locs() * k, 0, n_spatial_locs(), 1); }
    double alpha(std::size_t k) const { return alpha_; } // fixed to 1 

    // mask for u
    DMatrix<double> u(std::size_t k) const { 
        double init = 0;
        for (std::size_t l = 0; l <= k-1; l++){
            init += n(l);
        }
        double end = 0;
        for (std::size_t l = 0; l <= k; l++){
            end += n(l);
        }
        end -= 1; // controlare
        return u_.block(init, 0, end-init, 1); 
    };

    // mask for X
    DMatrix<double> X(std::size_t i) const { 
        int sum_n = 0;
        for (std::size_t k = 0; k < i; k++){
            sum_n += n(k);
        }
        return X().block(sum_n, 0, n(i), X_.cols());

    }; // reference: pag.26 thesis Ischia 

    // mask for Psi
    SpMatrix<double> mPsi(std::size_t i) const { 
        int sum_n = 0;
        int sum_loc = 0;
        for (std::size_t k = 0; k < i; k++){
            sum_n += n(k);
        }
        for (std::size_t k = 0; k < i; k++){
            sum_loc += n_basis(k);
        }
        return mPsi_.block(sum_n, sum_loc, n(i), n_basis(i));
    }

    // mask for R0
    SpMatrix<double> R0(std::size_t i) const { 
        int sum_loc = 0;
        for (std::size_t k = 0; k < i; k++){
            sum_loc += n_basis(k);
        }
        return R0().block(sum_loc, sum_loc, n_basis(i), n_basis(i));
    }

    // mask for R1
    SpMatrix<double> R1(std::size_t i) const { 
        int sum_loc = 0;
        for (std::size_t k = 0; k < i; k++){
            sum_loc += n_basis(k);
        }
        return R1().block(sum_loc, sum_loc, n_basis(i), n_basis(i));
    }

    // functional minimized by the iterative scheme:  J(f)=norm(y-X\nu-f_n)+\lambda\sum_{i=1}^m norm(\nabla f_i)
    double J(const DMatrix<double>& f, const DMatrix<double>& g) const{

        DMatrix<double> fhat = mPsi_ * f; 
        DMatrix<double> nu = invXtWX().solve(X().transpose()*(y() - fhat));
        DMatrix<double> term2 = g.transpose()*R0()*g; // this should be done "component-wise" (for each statistical unit) - pag.25 ischia

        return (y() - X() * nu - fhat).squaredNorm() + lambda_D()*term2.squaredNorm();
    }


    void solve() { 

        fdapde_assert(y().rows() != 0); // what is this?

        DVector<double> x_new;
        x_new = DMatrix<double>::Zero(A_.rows(), 1);
        DVector<double> x_old;  
        DVector<double> r_new; 
        DVector<double> r_old; 
        double Jnew;
        double Jold;
        
        // internal utilities (for initialization)
        SparseBlockMatrix<double, 2, 2> K {};         // notazione da rivedere
        fdapde::SparseLU<SpMatrix<double>> invK {};   
        DVector<double> bi;
        DVector<double> xi0; 
        DVector<double> f0; 
        DVector<double> g0; 
        f0 = DMatrix<double>::Zero(L*n_basis(), 1); // qui L*n_basis sarebbe la somma di n_basis
        g0 = DMatrix<double>::Zero(L*n_basis(), 1);

        for (std::size_t i = 1; i <= L; i++){

            K = SparseBlockMatrix<double, 2, 2>(
            -mPsi(i).transpose()*Q(i)*mPsi(i),      lambda_D() * R1(i).transpose(),
            lambda_D() * R1(i),                     lambda_D() * R0(i)                 );
            SpMatrix<double> obj = -mPsi(i).transpose()*Q(i)*mPsi(i);
            invK.compute(K); 
            bi = DMatrix<double>::Zero(K.rows(), 1);
            bi.block(0,0,n_basis(i),1) = -mPsi(i).transpose() * Q(i) * y(i-1);

            xi0 = invK.solve(bi);
        
            f0.block((i-1)*n_basis(i),0,n_basis(i),1) = xi0.head(n_basis(i)); // f0_i
            g0.block((i-1)*n_basis(i),0,n_basis(i),1) = xi0.tail(n_basis(i)); // g0_i
            // se si cambia n_basis poi bisogna cambiare questa roba con un ciclo for che tiene conto dei diversi n_basis

            b_.block((i-1)*n_basis(i), 0, bi.rows(), 1) = bi;
        }
        
        x_new.block(0,0,f0.rows(),1) = f0;
        x_new.block(f0.rows(),0,g0.rows(),1) = g0;
    
        // store result of smoothing
        f_ = x_new.head(L*n_basis()); // f0        
        g_ = x_new.tail(L*n_basis()); // g0

        // computation of r^{0} = b - Ax^{0} (residual at k=0)
        r_old = b_ - A_*x_new;

        // computation of z^{1} as solution of the linear system Pz^{1} = r^{0}
        // questo forse non serve, tanto lo fa all'inizio del loop
        DVector<double> z;
        z = invP_.solve(r_old); 

        // u_ = mPsiTD()*Q()*z; // forse?
        
        // Nota: f_ e g_ sono già definite in model_macros.h, vanno aggiornate a ogni iterazione

        x_old = x_new;

        double c = 2;

        r_new = r_old + DVector<double>::Ones(r_old.rows())*c; // in this way I can enter the loop the first time... maybe there is a better idea
        Jnew = J(f_,g_);
        Jold = Jnew + c;
        
        // iteration loop
        std::size_t k = 1;   // iteration number    
        bool rcheck = 1;    

        // iterative scheme for minimization of functional 
        while (k < max_iter_ && rcheck && std::abs((Jnew-Jold)/Jnew) > tol_) /* ischia pag 25 */ {

            z = invP_.solve(r_old);  // mi serve solo z all'iterazione corrente... giusto?

            x_new = x_old + alpha(k)*z;
            f_ = x_new.topRows(L*n_basis());
            g_ = x_new.bottomRows(L*n_basis());
            Jold = Jnew;
            Jnew = J(f_,g_);
            
            r_new = r_old - alpha(k)*A_*z;

            rcheck = ((r_new-r_old).squaredNorm() > tol_);
            // rcheck = r_new.norm() / b_.norm() > tol_;

            std::cout << "Iteration n." << k << std::endl;
            std::cout << "r:" << (r_new-r_old).squaredNorm() << std::endl;
            std::cout << "J:" << std::abs((Jnew-Jold)/Jnew)  << std::endl;
            x_old = x_new;
            r_old = r_new;
            k++;
        }

        return;
    }

    // GCV support
    double norm(const DMatrix<double>& op1, const DMatrix<double>& op2) const { return (op1 - op2).squaredNorm(); }

    // setters
    void set_tolerance(double tol) { tol_ = tol; }
    void set_max_iter(std::size_t max_iter) { max_iter_ = max_iter; }

    // getters
    const int n() const { return n_; }
    std::size_t q() const { return p+L*qV; }
    const DiagMatrix<double>& W() const { return W_; }
    const DMatrix<double>& X() const { return X_; }  
    const SpMatrix<double>& Q() const { return Q_; }  
    const SpMatrix<double> Gamma() const { return Gamma_; }  
    const DMatrix<double>& Wg() const { return df_.template get<double>(DESIGN_MATRIX_BLK); } 
    const DMatrix<double>& Vp() const { return df_.template get<double>(MIXED_EFFECTS_BLK); } 
    const SpMatrix<double> mPsi() const { return Kronecker(I_, Psi(not_nan())); }
    const SpMatrix<double> mPsiTD() const { return Kronecker(I_, PsiTD(not_nan())); }
    const SpMatrix<double> R0() const { return Kronecker(I_, pde_.mass()); }
    const SpMatrix<double> R1() const { return Kronecker(I_, pde_.stiff()); }

    virtual ~MixedSRPDE() = default;

}; // iterative 

};   // namespace models
}   // namespace fdapde

#endif   // __MIXED_SRPDE_H__
