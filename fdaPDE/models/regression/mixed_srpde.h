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
#include <iostream>
#include <fstream>

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
	
    int N;          // N: total observations (N=n*m)
    int n;          // n: observations for each statistical unit (patient)
    int m_;          // m: number of patients
    int qV;         // qV: patient-specific covariatess
    int p;          // p: group-specific covariates
    
    SpMatrix<double> I_;   // N x N sparse identity matrix 
    
    SpMatrix<double> mPsi_;
    SpMatrix<double> mPsiTD_;

    DVector<double> alpha_coeff_;                   // coefficients
    DVector<double> beta_coeff_;                    // beta coefficients of the original model
    // here I redeclared the beta() getter because beta_ is used in the function J
    SparseBlockMatrix<double, 2, 2> F_ {};          // matrix used for linear transformation of coefficients
    DMatrix<double> T_;

    
    void init_X() {

        N = Wg().rows();        // N: total observations (N=n*m)
        n = n_locs();           // n: observations for each statistical unit (patient)
        m_ = N/n;                // m: number of patients
        qV = Vp().cols();       // qV: patient-specific covariatess
        p = Wg().cols();        // p: group-specific covariates
        
        X_ = DMatrix<double>::Zero(N, q()); // q = p+m*qV
        X_.leftCols(p) = Wg(); // matrix W: first column of X
        
        for ( int i = 0; i < m_ ; i++ ) { // cycle over the number of patients that adds q patient-specific covariates to X_ at each loop
            X_.block( i*n_locs(), p+i*qV, n_locs(), qV ) = Vp().middleRows(i*n_locs(), n_locs()); // matrix V: diagonal block matrix with V1,V2,...,Vm on the diagonal
        }

        // matrix for retrieve initial coefficients (alpha and beta)
        DMatrix<double> Ip = {};
        Ip.resize(p,p);
        Ip.setIdentity();
        Ip = Ip/m_;
        
        DMatrix<double> Iqp = {};
        Iqp.resize(qV,qV);
        Iqp.setIdentity();

        DMatrix<double> Z1 = DMatrix<double>::Zero(p, qV);

        DMatrix<double> Z2 = DMatrix<double>::Zero(qV, m_*p);

        DMatrix<double> Ip_loop = {};
        Ip_loop.resize(p, m_*p);

        for (std::size_t i=0; i<m_; i++){
            Ip_loop.block(0,i*p,p,p) = Ip;
        }

        F_ = SparseBlockMatrix<double,2,2>(
            Z1.sparseView(), Ip_loop.sparseView(),
            Iqp.sparseView(), Z2.sparseView()
        ); 

        // std::cout << "F_: "<< F_.rows() << "x" << F_.cols() << std::endl;

        T_.resize(m_*p,m_*p);
        T_.setIdentity();
        T_ = (m_-1.0)/m_ * T_;

        Ip.setIdentity();


        for (std::size_t i=0; i<m_; i++){
            for(std::size_t j=0; j<m_; j++){
                if(T_(i*p,j*p) == 0){
                    T_.block(i*p,j*p,p,p) = -Ip;
                }
            }
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
        std::cout << "METODO MONOLITICO" << std::endl;
        // I_ is a NxN sparse identity matrix
        I_.resize(m_,m_);
        I_.setIdentity();
        
        // Kronecker products with the identity
        mPsi_ = mPsi();    
        
    
        if (runtime().query(runtime_status::is_lambda_changed)) {
            
            auto start = std::chrono::high_resolution_clock::now();

            A_ = SparseBlockMatrix<double, 2, 2>(
                    -mPsiTD()  * W() * mPsi(), lambda_D() * R1().transpose(),
                    lambda_D() * R1(),        lambda_D() * R0()            );

            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> duration = end - start;
            std::cout << "- assemblamento della matrice di sistema A_: " << duration.count() << std::endl;

            start = std::chrono::high_resolution_clock::now();
            
            invA_.compute(A_);

            end = std::chrono::high_resolution_clock::now();
            duration = end - start;
            std::cout << "- inversione della matrice di sistema A_: " << duration.count() << std::endl;

            // prepare rhs of linear system 
            b_.resize(A_.rows());
            
            for( int i=0; i < m_; ++i){
                b_.block((m_+i)*n_basis(), 0, n_basis(), 1) = lambda_D() * u(); // !!!! n_basis() * m
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
        
        auto start = std::chrono::high_resolution_clock::now();
        
        fdapde_assert(y().rows() != 0);
        DVector<double> sol;

        // parametric case
        // update rhs of SR-PDE linear system

        b_.block(0, 0, m_*n_basis(), 1) = -mPsiTD() * lmbQ(y());   // -\Psi^T*D*Q*z

        // matrices U and V for application of woodbury formula
        U_ = DMatrix<double>::Zero(2 * m_ * n_basis(), q());
        U_.block(0, 0, m_* n_basis(), q()) = mPsiTD_  * X(); // * W() * X();
        V_ = DMatrix<double>::Zero(q(), 2 * m_ * n_basis());
        V_.block(0, 0, q(), m_ * n_basis()) = X().transpose() * mPsi(); // W() * mPsi()

        // solve system (A_ + U_*(X^T*W_*X)*V_)x = b using woodbury formula from linear_algebra module
        sol = SMW<>().solve(invA_, U_, XtWX(), V_, b_);

        // store result of smoothing
        f_ = sol.head(m_*n_basis());
        // beta_ = invXtWX().solve(X().transpose() ) * (y() - mPsi_ * f_); // X().transpose() * W()
        beta_ = invXtWX().solve(X().transpose() * (y() - mPsi_ * f_)); 

        beta_coeff_ = F_*beta_;
        alpha_coeff_ = T_*beta_.tail(m_*p); 

        // store PDE misfit
        g_ = sol.tail(m_*n_basis());

    
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end - start;
        std::cout << "- solve(): " << duration.count() << std::endl;

        return;
    }
    // GCV support
    double norm(const DMatrix<double>& op1, const DMatrix<double>& op2) const { return (op1 - op2).squaredNorm(); }

    // getters
    std::size_t q() const { return p+m_*qV; }
    const DiagMatrix<double>& W() const { return W_; }
    const DMatrix<double>& X() const { return X_; }  
    const DMatrix<double>& Wg() const { return df_.template get<double>(DESIGN_MATRIX_BLK); } 
    const DMatrix<double>& Vp() const { return df_.template get<double>(MIXED_EFFECTS_BLK); } 
    const SpMatrix<double> mPsi() const { return Kronecker(I_, Psi()); }
    const SpMatrix<double> mPsiTD() const { return Kronecker(I_, PsiTD(not_nan())); }
    const SpMatrix<double> R0() const { return Kronecker(I_, pde_.mass()); }
    const SpMatrix<double> R1() const { return Kronecker(I_, pde_.stiff()); }
    const DVector<double> alpha() const { return alpha_coeff_; }
    const DVector<double> beta() const { return beta_coeff_; }
    
    virtual ~MixedSRPDE() = default;

}; // monolithic


template <>
class MixedSRPDE<SpaceOnly,iterative> : public RegressionBase<MixedSRPDE<SpaceOnly,iterative>, SpaceOnly> {
   private:
    // typedef RegressionBase<MixedSRPDE, SpaceOnly> Base;
    // SparseBlockMatrix<double, 2, 2> A_ {};         // system matrix of non-parametric problem (2N x 2N matrix)
    // fdapde::SparseLU<SpMatrix<double>> invA_ {};   // factorization of matrix A
    DVector<double> b_ {};                         // right hand side of problem's linear system (1 x 2N vector)
    DMatrix<double> X_ {};                         // design matrix
    SpMatrix<double> Q_ {};                        // Q = I - X(X^T X)^{-1}X^T
    // std::vector<DMatrix<double>> _W;
    // std::vector<DMatrix<double>> _V;
    // DMatrix<double> _sumWt;
    int N;                                         // N: total observations (N=n*m)
    // int n_;                                        // n: observations for each statistical unit (patient)
    int m_;                                        // m: number of patients
    int qV;                                        // qV: patient-specific covariatess
    int p;                                        // p: group-specific covariates
    double alpha_ = 1;                             // alpha: acceleration parameter
    SparseBlockMatrix<double, 2, 2> P_ {};         // preconditioning matrix of the Richardson scheme
    fdapde::SparseLU<SpMatrix<double>> invP_ {};   // factorization of matrix P
    // SpMatrix<double> Gamma_ {};                    // approximation of the north-east block of the matrix A
    SpMatrix<double> I_;                           // N x N sparse identity matrix 
    DMatrix<double> s_;                            // N x 1 initial condition vector
    DMatrix<double> u_;                            // discretized forcing [1/DeltaT * (u_1 + R_0*s) \ldots u_n]
    DVector<double> alpha_coeff_;                   // coefficients
    DVector<double> beta_coeff_;                    // beta coefficients of the original model
    // here I redeclared the beta() getter because beta_ is used in the function J
    SparseBlockMatrix<double, 2, 2> F_ {};          // matrix used for linear transformation of coefficients
    DMatrix<double> T_;
    std::vector<BlockFrame<double, int>> data_;       // vector of dataframes 
    std::vector<SpMatrix<double>> Psi_;                // override of Psi_
    std::vector<SpMatrix<double>> PsiTD_;              // override of Psi_
    SpMatrix<double> mPsi_;
    SpMatrix<double> mPsiTD_;
    DMatrix<double> y_;                         // vector of observations

    // construction of the design matrix X_
    void init_X() {
        std::cout << "METODO ITERATIVO" << std::endl;
        
        // n_ = n_locs(i);            // n: observations for each statistical unit (patient)
        m_ = data_.size();                                             // m_: number of patients
        N = n_locs();                                                  // N: total observations
        qV = data_[0].template get<double>(V_BLOCK).cols();         // qV: patient-specific covariatess
        p = data_[0].template get<double>(W_BLOCK).cols();          // p: group-specific covariates

        mPsi_ = mPsi();
        mPsiTD_ = mPsi_.transpose();

        y_ = y();
        
        X_ = DMatrix<double>::Zero(N, q());

        std::size_t sum_locs = 0;
        for ( int i = 0; i < m_ ; i++ ) { // cycle over the number of patients that adds q patient-specific covariates to X_ at each loop
            
            X_.block( sum_locs, 0, n_locs(i), p) = Wg(i);
            X_.block( sum_locs, p+i*qV, n_locs(i), qV ) = Vp(i); // matrix V: diagonal block matrix with V1,V2,...,Vm on the diagonal
            sum_locs += n_locs(i);
            
        }
        
    }

    // function for Q
    // SpMatrix<double> Q(std::size_t i) const { 

    //     DMatrix<double> I = {};
    //     I.resize(n_locs(i),n_locs(i));
    //     I.setIdentity();
        
    //     // le seguenti due righe potrebbero essere sostituite da invXtWX() già implementato ma devo capire come si usa

    //     DMatrix<double> XtWX = X().transpose() * X();   // qui sarebbe X().transpose() * W() * X() // ma bisogna controllare le dimensioni di W
    //     DMatrix<double> invXtWX = XtWX.inverse(); 

    //     DMatrix<double> Qi = I - X(i)*invXtWX*(X(i).transpose());
    //     return Qi.sparseView();

    // }; 
    
    // DMatrix<double> lmbH(const DMatrix<double>& x) const{
    // 	DMatrix<double> v = X().transpose() * x;   // X_i^\top*x
    //    	DMatrix<double> z = invXtWX().solve(v);          // (X^\top*X)^{-1}*X_i^\top*x
    //    // compute  W*X*z = (W*X*(X^\top*W*X)^{-1}*X^\top*W)*x = W(H)*x 
    //    return X() * z;
    // }
    
    // DMatrix<double> lmbH(const DMatrix<double>& x, std::size_t i) const{
    // 	DMatrix<double> v = X(i).transpose() * x;   // X_i^\top*x
    //    	DMatrix<double> z = invXtWX().solve(v);          // (X^\top*X)^{-1}*X_i^\top*x
    //    // compute  W*X*z = (W*X*(X^\top*W*X)^{-1}*X^\top*W)*x = W(H)*x 
    //    return X(i) * z;
    // }
    
    
    DMatrix<double> aux(const DMatrix<double>& x, std::size_t i) const{
    	//DMatrix<double> v = (mPsi().transpose()*X()).transpose() * x.topRows(n_basis()*m_);   
       	DMatrix<double> v = DMatrix<double>::Zero(N,1);
        
        int sum = 0;
       	for(std::size_t k = 0; k < m_; ++k){
       		v.block(sum, 0, n_locs(i), 1) = Psi_[k]*x.block(k*n_basis(), 0, n_basis(),1);	
            sum += n_locs(i);
       	}

       	DMatrix<double> u = DMatrix<double>::Zero(q(),1);
       	
       	//u.block(0, 0, p, 1) = _sumWt * v;

        sum = 0;
       	for(std::size_t k = 0; k < m_; ++k){
       		u.block(0, 0, p, 1) += Wg(k).transpose()*v.block(sum, 0, n_locs(k),1);	
       		u.block(k*qV+p, 0, qV, 1) = Vp(k).transpose()*v.block(sum, 0, n_locs(k), 1);	
            sum += n_locs(k);
       	}
       
       	DMatrix<double> z = invXtWX().solve(u);         
       	
		DMatrix<double> w = Wg(i)*z.block(0, 0, p, 1) + Vp(i)*z.block(i*qV+p, 0, qV, 1);
       	
       	//DMatrix<double> U_i = (mPsi().transpose()*X()).block(i*n_basis(), 0, n_basis(), q()); 
        //DMatrix<double> res = DMatrix<double>::Zero(n_basis(),1);
        
        //return U_i * z;
        std::cout << (Psi_[i].transpose()*w).rows() << std::endl;

        return Psi_[i].transpose()*w; // ????? 
    }
    
    
    // lmbMonolithic ???
    
    // lmbPsiT ???
    
    
    // iterative scheme 
    double tol_ = 1e-4;             // tolerance (stopping criterion)
    double tol_res = 1e-8;	
    std::size_t max_iter_ = 10;     // maximum number of iteration

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
        std::cout << "init_regul" << std::endl;
        pde_.init();
        s_ = pde_.initial_condition();
        u_ = pde_.force();   // forcing term... questo è definito in (3.5) thesis Ischia pag. 26
        // u = \Psi^T*Q*z        // check: I'm not sure about this 
    }

    void analyze_data() { 
        std::cout << "analyze_data" << std::endl;

        // build multi-domain model design matrix 
    	init_X();
        
        // initialize empty masks
        if (!y_mask_.size()) y_mask_.resize(N);
        if (!nan_mask_.size()) nan_mask_.resize(N);

        // compute q x q dense matrix X^\top*W*X and its factorization
        // if (has_weights() && df_.is_dirty(WEIGHTS_BLK)) {
        //     W_ = df_.template get<double>(WEIGHTS_BLK).col(0).asDiagonal(); // !!! da modificare !!!
        //     model().runtime().set(runtime_status::require_W_update);
        // } else if (is_empty(W_)) {
            W_ = DVector<double>::Ones(N).asDiagonal(); // W_ in R^{N times N}
        // }
    	
        // compute q x q dense matrix X^\top*W*X and its factorization
        // if (has_covariates() && (df_.is_dirty(DESIGN_MATRIX_BLK) || df_.is_dirty(WEIGHTS_BLK))) {
            XtWX_ = X().transpose() * W_ * X(); // Da valutare
            invXtWX_ = XtWX_.partialPivLu(); 
        // }

        // derive missingness pattern from observations vector (if changed)
        // if (df_.is_dirty(OBSERVATIONS_BLK)) {
        //     n_nan_ = 0;
        //     for (std::size_t i = 0; i < df_.template get<double>(OBSERVATIONS_BLK).size(); ++i) {
        //         if (std::isnan(y()(i, 0))) {   // requires -ffast-math compiler flag to be disabled
        //             nan_mask_.set(i);
        //             n_nan_++;
        //             df_.template get<double>(OBSERVATIONS_BLK)(i, 0) = 0.0;   // zero out NaN
        //         }
        //     }
        //     if (has_nan()) model().runtime().set(runtime_status::require_psi_correction);
        // }
        return;
    }

    void init_model() { 

        auto start = std::chrono::high_resolution_clock::now();

        b_ = DMatrix<double>::Zero(2*n_basis()*m_, 1);

        // matrix for retrieve initial coefficients (alpha and beta)
        DMatrix<double> Ip = {};
        Ip.resize(p,p);
        Ip.setIdentity();
        Ip = Ip/m_;
        
        DMatrix<double> Iqp = {};
        Iqp.resize(qV,qV);
        Iqp.setIdentity();

        DMatrix<double> Z1 = DMatrix<double>::Zero(p, qV);

        DMatrix<double> Z2 = DMatrix<double>::Zero(qV, m_*p);

        DMatrix<double> Ip_loop = {};
        Ip_loop.resize(p, m_*p);

        for (std::size_t i=0; i<m_; i++){
            Ip_loop.block(0,i*p,p,p) = Ip;
        }

        F_ = SparseBlockMatrix<double,2,2>(
            Z1.sparseView(), Ip_loop.sparseView(),
            Iqp.sparseView(), Z2.sparseView()
        ); 

        T_.resize(m_*p,m_*p);
        T_.setIdentity();
        T_ = (m_-1.0)/m_ * T_;

        Ip.setIdentity();

        for (std::size_t i=0; i<m_; i++){
            for(std::size_t j=0; j<m_; j++){
                if(T_(i*p,j*p) == 0){
                    T_.block(i*p,j*p,p,p) = -Ip;
                }
            }
        }
    
        std::chrono::duration<double> duration = std::chrono::high_resolution_clock::now() - start;
        std::cout << "- assemblamento delle matrici per i coefficienti beta: " << duration.count() << std::endl;
        return; 
    }

    // internal utilities
    // int n(std::size_t k) const { return n_; } // at this point n is assumed to be the same for everyone (?)
    // int n_basis() const { 
    //     int sum = 0;
    //     for(std::size_t i=0; i<m_; i++){
    //         sum += data_[i].template get<double>(LOCS_BLOCK).rows();
    //     }
    //     return sum;
    // }
    
    
    // int n_basis(std::size_t i) const { return data_[i].template get<double>(LOCS_BLOCK).rows(); }

    // DMatrix<double> y(std::size_t k) const { return y().block(n_spatial_locs() * k, 0, n_spatial_locs(), 1); }
    

    // mask for X
    // DMatrix<double> X(std::size_t i) const { // SE salvo X_i, V_i questa diventa inutile 
    //     int sum_n = 0;
    //     for (std::size_t k = 0; k < i; k++){
    //         sum_n += n_locs(k);
    //     }
    //     return X().block(sum_n, 0, n_locs(i), X_.cols());

    // }; // reference: pag.26 thesis Ischia 

    // functional minimized by the iterative scheme:  J(f)=norm(y-X\nu-f_n)+\lambda\sum_{i=1}^m norm(\nabla f_i)
    double J(const DMatrix<double>& f, const DMatrix<double>& g) const{

        DMatrix<double> fhat = mPsi_ * f; 
        // DMatrix<double> nu = invXtWX().solve(X().transpose()*(y() - fhat));
        
        return (y_ - X() * beta_ - fhat).squaredNorm() + lambda_D()*g.squaredNorm(); // SE salvo X_i, V_i mi basta fornire lmbX 
    }

    void solve() { 
        std::cout << "solve" << std::endl;
        auto start = std::chrono::high_resolution_clock::now();
        
        fdapde_assert(y_.rows() != 0); // what is this?

        DVector<double> x_new = DMatrix<double>::Zero(2*n_basis()*m_, 1); // queste dimensioni boh
        
        b_.block(0, 0, n_basis()*m_, 1) = -mPsiTD_ * lmbQ(y_); // questo dà problemi - controllare dimensioni
        
        // U_ = DMatrix<double>::Zero(2*m_*n_basis(), q());
        // V_ = DMatrix<double>::Zero(q(), 2*m_*n_basis());
        
        // U_.block(0, 0, m_* n_basis(), q()) = mPsiTD()  * X(); // * W() * X(); / SE salvo X_i, V_i POSSO implementare direttamente U_i ?!
        // V_.block(0, 0, q(), m_ * n_basis()) = X().transpose() * mPsi(); // / SE salvo X_i, V_i POSSO implementare direttamente V_i?!
        
        DVector<double> x_old = x_new; 
        DVector<double> r = b_;//DMatrix<double>::Zero(2*m_*n_basis(), 1); 
        double Jnew;
        double Jold;
        
        // internal utilities (for initialization)
        //SparseBlockMatrix<double, 2, 2> P {};           // this is P_i (i = 1, ..., m)
        //fdapde::SparseLU<SpMatrix<double>> invP {};   
        DVector<double> bi = DMatrix<double>::Zero(2*n_basis(), 1);
        DVector<double> xi0; 
        // DVector<double> f0; 
        // DVector<double> g0; 
        // f0 = DMatrix<double>::Zero(n_basis()*m_, 1); // qui m_*n_basis sarebbe la somma di n_basis
        // g0 = DMatrix<double>::Zero(n_basis()*m_, 1);
        
        // P R^{2 n_basis() x 2 n_basis()} NO COVARIATE
        /*
        auto _start = std::chrono::high_resolution_clock::now();
        P_ = SparseBlockMatrix<double, 2, 2>(
            -PsiTD_*Psi_,                     lambda_D() * pde_.stiff().transpose(),
            lambda_D() * pde_.stiff(),             lambda_D() * pde_.mass()                 );
        std::chrono::duration<double> _duration = std::chrono::high_resolution_clock::now() - _start;
        std::cout << "-		assemblamento P_: " << _duration.count() << std::endl;
        
        _start = std::chrono::high_resolution_clock::now();
        invP.compute(P_);   
        _duration = std::chrono::high_resolution_clock::now() - _start;
        std::cout << "-		inversione P_: " << _duration.count() << std::endl;
        */

        DMatrix<double> U_i = DMatrix<double>::Zero(2*n_basis(), q());
        DMatrix<double> V_i = DMatrix<double>::Zero(q(), 2*n_basis());
        
        auto _start = std::chrono::high_resolution_clock::now();   
        
        for (std::size_t i = 0; i < m_; i++){

            std::chrono::duration<double> duration = std::chrono::high_resolution_clock::now() - start;
            std::cout << "- assemblamento della matrice di sistema A_: " << duration.count() << std::endl;
    
            start = std::chrono::high_resolution_clock::now();
            P_ = SparseBlockMatrix<double, 2, 2>(
                -PsiTD_[i]*Psi_[i],                     lambda_D() * pde_.stiff().transpose(),
                lambda_D() * pde_.stiff(),             lambda_D() * pde_.mass()                 );
            duration = std::chrono::high_resolution_clock::now() - start;
            std::cout << "-		assemblamento P_: " << duration.count() << std::endl;
            
            start = std::chrono::high_resolution_clock::now();
            invP_.compute(P_);   
            duration = std::chrono::high_resolution_clock::now() - start;
            std::cout << "-		inversione P_: " << duration.count() << std::endl;

            bi.block(0,0,n_basis(),1) = b_.block(i*n_basis(), 0, n_basis(), 1);
            
            // U_i.block(0, 0, n_basis(i), q()) = U_.block(i*n_basis(i), 0, n_basis(i), q()); //GUARDA SU
            // V_i.block(0, 0, q(), n_basis(i)) = V_.block(0, i*n_basis(i), q(), n_basis(i)); // GUARDA SU
            
            U_i.block(0, p+i*qV, n_basis(), qV ) = PsiTD_[i]*Vp(i);
            U_i.block(0, 0, n_basis(), p) = PsiTD_[i]*Wg(i);

            V_i.block(0, 0, p, n_basis()) = Wg(i).transpose()*Psi_[i];
            V_i.block(p+i*qV, 0, qV, n_basis()) = Vp(i).transpose()*Psi_[i]; 

            // solve system (A_ + U_*(X^T*W_*X)*V_)x = b using woodbury formula from linear_algebra module
            xi0 = SMW<>().solve(invP_, U_i, XtWX(), V_i, bi);

            x_new.block(i*n_basis(), 0, n_basis(),1) = xi0.head(n_basis());
            x_new.block((i+m_)*n_basis(),0, n_basis(),1) = xi0.tail(n_basis()); 
           
            // computation of r^{0} = b - Ax^{0} (residual at k=0)
            // r.block(i*n_basis(i),0,bi.rows(),1) = bi - P_*xi0;
            
            /*
            r.block(i*n_basis(i),0,n_basis(i),1) = bi.block(0,0, n_basis(i),1) +  PsiTD_ * Psi_ * xi0.head(n_basis(i)) - 
                                                                                  PsiTD_*lmbH(Psi_*xi0.head(n_basis(i)),i) - 
                                                                                  lambda_D() * pde_.stiff().transpose()*xi0.tail(n_basis(i)); // Psi_i^\top * Q(i) * Psi_i * f_i
            r.block((i+m_)*n_basis(i),0,n_basis(i),1) = bi.block(n_basis(i),0, n_basis(i),1) - ( lambda_D() * pde_.stiff() * xi0.head(n_basis(i)) + 
                                                                                                lambda_D() * pde_.mass() * xi0.tail(n_basis(i)) ); 
            */
            
            r.block(n_basis()*i,0, n_basis(),1) -=   ((-PsiTD_[i]* Psi_[i]) * xi0.head(n_basis()) +
                                           			  lambda_D()*pde_.stiff().transpose()*xi0.tail(n_basis()));
  
            // correzione cov fuori
            r.block(n_basis()*(m_+i),0, n_basis(),1) -= (lambda_D()*pde_.stiff()*xi0.head(n_basis()) +
                                                          lambda_D()*pde_.mass()*xi0.tail(n_basis()) );
        
            // if cov ... RIGA SOTTO SBAGLIATA.                                        
    
         //r.block(i*n_basis(i),0, n_basis(i), 1) -= Psi_.transpose()*lmbH(Psi_*xi0.head(n_basis(i)), i); // correzione per cov
        }
		
        std::chrono::duration<double> _duration = std::chrono::high_resolution_clock::now() - _start;
        std::cout << "-		inizializzazione: " << _duration.count() << std::endl;
        
        
		// if cov AGGIORNAMNETO COV
		
		_start = std::chrono::high_resolution_clock::now();
        
		for(std::size_t i = 0; i < m_; i++){
			r.block(i*n_basis(),0, n_basis(), 1) -= aux(x_new, i); 
		}
	
		_duration = std::chrono::high_resolution_clock::now() - _start;
        std::cout << "-		residuo: " << _duration.count() << std::endl;
        
		
        // store result of smoothing
        f_ = x_new.head(m_*n_basis()); // f0        
        g_ = x_new.tail(m_*n_basis()); // g0

        beta_ = invXtWX().solve(X().transpose() * (y_ - mPsi_ * f_)); 
        
        beta_coeff_ = F_*beta_;
        alpha_coeff_ = T_*beta_.tail(m_*p); 
        
        // compute residual (k=0)
        //r = b_ - A_ * x_new; // senza cov
        //r.block(0, 0, m_*n_basis(), 1) -= mPsi().transpose()*lmbH(mPsi()*f_); // correzione per cov
        
        DVector<double> z;
        z = DMatrix<double>::Zero(r.rows(), 1);
        // u_ = mPsiTD()*Q()*z; // forse?

        // Nota: f_ e g_ sono già definite in model_macros.h, vanno aggiornate a ogni iterazione

        //x_old = x_new;

        double c = 2;

        Jnew = J(f_,g_);
        Jold = (1 + c) * Jnew;
        
        // iteration loop
        std::size_t k = 1;   // iteration number    
        bool rcheck = r.norm() / b_.norm() < tol_res; // stop by residual    
        bool Jcheck =  std::abs((Jnew-Jold)/Jnew) < tol_; // stop by J
        bool exit_ = Jcheck && rcheck;
      
        DVector<double> ri = DMatrix<double>::Zero(2*n_basis(),1);
        DVector<double> zi = DMatrix<double>::Zero(2*n_basis(),1);
        // std::cout << "J(f_,g_): " << Jnew  << std::endl;
        // std::cout << "r.norm(): " << r.norm()  << std::endl;
        // std::cout << "r.norm() / b_.norm(): " << r.norm() / b_.norm() << std::endl;
        _start = std::chrono::high_resolution_clock::now();
        // iterative scheme for minimization of functional 
        while (k < max_iter_ && !exit_) /* ischia pag 25 */ {
            
            auto __start = std::chrono::high_resolution_clock::now();
            
            for(std::size_t i = 0; i < m_; i++){
                    
                   // bi = DMatrix<double>::Zero(P_.rows(), 1);
                    
                    // valutare implementazione di lmbQ(yi)
                    bi.block(0,0,n_basis(),1) = r.block(i*n_basis(), 0, n_basis(), 1) ; 
                    bi.block(n_basis(), 0, n_basis(), 1) = r.block( i*n_basis() + n_basis(), 0, n_basis(), 1) ;
                    
                    //DMatrix<double> U_i = DMatrix<double>::Zero(2*n_basis(), q());
                    //DMatrix<double> V_i = DMatrix<double>::Zero(q(), 2*n_basis());
                    
                    // U_i.block(0, 0, n_basis(i), q()) = U_.block(i*n_basis(i), 0, n_basis(i), q()); // GUARDA SU - come prima
                    // V_i.block(0, 0, q(), n_basis(i)) = V_.block(0, i*n_basis(i), q(), n_basis(i)); // GUARDA SU - come prima
                    U_i.block(0, p+i*qV, n_basis(), qV ) = PsiTD_[i]*Vp(i);
                    U_i.block(0, 0, n_basis(), p) = PsiTD_[i]*Wg(i);

                    V_i.block(0, 0, p, n_basis()) =  Wg(i).transpose()*Psi_[i];
                    V_i.block(p+i*qV, 0, qV, n_basis()) = Vp(i).transpose()*Psi_[i]; 

                    // solve system (A_ + U_*(X^T*W_*X)*V_)x = b using woodbury formula from linear_algebra module
                    auto ___start = std::chrono::high_resolution_clock::now();
                    zi = SMW<>().solve(invP_, U_i, XtWX(), V_i, bi);
                    std::chrono::duration<double> ___duration = std::chrono::high_resolution_clock::now() - ___start;
        	        std::cout << "-			costo SMW/unita: " << 
        											___duration.count() << std::endl;
                    
                    x_new.block(n_basis(),0, n_basis(),1) += alpha(k)*zi.head(n_basis());  
                    x_new.block(n_basis()*(m_+i),0, n_basis(),1) += alpha(k)*zi.tail(n_basis());
                    
                    r.block(n_basis(),0, n_basis(),1) -=  alpha(k) * ((-PsiTD_[i]* Psi_[i]) * zi.head(n_basis()) +
                                                                         lambda_D()*pde_.stiff().transpose()*zi.tail(n_basis()));
                                                                         
                    // correzione cov fuori
                    
                    r.block(n_basis()*(m_+i),0, n_basis(),1) -= alpha(k)*(lambda_D()*pde_.stiff()*zi.head(n_basis()) +
                                                                            lambda_D()*pde_.mass()*zi.tail(n_basis()) );
                    
                    z.block(i*n_basis(),0, n_basis(i),1) = zi.head(n_basis());  
                    z.block(n_basis()*(m_+i),0, n_basis(),1) = zi.tail(n_basis()); 
                	// prova RIGA SOTTO SBAGLIATA
                	//r.block((i+m_)*n_basis(i),0, n_basis(i), 1) -= alpha(k) * Psi_.transpose()*lmbH(Psi_*zi.head(n_basis(i)), i);
                    
            }
            std::chrono::duration<double> __duration = std::chrono::high_resolution_clock::now() - __start;
        	std::cout << "-		costo singola iter: " << __duration.count() << std::endl;
		
            // vecchia libreria ???
            
            /*
            for(std::size_t i = 0; i < m_; ++i){
                r.block(i*n_basis(i),0, n_basis(i),1) = b_.block(i*n_basis(i),0, n_basis(i),1) - ((-PsiTD_* Psi_) * x_new.block(i*n_basis(i),0, n_basis(i),1) +
                                                                         lambda_D()*pde_.stiff().transpose()*x_new.block(i*n_basis(i),0, n_basis(i),1));
                r.block((i+m_)*n_basis(i),0, n_basis(i),1) = b_.block(i*n_basis(i),0, n_basis(i),1) - (lambda_D()*pde_.stiff()*x_new.block(i*n_basis(i),0, n_basis(i),1) +
                                                                        lambda_D()*pde_.mass()*x_new.block((i+m_)*n_basis(i),0, n_basis(i),1) );
                
                U_i.block(0, 0, n_basis(i), q()) = U_.block(i*n_basis(i), 0, n_basis(i), q());
                // cov correction    
                r.block(i*n_basis(i),0, n_basis(i),1) -= (U_i * XtWX() * (U_.transpose() * x_new.head(m_* n_basis())) );
            }
            */
            f_ = x_new.topRows(n_basis()*m_);
            g_ = x_new.bottomRows(n_basis()*m_);
       
       		__start = std::chrono::high_resolution_clock::now();
            beta_ = invXtWX().solve(X().transpose() * (y_ - mPsi_ * f_)); 
       		__duration = std::chrono::high_resolution_clock::now() - __start;
        	std::cout << "-		compute nu: " << __duration.count() << std::endl;
        	
        	__start = std::chrono::high_resolution_clock::now();
            beta_coeff_ = F_*beta_;
            alpha_coeff_ = T_*beta_.tail(m_*p); 
            __duration = std::chrono::high_resolution_clock::now() - __start;
        	std::cout << "-		compute beta & alpha: " << __duration.count() << std::endl;
            //r = b_ - A_ * x_new; // senza cov
            // correzione covariate res
            //r.block(0, 0, m_*n_basis(), 1) -= alpha(k) * mPsi().transpose()*lmbH(mPsi()*z.head(m_*n_basis())); // correzione per cov
            
            // AGGIORNAMENTO per COV
            
            for(std::size_t i = 0; i < m_; i++) {
                r.block(i*n_basis(),0, n_basis(), 1) -= aux(z, i); 
            }
            
            Jold = Jnew;
            Jnew = J(f_,g_);

            rcheck = r.norm() / b_.norm() < tol_res;
            Jcheck =  std::abs((Jnew-Jold)/Jnew) < tol_;
            exit_ = Jcheck && rcheck;

            // std::cout << "Iteration n." << k << std::endl;
            // std::cout << "r:" << r.norm() / b_.norm() << std::endl;
            // std::cout << "J:" << std::abs((Jnew-Jold)/Jnew)  << std::endl;

            x_old = x_new;
            k++;
            //return;
        }
        _duration = std::chrono::high_resolution_clock::now() - _start;
        std::cout << "-		end while loop: " << _duration.count() << std::endl;
        
		std::cout << "iter: " << k << std::endl;
    
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end - start;
        std::cout << "- metodo solve(): " << duration.count() << std::endl;

        // std::ofstream file("mpsi.csv");
        // if (!file.is_open()) {
        //     std::cerr << "Unable to open file: " << "mpsii.csv" << std::endl;
        //     return;
        // }

        // // Write the header
        // file << "row,col,value\n";

        // for (int k = 0; k < mPsi_.outerSize(); ++k) {
        //     for (Eigen::SparseMatrix<double>::InnerIterator it(mPsi_, k); it; ++it) {
        //         file << it.row() << "," << it.col() << "," << it.value() << "\n";
        //     }
        // }

        // file.close();


        return;
    }

    void init_sampling(bool forced = true) {
        
        // switch (s) {
        // case Sampling::pointwise: {   // data sampled at general locations p_1, p_2, ... p_n
            // query pde to evaluate functional basis at given locations
            Psi_.resize(3);
            PsiTD_.resize(3);
            
            for(std::size_t i=0; i<data_.size(); i++){
                auto basis_evaluation = model().pde().eval_basis(core::eval::pointwise, data_[i].template get<double>(LOCS_BLOCK));
                Psi_[i] = basis_evaluation->Psi;
                // model().tensorize_psi();   // tensorize \Psi for space-time problems
                // D_ = DVector<double>::Ones(Psi_.rows()).asDiagonal();
                PsiTD_[i] = Psi_[i].transpose();
            }
        // } break;
        // }

        return;
    }

    // GCV support
    double norm(const DMatrix<double>& op1, const DMatrix<double>& op2) const { return (op1 - op2).squaredNorm(); }

    // setters
    void set_tolerance(double tol) { tol_ = tol; }
    void set_max_iter(std::size_t max_iter) { max_iter_ = max_iter; }
    void set_data(const std::vector<BlockFrame<double, int>>& data, bool reindex = false) {
        data_ = data;
    // insert an index row (if not yet present or requested)
    //     if (!df_.has_block(INDEXES_BLK) || reindex) {
    //         std::size_t n = df_.rows();
    //         DMatrix<int> idx(n, 1);
    //         for (std::size_t i = 0; i < n; ++i) idx(i, 0) = i;
    //         df_.insert(INDEXES_BLK, idx);
    //     }
	// model().runtime().set(runtime_status::require_data_stack_update);
    }

    // getters
    std::size_t q() const { return p+m_*qV; } // numero delle colonne di X
    const DiagMatrix<double>& W() const { return W_; }
    const DMatrix<double>& X() const { return X_; }  
    // const SpMatrix<double>& Q() const { return Q_; }  
    // const SpMatrix<double> Gamma() const { return Gamma_; }  
    const DMatrix<double>& Wg(std::size_t i) const { return data_[i].template get<double>(W_BLOCK); } 
    const DMatrix<double>& Vp(std::size_t i) const { return data_[i].template get<double>(V_BLOCK); } 
    const DMatrix<double>& y(std::size_t i) const { return data_[i].template get<double>(Y_BLOCK); } 
    int n_basis(std::size_t i) const { return n_basis(); }
    double alpha(std::size_t k) const { return alpha_; } // fixed to 1 
    const DMatrix<double> y() const {  // sistemare
        DMatrix<double> yps;
        yps.resize(N,1);
        int sum = 0;
        for(std::size_t i=0; i<m_; i++){
            yps.block(sum,0,n_locs(i),1) = data_[i].template get<double>(Y_BLOCK);
            sum += n_locs(i);
        }
        return yps; 
    } 
    const DMatrix<double>& locs(std::size_t i) const { return data_[i].template get<double>(LOCS_BLOCK); } 
    const SpMatrix<double> mPsi() const { 
        SpMatrix<double> mpsi;
        mpsi.resize(N, n_basis()*m_);
        std::vector<fdapde::Triplet<double>> triplet_list;
        triplet_list.reserve(N);
        std::size_t sum_row = 0;
        std::size_t sum_col = 0;
        for (std::size_t i = 0; i < m_; ++i){
            // for (SpMatrix<double> value : Psi_) {
            for(std::size_t k = 0; k < Psi_[i].outerSize(); k++){
                for (SpMatrix<double>::InnerIterator it(Psi_[i],k); it; ++it){
                    triplet_list.emplace_back(sum_row + it.row(), sum_col + it.col(), it.value());
                }
            }
            sum_row += n_locs(i);
            sum_col += n_basis();
        } 
        mpsi.setFromTriplets(triplet_list.begin(), triplet_list.end());
        mpsi.makeCompressed();
        return mpsi;
    }

    const DVector<double> alpha() const { return alpha_coeff_; }
    const DVector<double> beta() const { return beta_coeff_; }

    std::size_t n_locs(std::size_t i) const { return data_[i].template get<double>(LOCS_BLOCK).rows(); }
    std::size_t n_locs() const { 
        std::size_t sum = 0;
        for(std::size_t i=0; i<m_; i++){
            sum += n_locs(i);
        }
        return sum; 
    }

    virtual ~MixedSRPDE() = default;

}; // iterative 

};   // namespace models
}   // namespace fdapde

#endif   // __MIXED_SRPDE_H__
