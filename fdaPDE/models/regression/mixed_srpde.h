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
#include <Eigen/Dense> // for leftCols(r_); r_ # of cols

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
    std::vector<BlockFrame<double, int>> data_;       // vector of dataframes 
    SparseBlockMatrix<double, 2, 2> A_ {};         // system matrix of non-parametric problem (2N x 2N matrix)
    fdapde::SparseLU<SpMatrix<double>> invA_ {};   // factorization of matrix A
    DVector<double> b_ {};                         // right hand side of problem's linear system (1 x 2N vector)
    DMatrix<double> y_ {}; 
    DMatrix<double> X_ {};      // dimensione: N osservazioni totali * r_ covariate gruppo specifiche
	
    int N;          // N: total observations (N=n*m)
    int m_;          // m: number of levels
    int p_;         // p_: number of level-specific covariatess
    int r_;         // r_:  group-specific covariates

    int q_;          // q_: "input" DesignMatrix columns -> q_ - p_ = r_
    
    //DMatrix<double> Wg {};
    
    SpMatrix<double> I_;   // N x N sparse identity matrix 

    std::vector<SpMatrix<double>> Psi_;                // override of Psi_
    std::vector<SpMatrix<double>> PsiTD_;              // override of Psi_
    
    SpMatrix<double> mPsi_;
    SpMatrix<double> mPsiTD_;

    DMatrix<double> alpha_coeff_;                   // coefficients
    DMatrix<double> beta_coeff_;                    // beta coefficients of the original model
    SparseBlockMatrix<double, 2, 2> F_ {};          // matrix used for linear transformation of coefficients
    DMatrix<double> T_;
    DVector<int> n_locs_cum;
   
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

    void init_sampling(bool forced = true) {
        // switch (s) {
        // case Sampling::pointwise: {   // data sampled at general locations p_1, p_2, ... p_n
            // query pde to evaluate functional basis at given locations
            Psi_.resize(data_.size());
            PsiTD_.resize(data_.size());
            for(std::size_t i=0; i<data_.size(); i++){
                auto basis_evaluation = model().pde().eval_basis(core::eval::pointwise, data_[i].template get<double>(LOCS_BLOCK));
                Psi_[i] = basis_evaluation->Psi;
                PsiTD_[i] = Psi_[i].transpose();
            }
        // } break;
        // }
        return;
    }

    void init_X() {
        set_n_locs_cum();
        m_ = data_.size();                                             // m_: number of patients
        p_ = data_[0].template get<double>(V_BLOCK).cols();         // p_: patient-specific covariatess
        r_ = data_[0].template get<double>(W_BLOCK).cols();          // r_: group-specific covariates
        q_ = r_ + p_;
        X_ = DMatrix<double>::Zero(N, q()); // q = r_+m*p_

        y_.resize(N,1);
        for(std::size_t i=0; i<m_; i++){
            y_.block(n_locs_cum[i],0,n_locs(i),1) = data_[i].template get<double>(Y_BLOCK);
            X_.block(n_locs_cum[i], 0, n_locs(i), r_) = data_[i].template get<double>(W_BLOCK);
            X_.block(n_locs_cum[i], r_+i*p_, n_locs(i), p_) = data_[i].template get<double>(V_BLOCK);
        }
        
        set_F_T();
    }
    
    void analyze_data() {

    	// build multi-domain model design matrix 
    	init_X();

        // initialize empty masks
        if (!y_mask_.size()) y_mask_.resize(N); // da sostituire N - Base::n_locs()
        if (!nan_mask_.size()) nan_mask_.resize(N); // da sostiture N - Base::n_locs()

        // compute q x q dense matrix X^\top*W*X and its factorization
        // if (has_weights() && df_.is_dirty(WEIGHTS_BLK)) {
            // W_ = df_.template get<double>(WEIGHTS_BLK).col(0).asDiagonal(); // !!! da modificare !!!
            // model().runtime().set(runtime_status::require_W_update);
        // } else if (is_empty(W_)) {
            W_ = DVector<double>::Ones(N).asDiagonal(); // W_ in R^{N times N}
        // }
    	
        // compute q x q dense matrix X^\top*W*X and its factorization
        // if (has_covariates() && (df_.is_dirty(DESIGN_MATRIX_BLK) || df_.is_dirty(WEIGHTS_BLK))) {
            XtWX_ = X_.transpose() * W_ * X_; 
            invXtWX_ = XtWX_.partialPivLu(); 
        // }
        

        // derive missingness pattern from observations vector (if changed)
        if ( y_.size() > 0 ) {      // if y is not empty
            n_nan_ = 0;
            for (std::size_t i = 0; i < N; ++i) {
                if (std::isnan(y_(i, 0))) {   // requires -ffast-math compiler flag to be disabled
                    nan_mask_.set(i);
                    n_nan_++;
                    y_(i) = 0.0;   // zero out NaN
                }
            }
        }

        // updating Psi_
        for(std::size_t i = 0; i < data_.size(); ++i){
            for (std::size_t k = 0; k < n_locs(i); ++k) {
                if (nan_mask_(n_locs_cum[i]+k, 0)) {           
                    for (Eigen::SparseMatrix<double>::InnerIterator it(PsiTD_[i], k); it; ++it) {
                        it.valueRef() = 0;  
                    }
                }
            }
            PsiTD_[i].prune(0.0);
            Psi_[i] = PsiTD_[i].transpose();
        }

        return;
    }
    
    void init_model() { 

        // I_ is a NxN sparse identity matrix
        I_.resize(m_,m_);
        I_.setIdentity();
        
        // Kronecker products with the identity
        make_mPsi(); 
        mPsiTD_ = mPsi_.transpose();   

        if (runtime().query(runtime_status::is_lambda_changed)) {
            
            // auto start = std::chrono::high_resolution_clock::now();

            A_ = SparseBlockMatrix<double, 2, 2>(
                    -mPsiTD_  * W_ * mPsi_, lambda_D() * R1().transpose(), //!? 1/N in (0,0)
                    lambda_D() * R1(),        lambda_D() * R0()            );

            // auto end = std::chrono::high_resolution_clock::now();
            // std::chrono::duration<double> duration = end - start;
            // std::cout << "- assemblamento della matrice di sistema A_: " << duration.count() << std::endl;

            // start = std::chrono::high_resolution_clock::now();
            
            invA_.compute(A_);

            // end = std::chrono::high_resolution_clock::now();
            // duration = end - start;
            // std::cout << "- inversione della matrice di sistema A_: " << duration.count() << std::endl;

            // prepare rhs of linear system 
            b_.resize(A_.rows());
            for( int i=0; i < m_; ++i){
                b_.block((m_+i)*n_basis(), 0, n_basis(), 1) = lambda_D() * u(); // !!!! n_basis() * m
            }
            return;
        }

        if (runtime().query(runtime_status::require_W_update)) {
            // adjust north-west block of matrix A_ only
            A_.block(0, 0) = -mPsiTD_ * W_ * mPsi_;      // W() problemi . 
            invA_.compute(A_);
            return;
        }
    }

    DMatrix<double> lmbQ(const DMatrix<double>& x) const {
       DMatrix<double> v = X_.transpose() * W_ * x;             // X^\top*W*x
       DMatrix<double> z = invXtWX_.solve(v);                   // (X^\top*W*X)^{-1}*X^\top*W*x
       // compute W*x - W*X*z = W*x - (W*X*(X^\top*W*X)^{-1}*X^\top*W)*x = W(I - H)*x = Q*x
    //    std::cout << z << std::endl;
       return W_ * x - W_ * X_ * z;
    }

    void solve() {
        
        auto start = std::chrono::high_resolution_clock::now();
        
        fdapde_assert(y_.rows() != 0);
        DVector<double> sol;
        std::cout << "b: " << b_.rows() << "x" << b_.cols() << std::endl;
        // parametric case
        // update rhs of SR-PDE linear system
        b_.block(0, 0, m_*n_basis(), 1) = -mPsiTD_ * W_ * lmbQ(y_);   // !? 1/N .. -\Psi^T*D*Q*z
        
        // matrices U and V for application of woodbury formula
        U_ = DMatrix<double>::Zero(2 * m_ * n_basis(), q());
        U_.block(0, 0, m_* n_basis(), q()) =  mPsiTD_ * W_ * X_; // * W() * X(); !? 1/N
        V_ = DMatrix<double>::Zero(q(), 2 * m_ * n_basis());
        V_.block(0, 0, q(), m_ * n_basis()) = X_.transpose() * W_ * mPsi_; // W() * mPsi()

        // solve system (A_ + U_*(X^T*W_*X)*V_)x = b using woodbury formula from linear_algebra module
        sol = SMW<>().solve(invA_, U_, XtWX_, V_, b_);
        std::cout << "sol: " << sol.rows() << "x" << sol.cols() << std::endl;

        // store result of smoothing
        f_ = sol.head(m_*n_basis());
        beta_ = invXtWX_.solve(X_.transpose() * W_ * (y_ - mPsi_ * f_)); 

        beta_coeff_ = F_ * beta_;    
        alpha_coeff_ = T_*beta_.tail(m_*p_); 

        // store PDE misfit
        g_ = sol.tail(m_*n_basis());

        return;
    }

    // makers
    void make_mPsi() { 
        mPsi_.resize(N, n_basis()*m_);
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
        mPsi_.setFromTriplets(triplet_list.begin(), triplet_list.end());
        mPsi_.makeCompressed();
    }

    const SpMatrix<double> R0() { return Kronecker(I_, pde_.mass()); }
    const SpMatrix<double> R1() const { return Kronecker(I_, pde_.stiff()); }

    // GCV support
    double norm(const DMatrix<double>& op1, const DMatrix<double>& op2) const { return (op1 - op2).squaredNorm(); }

    // setters
    void set_data(const std::vector<BlockFrame<double, int>>& data, bool reindex = false) { data_ = data; }
    void set_N(std::size_t number) { N = number; }
    void set_F_T() {
        DMatrix<double> Ip = {}; Ip.resize(p_,p_); Ip.setIdentity(); Ip = Ip/m_;
        DMatrix<double> Iqp = {}; Iqp.resize(r_,r_); Iqp.setIdentity();
        DMatrix<double> Z1 = DMatrix<double>::Zero(p_,r_);
        DMatrix<double> Z2 = DMatrix<double>::Zero(r_, m_*p_);
        DMatrix<double> Ip_loop = {}; Ip_loop.resize(p_, m_*p_);
        for (std::size_t i=0; i<m_; i++){ Ip_loop.block(0,i*p_,p_,p_) = Ip; }
        F_ = SparseBlockMatrix<double,2,2>(
            Z1.sparseView(), Ip_loop.sparseView(),
            Iqp.sparseView(), Z2.sparseView() );       
        T_.resize(m_*p_,m_*p_); T_.setIdentity(); T_ = (m_-1.0)/m_ * T_;
        
        for (std::size_t i=0; i<m_; i++){
            for(std::size_t j=0; j<m_; j++){
                if(T_(i*p_,j*p_) == 0){ T_.block(i*p_,j*p_,p_,p_) = -Ip; }
            }
        }
        return;
    }
    void set_n_locs_cum() {
        int sum = 0;
        n_locs_cum.resize(data_.size());
        for(std::size_t i = 0; i < data_.size(); ++i){
            n_locs_cum[i] = sum;
            sum += data_[i].rows();
        }
        return;
    }

    // getters
    std::size_t q() const { return r_+m_*p_; }
    std::size_t n_locs(std::size_t i) const { return data_[i].template get<double>(LOCS_BLOCK).rows(); }
    // const DMatrix<double>& locs(std::size_t i) const { return data_[i].template get<double>(LOCS_BLOCK); } 
    const DVector<double>& f() const { return f_; };
    const DVector<double> f(std::size_t i) const {return f_.block(i*n_basis(), 0, n_basis(), 1); }
    const DVector<double> alpha() const { return alpha_coeff_; }
    const DVector<double> beta() const { return beta_coeff_; }
    const DVector<double> betanp() const { return beta_; }
    
    virtual ~MixedSRPDE() = default;

}; // monolithic


template <>
class MixedSRPDE<SpaceOnly,iterative> : public RegressionBase<MixedSRPDE<SpaceOnly,iterative>, SpaceOnly> {
   private:
    DVector<double> b_ {};                         // right hand side of problem's linear system (1 x 2N vector)
    DMatrix<double> X_ {};                         // design matrix
    // SpMatrix<double> Q_ {};                     // Q = I - X(X^T X)^{-1}X^T
    int N;                                         // N: total observations (N=n*m)
    int m_;                                        // m: number of patients
    int p_;                                        // p_: patient-specific covariatess
    int r_;                                         // r_: group-specific covariates
    double alpha_ = 1;                             // alpha: acceleration parameter
    SparseBlockMatrix<double, 2, 2> P_ {};         // preconditioning matrix of the Richardson scheme
    std::vector<fdapde::SparseLU<SpMatrix<double>>> invP_ ;   // factorization of matrix P
    SpMatrix<double> I_;                           // N x N sparse identity matrix 
    DMatrix<double> s_;                            // N x 1 initial condition vector
    DMatrix<double> u_;                            // discretized forcing [1/DeltaT * (u_1 + R_0*s) \ldots u_n]
    DMatrix<double> alpha_coeff_;                  // coefficients
    DMatrix<double> beta_coeff_;                   // beta coefficients of the original model
    SparseBlockMatrix<double, 2, 2> F_ {};         // matrix used for linear transformation of coefficients
    DMatrix<double> T_;
    std::vector<BlockFrame<double, int>> data_;    // vector of dataframes 
    std::vector<SpMatrix<double>> Psi_;            // override of Psi_
    std::vector<SpMatrix<double>> PsiTD_;          // override of Psi_
    SpMatrix<double> mPsi_;
    SpMatrix<double> mPsiTD_;
    DMatrix<double> y_;                            // vector of observations
    DVector<int> n_locs_cum;                       // cumulative sum of locations

    std::vector<DMatrix<double>> _U;
    std::vector<DMatrix<double>> _V;
    // construction of the design matrix X_
    void init_X() {

        m_ = data_.size();                                           // m_: number of patients
        p_ = data_[0].template get<double>(V_BLOCK).cols();          // p_: patient-specific covariatess
        r_ = data_[0].template get<double>(W_BLOCK).cols();           // r_: group-specific covariates

        mPsi_ = mPsi();
        mPsiTD_ = mPsi_.transpose();

        y_.resize(N,1);
        X_ = DMatrix<double>::Zero(N, q());
        for(std::size_t i=0; i<m_; i++){
            y_.block(n_locs_cum[i],0,n_locs(i),1) = data_[i].template get<double>(Y_BLOCK);
            X_.block(n_locs_cum[i], 0, n_locs(i), r_) = Wg(i);
            X_.block(n_locs_cum[i], r_+i*p_, n_locs(i), p_ ) = Vp(i); // matrix V: diagonal block matrix with V1,V2,...,Vm on the diagonal
        }

    }

    // iterative scheme 
    double tol_ = 1e-4;             // tolerance (stopping criterion)
    double tol_res = 1e-8;	
    std::size_t max_iter_ = 10;     // maximum number of iteration

   public:
    using RegularizationType = SpaceOnly;
    using Base = RegressionBase<MixedSRPDE<RegularizationType, iterative>, RegularizationType>;
    IMPORT_REGRESSION_SYMBOLS;
    using Base::lambda_D;           // smoothing parameter in space
    using Base::n_basis;            // number of spatial basis
    using Base::runtime;            // runtime model status
    using Base::pde_;               // parabolic differential operator df/dt + Lf - u
    static constexpr int n_lambda = 1;

    // constructors
    MixedSRPDE() = default;
    MixedSRPDE(const pde_ptr& pde, Sampling s) : Base(pde, s) { pde_.init(); };

    // check: if SpaceOnly tensorizes \Psi matrix -> if yes uncomment the following line
    // void tensorize_psi() { return; }   // avoid tensorization of \Psi matrix
    void init_regularization(){
        pde_.init();
        s_ = pde_.initial_condition();
        u_ = pde_.force();   // forcing term... questo è definito in (3.5) thesis Ischia pag. 26  // u = \Psi^T*Q*z        // check: I'm not sure about this 
    }

    void init_sampling(bool forced = true) {
        // switch (s) {
        // case Sampling::pointwise: {   // data sampled at general locations p_1, p_2, ... p_n
            // query pde to evaluate functional basis at given locations
            Psi_.resize(data_.size());
            PsiTD_.resize(data_.size());
            for(std::size_t i=0; i<data_.size(); i++){
                auto basis_evaluation = model().pde().eval_basis(core::eval::pointwise, data_[i].template get<double>(LOCS_BLOCK));
                Psi_[i] = basis_evaluation->Psi;
                PsiTD_[i] = Psi_[i].transpose();
            }
        // } break;
        // }
        return;
    }

    void analyze_data() { 

        set_n_locs_cum();

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
        // if (has_covariates() && (df_.is_dirty(DESIGN_MATRIX_BLK) || df_.is_dirty(WEIGHTS_BLK))) { // non ho più queste matrici popolate
            XtWX_ = X().transpose() * W_ * X(); // Da valutare
            invXtWX_ = XtWX_.partialPivLu(); 
        // }

        // derive missingness pattern from observations vector (if changed)
        if ( y_.size() > 0 ) {      // if y is not empty
            n_nan_ = 0;
            for (std::size_t i = 0; i < y_.rows(); ++i) {
                if (std::isnan(y_(i, 0))) {   // requires -ffast-math compiler flag to be disabled
                    nan_mask_.set(i);
                    n_nan_++;
                    y_(i) = 0.0;   // zero out NaN
                }
            }
        //     if (has_nan()) model().runtime().set(runtime_status::require_psi_correction);
        }

        for(std::size_t i = 0; i < data_.size(); ++i){
            for (std::size_t k = 0; k < n_locs(i); ++k) {
                if (nan_mask_(n_locs_cum[i]+k, 0)) {           
                    for (Eigen::SparseMatrix<double>::InnerIterator it(PsiTD_[i], k); it; ++it) {
                        it.valueRef() = 0; 
                        // std::cout << "(" << it.row() << ", " << it.col() << ") -> " << it.value() << std::endl;  
                    }
                }
            }
            PsiTD_[i].prune(0.0);
            Psi_[i] = PsiTD_[i].transpose();
        }

        return;
    }

    DMatrix<double> lmbQ(const DMatrix<double>& x) const {
       DMatrix<double> v = X_.transpose() * W_ * x;     // X^\top*W*x
       DMatrix<double> z = invXtWX_.solve(v);           // (X^\top*W*X)^{-1}*X^\top*W*x
       // compute W*x - W*X*z = W*x - (W*X*(X^\top*W*X)^{-1}*X^\top*W)*x = W(I - H)*x = Q*x
       return W_ * x - W_ * X_ * z;
    }


    void init_model() { 

        auto start = std::chrono::high_resolution_clock::now();

        b_ = DMatrix<double>::Zero(2*n_basis()*m_, 1);

        invP_.resize(data_.size());
        _U.resize(data_.size());
        _V.resize(data_.size());
        // Initialization
        auto _start = std::chrono::high_resolution_clock::now();
        auto _start_ = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> _duration_ = std::chrono::high_resolution_clock::now() - _start_;

        for (std::size_t i = 0; i < m_; i++){

            _U[i] = DMatrix<double>::Zero(2*n_basis(), q()); 
            _V[i] = DMatrix<double>::Zero(q(), 2*n_basis());
        
            // start = std::chrono::high_resolution_clock::now();
            _start_ = std::chrono::high_resolution_clock::now();    
            P_ = SparseBlockMatrix<double, 2, 2>(
                -PsiTD_[i]*Psi_[i],                     lambda_D() * pde_.stiff().transpose(),
                lambda_D() * pde_.stiff(),             lambda_D() * pde_.mass()                 );
            _duration_ = std::chrono::high_resolution_clock::now() - _start_;
            std::cout << "-		    build P_: " << _duration_.count() << std::endl;
            // duration = std::chrono::high_resolution_clock::now() - start;
            // start = std::chrono::high_resolution_clock::now();
            _start_ = std::chrono::high_resolution_clock::now();
            invP_[i].compute(P_); 
            _duration_ = std::chrono::high_resolution_clock::now() - _start_;
            std::cout << "-		    build invP_: " << _duration_.count() << std::endl;

            _start_ = std::chrono::high_resolution_clock::now();
            _U[i].block(0, 0, n_basis(), r_) = PsiTD_[i]*Wg(i);
            _U[i].block(0, r_+i*p_, n_basis(), p_ ) = PsiTD_[i]*Vp(i);
            _duration_ = std::chrono::high_resolution_clock::now() - _start_;
            std::cout << "-		    build _U[i]: " << _duration_.count() << std::endl;
            _start_ = std::chrono::high_resolution_clock::now();
            _V[i].block(0, 0, r_, n_basis()) = Wg(i).transpose()*Psi_[i];
            _V[i].block(r_+i*p_, 0, p_, n_basis()) = Vp(i).transpose()*Psi_[i]; 
            _duration_ = std::chrono::high_resolution_clock::now() - _start_;
            std::cout << "-		    build _V[i]: " << _duration_.count() << std::endl;
        }
        std::chrono::duration<double> _duration = std::chrono::high_resolution_clock::now() - _start;
        std::cout << "-		    P[i], invP[i], U[i], V[i]: " << _duration.count() << std::endl;
        // matrix for retrieve initial coefficients (alpha and beta)
        _start = std::chrono::high_resolution_clock::now();
        set_F_T();
        _duration = std::chrono::high_resolution_clock::now() - _start;
        std::cout << "-		    build F & T: " << _duration.count() << std::endl;
        
        std::chrono::duration<double> duration = std::chrono::high_resolution_clock::now() - start;
        std::cout << "-     init model: " << duration.count() << std::endl;
        return; 
    }

    // functional minimized by the iterative scheme:  J(f)=norm(y-X\nu-f_n)+\lambda\sum_{i=1}^m norm(\nabla f_i)
    double J(const DMatrix<double>& f, const DMatrix<double>& g) const{
        DMatrix<double> fhat = mPsi_ * f; 
        return (y_ - X() * beta_ - fhat).squaredNorm() + lambda_D()*g.squaredNorm(); 
    }

    void solve() { 
        std::cout << "solve" << std::endl;
        auto start = std::chrono::high_resolution_clock::now();
        
        fdapde_assert(y_.rows() != 0);

        DVector<double> x_new = DMatrix<double>::Zero(2*n_basis()*m_, 1); 
        b_.block(0, 0, n_basis()*m_, 1) = -mPsiTD_ * lmbQ(y_); 
        
        DVector<double> r = b_; //DMatrix<double>::Zero(2*m_*n_basis(), 1);
        double Jnew;
        double Jold;
        
        // internal utilities (for initialization)
        DVector<double> bi = DMatrix<double>::Zero(2*n_basis(), 1);
        DVector<double> ri = DMatrix<double>::Zero(2*n_basis(),1);
        DVector<double> zi = DMatrix<double>::Zero(2*n_basis(),1);
         
        // invP_.resize(data_.size());
        // _U.resize(data_.size());
        // _V.resize(data_.size());
        // // Initialization
        auto _start = std::chrono::high_resolution_clock::now();
        auto _start_ = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> _duration_ = std::chrono::high_resolution_clock::now() - _start_;

        std::cout << " --- initialization ---" << std::endl; 
        for (std::size_t i = 0; i < m_; i++){

            bi.block(0,0,n_basis(),1) = b_.block(i*n_basis(), 0, n_basis(), 1);
                        
            // solve system (A_ + U_*(X^T*W_*X)*V_)x = b using woodbury formula from linear_algebra module
            _start_ = std::chrono::high_resolution_clock::now();
            zi = SMW<>().solve(invP_[i], _U[i], XtWX(), _V[i], bi);
            _duration_ = std::chrono::high_resolution_clock::now() - _start_;
            std::cout << "-		    SMW: " << _duration_.count() << std::endl;
            
            _start_ = std::chrono::high_resolution_clock::now();
            x_new.block(i*n_basis(), 0, n_basis(),1) = zi.head(n_basis());
            x_new.block((i+m_)*n_basis(),0, n_basis(),1) = zi.tail(n_basis()); 
             
        
            r.block(n_basis()*i,0, n_basis(),1) -=   ((-PsiTD_[i]*Psi_[i]) * zi.head(n_basis()) +
                                           			  lambda_D()*pde_.stiff().transpose()*zi.tail(n_basis()));
  
            // correzione cov fuori
            r.block(n_basis()*(m_+i),0, n_basis(),1) -= (lambda_D()*pde_.stiff()*zi.head(n_basis()) +
                                                          lambda_D()*pde_.mass()*zi.tail(n_basis()) );
            
            _duration_ = std::chrono::high_resolution_clock::now() - _start_;
            std::cout << "-		    linear algebra: " << _duration_.count() << std::endl;
        }
		
        std::chrono::duration<double> _duration = std::chrono::high_resolution_clock::now() - _start;
        std::cout << "-		inizializzazione: " << _duration.count() << std::endl;
		_start = std::chrono::high_resolution_clock::now();
        
        _start = std::chrono::high_resolution_clock::now();
        auto tmp = Wg(0).rows();
        _duration = std::chrono::high_resolution_clock::now() - _start;
        std::cout << "-		access data frame: " << _duration.count() << "" << std::endl;
        bi = DMatrix<double>::Zero(2*n_basis(), 1);

        // correzione covariate
        DMatrix<double> u = DMatrix<double>::Zero(q(),1);
       	for(std::size_t i = 0; i < m_; ++i){
       		u.block(0, 0, r_, 1) += Wg(i).transpose()*Psi_[i]*x_new.block(i*n_basis(), 0, n_basis(), 1); 	
       		u.block(i*p_+r_, 0, p_, 1) = Vp(i).transpose()*Psi_[i]*x_new.block(i*n_basis(), 0, n_basis(), 1);	
       	}
        DMatrix<double> w = invXtWX().solve(u);         
		
		for(std::size_t i = 0; i < m_; i++){
            //DMatrix<double> w = Wg(i)*z.block(0, 0, r_, 1) + Vp(i)*z.block(i*p_+r_, 0, p_, 1);
			r.block(i*n_basis(),0, n_basis(), 1) -= Psi_[i].transpose()*(Wg(i)*w.block(0, 0, r_, 1) + 
                                                                         Vp(i)*w.block(i*p_+r_, 0, p_, 1)); 
		}
        
		_duration = std::chrono::high_resolution_clock::now() - _start;
        std::cout << "-		residuo: " << _duration.count() << std::endl;
		
        // store result of smoothing
        f_ = x_new.head(m_*n_basis());      // f0        
        g_ = x_new.tail(m_*n_basis());      // g0

        beta_ = invXtWX().solve(X().transpose() * (y_ - mPsi_ * f_)); 
        
        beta_coeff_ = F_*beta_;
        alpha_coeff_ = T_*beta_.tail(m_*p_); 
        
        DVector<double> z;
        z = DMatrix<double>::Zero(r.rows(), 1);

        double c = 2;
        Jnew = J(f_,g_);
        Jold = (1 + c) * Jnew;
        
        // iteration loop
        std::size_t k = 1;                                      // iteration number    
        bool rcheck = r.norm() / b_.norm() < tol_res;           // stop by residual    
        bool Jcheck =  std::abs((Jnew-Jold)/Jnew) < tol_;       // stop by J
        bool exit_ = Jcheck && rcheck;
      
        _start = std::chrono::high_resolution_clock::now();

        // iterative scheme for minimization of functional 
        while (k < max_iter_ && !exit_) /* ischia pag 25 */ {
            auto __start = std::chrono::high_resolution_clock::now();
            for(std::size_t i = 0; i < m_; i++){                  
                    // valutare implementazione di lmbQ(yi)
                    _start_ = std::chrono::high_resolution_clock::now();
                    bi.block(0,0,n_basis(),1) = r.block(i*n_basis(), 0, n_basis(), 1) ; 
                    bi.block(n_basis(), 0, n_basis(), 1) = r.block( (i+m_)*n_basis(), 0, n_basis(), 1);
                    _duration_ = std::chrono::high_resolution_clock::now() - _start_;
                    std::cout << "-		    update b_i: " << _duration_.count() << std::endl;

                    _start_ = std::chrono::high_resolution_clock::now();
                    zi = SMW<>().solve(invP_[i], _U[i], XtWX(), _V[i], bi);
                    _duration_ = std::chrono::high_resolution_clock::now() - _start_;
                    std::cout << "-		    SMW: " << _duration_.count() << std::endl;

                    _start_ = std::chrono::high_resolution_clock::now();
                    x_new.block(n_basis()*i,0, n_basis(),1) += alpha(k)*zi.head(n_basis());  
                    x_new.block(n_basis()*(m_+i),0, n_basis(),1) += alpha(k)*zi.tail(n_basis());
                    
                    r.block(n_basis()*i,0, n_basis(),1) -=  alpha(k) * ((-PsiTD_[i]* Psi_[i]) * zi.head(n_basis()) +
                                                                         lambda_D()*pde_.stiff().transpose()*zi.tail(n_basis()));
                                                                                             
                    r.block(n_basis()*(m_+i),0, n_basis(),1) -= alpha(k)*(lambda_D()*pde_.stiff()*zi.head(n_basis()) +
                                                                            lambda_D()*pde_.mass()*zi.tail(n_basis()) );
                    
                    z.block(i*n_basis(),0, n_basis(),1) = zi.head(n_basis());  
                    z.block(n_basis()*(m_+i),0, n_basis(),1) = zi.tail(n_basis());
                    _duration_ = std::chrono::high_resolution_clock::now() - _start_;
                    std::cout << "-		    linear algebra: " << _duration_.count() << std::endl;
 
                    
            }

            std::chrono::duration<double> __duration = std::chrono::high_resolution_clock::now() - __start;
        	std::cout << "-		costo singola iter: " << __duration.count() << std::endl;
		
            f_ = x_new.topRows(n_basis()*m_);
            g_ = x_new.bottomRows(n_basis()*m_);
       
       		__start = std::chrono::high_resolution_clock::now();

            beta_ = invXtWX().solve(X().transpose() * (y_ - mPsi_ * f_)); 

       		__duration = std::chrono::high_resolution_clock::now() - __start;
        	std::cout << "-		compute nu: " << __duration.count() << std::endl;
        	__start = std::chrono::high_resolution_clock::now();
            
            beta_coeff_ = F_*beta_;
            alpha_coeff_ = T_*beta_.tail(m_*p_); 

            __duration = std::chrono::high_resolution_clock::now() - __start;
        	std::cout << "-		compute beta & alpha: " << __duration.count() << std::endl;

            // correzione covariate 
            u = DMatrix<double>::Zero(q(),1);
            for(std::size_t i = 0; i < m_; ++i){
                u.block(0, 0, r_, 1) += Wg(i).transpose()*Psi_[i]*z.block(i*n_basis(), 0, n_basis(), 1); 	
       		    u.block(i*p_+r_, 0, p_, 1) = Vp(i).transpose()*Psi_[i]*z.block(i*n_basis(), 0, n_basis(), 1);	
       	    }
            w = invXtWX().solve(u);         
		
		    for(std::size_t i = 0; i < m_; i++){
			    r.block(i*n_basis(),0, n_basis(), 1) -= Psi_[i].transpose()*(Wg(i)*w.block(0, 0, r_, 1) + 
                                                                         Vp(i)*w.block(i*p_+r_, 0, p_, 1)); 
		    }
	          
            Jold = Jnew;
            Jnew = J(f_,g_);

            rcheck = r.norm() / b_.norm() < tol_res;
            Jcheck =  std::abs((Jnew-Jold)/Jnew) < tol_;
            exit_ = Jcheck && rcheck;

            k++;

        }

        _duration = std::chrono::high_resolution_clock::now() - _start;
        std::cout << "-		end while loop: " << _duration.count() << std::endl;
        
		std::cout << "iter: " << k << std::endl;
    
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end - start;
        std::cout << "- metodo solve(): " << duration.count() << std::endl;

        return;
    }

    // GCV support
    double norm(const DMatrix<double>& op1, const DMatrix<double>& op2) const { return (op1 - op2).squaredNorm(); }

    // setters
    void set_tolerance(double tol) { tol_ = tol; }
    void set_max_iter(std::size_t max_iter) { max_iter_ = max_iter; }
    void set_data(const std::vector<BlockFrame<double, int>>& data, bool reindex = false) {  data_ = data;  }
    void set_N(std::size_t number) { N = number; }
    void set_n_locs_cum() {
        int sum = 0;
        n_locs_cum.resize(data_.size());
        for(std::size_t i = 0; i < data_.size(); ++i){
            n_locs_cum[i] = sum;
            sum += data_[i].rows();
        }
        return;
    }

    // getters
    std::size_t q() const { return r_+m_*p_; } // numero delle colonne di X
    const DMatrix<double>& X() const { return X_; }  
    const DMatrix<double>& Wg(std::size_t i) const { return data_[i].template get<double>(W_BLOCK); } 
    const DMatrix<double>& Vp(std::size_t i) const { return data_[i].template get<double>(V_BLOCK); } 
    const DMatrix<double>& y(std::size_t i) const { return data_[i].template get<double>(Y_BLOCK); } 
    int n_basis(std::size_t i) const { return n_basis(); }
    double alpha(std::size_t k) const { return alpha_; } // fixed to 1 
    
    // const DMatrix<double>& locs(std::size_t i) const { return data_[i].template get<double>(LOCS_BLOCK); } 
    const DVector<double>& f() const { return f_; };
    const DVector<double> f(std::size_t i) const {return f_.block(i*n_basis(), 0, n_basis(), 1); }

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

    void set_F_T(){
        DMatrix<double> Ip = {}; Ip.resize(p_,p_); Ip.setIdentity(); Ip = Ip/m_;
        DMatrix<double> Iqp = {}; Iqp.resize(r_,r_); Iqp.setIdentity();
        DMatrix<double> Z1 = DMatrix<double>::Zero(p_,r_);
        DMatrix<double> Z2 = DMatrix<double>::Zero(r_, m_*p_);
        DMatrix<double> Ip_loop = {}; Ip_loop.resize(p_, m_*p_);
        for (std::size_t i=0; i<m_; i++){ Ip_loop.block(0,i*p_,p_,p_) = Ip; }
        F_ = SparseBlockMatrix<double,2,2>(
            Z1.sparseView(), Ip_loop.sparseView(),
            Iqp.sparseView(), Z2.sparseView() ); 
        T_.resize(m_*p_,m_*p_); T_.setIdentity(); T_ = (m_-1.0)/m_ * T_;
        for (std::size_t i=0; i<m_; i++){
            for(std::size_t j=0; j<m_; j++){
                if(T_(i*p_,j*p_) == 0){ T_.block(i*p_,j*p_,p_,p_) = -Ip; }
            }
        }
    }

    const DVector<double> alpha() const { return alpha_coeff_; }
    const DVector<double> beta() const { return beta_coeff_; }
    const DVector<double> betanp() const { return beta_; }
    

    std::size_t n_locs(std::size_t i) const { return data_[i].template get<double>(LOCS_BLOCK).rows(); }

    virtual ~MixedSRPDE() = default;

}; // iterative 

};   // namespace models
}   // namespace fdapde

#endif   // __MIXED_SRPDE_H__