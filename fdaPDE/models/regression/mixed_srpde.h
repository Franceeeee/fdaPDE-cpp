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
    std::vector<BlockFrame<double, int>> data_;       // vector of dataframes 
    SparseBlockMatrix<double, 2, 2> A_ {};         // system matrix of non-parametric problem (2N x 2N matrix)
    fdapde::SparseLU<SpMatrix<double>> invA_ {};   // factorization of matrix A
    DVector<double> b_ {};                         // right hand side of problem's linear system (1 x 2N vector)
    DMatrix<double> y_ {}; 
    DMatrix<double> X_ {};      // dimensione: N osservazioni totali * p covariate gruppo specifiche
	
    int N;          // N: total observations (N=n*m)
    int m_;          // m: number of patients
    int qV;         // qV: patient-specific covariatess
    int p;          // p: group-specific covariates

    DMatrix<double> Wg {};
    
    SpMatrix<double> I_;   // N x N sparse identity matrix 

    std::vector<SpMatrix<double>> Psi_;                // override of Psi_
    std::vector<SpMatrix<double>> PsiTD_;              // override of Psi_
    
    SpMatrix<double> mPsi_;
    SpMatrix<double> mPsiTD_;

    DVector<double> alpha_coeff_;                   // coefficients
    DVector<double> beta_coeff_;                    // beta coefficients of the original model
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
        qV = data_[0].template get<double>(V_BLOCK).cols();         // qV: patient-specific covariatess
        p = data_[0].template get<double>(W_BLOCK).cols();          // p: group-specific covariates

        X_ = DMatrix<double>::Zero(N, q()); // q = p+m*qV

        // populating y
        y_.resize(N,1);
        for(std::size_t i=0; i<m_; i++){
            y_.block(n_locs_cum[i],0,n_locs(i),1) = data_[i].template get<double>(Y_BLOCK);
        }

        // populating Wg
        Wg.resize(N,p);
        for(std::size_t i = 0; i < m_; i++){
            Wg.block(n_locs_cum[i], 0, n_locs(i), p) = data_[i].template get<double>(W_BLOCK);
        }
        
        // populating X
        X_.leftCols(p) = Wg; // matrix W: first column of X
        for ( int i = 0; i < m_ ; i++ ) { // cycle over the number of patients that adds q patient-specific covariates to X_ at each loop
            X_.block(n_locs_cum[i], p+i*qV, n_locs(i), qV) = data_[i].template get<double>(V_BLOCK); // matrix V: diagonal block matrix with V1,V2,...,Vm on the diagonal
        }

        // matrixes for retrieve initial coefficients (alpha and beta)
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
                    -1/N*mPsiTD_  * W_ * mPsi_, lambda_D() * R1().transpose(), //!!!
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
        b_.block(0, 0, m_*n_basis(), 1) = -1/N*mPsiTD_ * W_ * lmbQ(y_);   // !!! -\Psi^T*D*Q*z
        
        // matrices U and V for application of woodbury formula
        U_ = DMatrix<double>::Zero(2 * m_ * n_basis(), q());
        U_.block(0, 0, m_* n_basis(), q()) = mPsiTD_ * W_ * X_; // * W() * X();
        V_ = DMatrix<double>::Zero(q(), 2 * m_ * n_basis());
        V_.block(0, 0, q(), m_ * n_basis()) = X_.transpose() * W_ * mPsi_; // W() * mPsi()

        // solve system (A_ + U_*(X^T*W_*X)*V_)x = b using woodbury formula from linear_algebra module
        sol = SMW<>().solve(invA_, U_, XtWX_, V_, b_);
        std::cout << "sol: " << sol.rows() << "x" << sol.cols() << std::endl;

        // store result of smoothing
        f_ = sol.head(m_*n_basis());
        // beta_ = invXtWX().solve(X().transpose() ) * (y_ - mPsi_ * f_); // X().transpose() * W()
        beta_ = invXtWX_.solve(X_.transpose() * W_ * (y_ - mPsi_ * f_)); 

        // std::cout << "nu:\n" << beta_ << std::endl;

        // std::cout << "dimensioni F_: " << F_.rows() <<" x " << F_.cols() << std::endl;
        // std::cout << "dimensioni T_: " << T_.rows() <<" x " << T_.cols() << std::endl;
        // std::cout << "dimensioni nu_: " << beta_.rows() <<" x " << beta_.cols() << std::endl;
        
        // std::cout << "Matrix F_: \n" << F_ << std::endl;
        // std::cout << "Matrix T_: \n" << T_ << std::endl;
         
        beta_coeff_ = F_*beta_;
        alpha_coeff_ = T_*beta_.tail(m_*qV); 

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
        DMatrix<double> Ip = {}; Ip.resize(qV,qV); Ip.setIdentity(); Ip = Ip/m_;
        DMatrix<double> Iqp = {}; Iqp.resize(p,p); Iqp.setIdentity();
        DMatrix<double> Z1 = DMatrix<double>::Zero(qV,p);
        DMatrix<double> Z2 = DMatrix<double>::Zero(p, m_*qV);
        DMatrix<double> Ip_loop = {}; Ip_loop.resize(qV, m_*qV);
        for (std::size_t i=0; i<m_; i++){ Ip_loop.block(0,i*qV,qV,qV) = Ip; }
        F_ = SparseBlockMatrix<double,2,2>(
            Z1.sparseView(), Ip_loop.sparseView(),
            Iqp.sparseView(), Z2.sparseView() );       
        T_.resize(m_*qV,m_*qV); T_.setIdentity(); T_ = (m_-1.0)/m_ * T_;
        
        for (std::size_t i=0; i<m_; i++){
            for(std::size_t j=0; j<m_; j++){
                if(T_(i*qV,j*qV) == 0){ T_.block(i*qV,j*qV,qV,qV) = -Ip; }
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
    std::size_t q() const { return p+m_*qV; }
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
    int qV;                                        // qV: patient-specific covariatess
    int p;                                         // p: group-specific covariates
    double alpha_ = 1;                             // alpha: acceleration parameter
    SparseBlockMatrix<double, 2, 2> P_ {};         // preconditioning matrix of the Richardson scheme
    std::vector<fdapde::SparseLU<SpMatrix<double>>> invP_ ;   // factorization of matrix P
    SpMatrix<double> I_;                           // N x N sparse identity matrix 
    DMatrix<double> s_;                            // N x 1 initial condition vector
    DMatrix<double> u_;                            // discretized forcing [1/DeltaT * (u_1 + R_0*s) \ldots u_n]
    DVector<double> alpha_coeff_;                  // coefficients
    DVector<double> beta_coeff_;                   // beta coefficients of the original model
    SparseBlockMatrix<double, 2, 2> F_ {};         // matrix used for linear transformation of coefficients
    DMatrix<double> T_;
    std::vector<BlockFrame<double, int>> data_;    // vector of dataframes 
    std::vector<SpMatrix<double>> Psi_;            // override of Psi_
    std::vector<SpMatrix<double>> PsiTD_;          // override of Psi_
    SpMatrix<double> mPsi_;
    SpMatrix<double> mPsiTD_;
    DMatrix<double> y_;                            // vector of observations
    DVector<int> n_locs_cum;                       // cumulative sum of locations

    // construction of the design matrix X_
    void init_X() {

        m_ = data_.size();                                           // m_: number of patients
        qV = data_[0].template get<double>(V_BLOCK).cols();          // qV: patient-specific covariatess
        p = data_[0].template get<double>(W_BLOCK).cols();           // p: group-specific covariates

        mPsi_ = mPsi();
        mPsiTD_ = mPsi_.transpose();

        y_.resize(N,1);
        for(std::size_t i=0; i<m_; i++){
            y_.block(n_locs_cum[i],0,n_locs(i),1) = data_[i].template get<double>(Y_BLOCK);
        }

        X_ = DMatrix<double>::Zero(N, q());

        for ( int i = 0; i < m_ ; i++ ) { // cycle over the number of patients that adds q patient-specific covariates to X_ at each loop
            X_.block(n_locs_cum[i], 0, n_locs(i), p) = Wg(i);
            X_.block(n_locs_cum[i], p+i*qV, n_locs(i), qV ) = Vp(i); // matrix V: diagonal block matrix with V1,V2,...,Vm on the diagonal
        }
        
    }

    DMatrix<double> aux(const DMatrix<double>& x, std::size_t i) const{
       	DMatrix<double> v = DMatrix<double>::Zero(N,1);
       	for(std::size_t k = 0; k < m_; ++k){
       		v.block(n_locs_cum[k], 0, n_locs(k), 1) = Psi_[k]*x.block(k*n_basis(), 0, n_basis(), 1);	
       	}

       	DMatrix<double> u = DMatrix<double>::Zero(q(),1);

       	for(std::size_t k = 0; k < m_; ++k){
       		u.block(0, 0, p, 1) += Wg(k).transpose()*v.block(n_locs_cum[k], 0, n_locs(k),1);	
       		u.block(k*qV+p, 0, qV, 1) = Vp(k).transpose()*v.block(n_locs_cum[k], 0, n_locs(k), 1);	
       	}

       	DMatrix<double> z = invXtWX().solve(u);         
		DMatrix<double> w = Wg(i)*z.block(0, 0, p, 1) + Vp(i)*z.block(i*qV+p, 0, qV, 1);
        return Psi_[i].transpose()*w; 
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

        // matrix for retrieve initial coefficients (alpha and beta)
        set_F_T();

        std::chrono::duration<double> duration = std::chrono::high_resolution_clock::now() - start;
        // std::cout << "- assemblamento delle matrici per i coefficienti beta: " << duration.count() << std::endl;
        return; 
    }

    // functional minimized by the iterative scheme:  J(f)=norm(y-X\nu-f_n)+\lambda\sum_{i=1}^m norm(\nabla f_i)
    double J(const DMatrix<double>& f, const DMatrix<double>& g) const{
        DMatrix<double> fhat = mPsi_ * f; 
        return (y_ - X() * beta_ - fhat).squaredNorm() + lambda_D()*g.squaredNorm(); 
    }

    void solve() { 
        // std::cout << "solve" << std::endl;
        // auto start = std::chrono::high_resolution_clock::now();
        
        fdapde_assert(y_.rows() != 0);

        DVector<double> x_new = DMatrix<double>::Zero(2*n_basis()*m_, 1); 
        b_.block(0, 0, n_basis()*m_, 1) = -mPsiTD_ * lmbQ(y_); 
        
        //DVector<double> x_old = x_new; 
        DVector<double> r = b_; //DMatrix<double>::Zero(2*m_*n_basis(), 1);
        double Jnew;
        double Jold;
        
        // internal utilities (for initialization)
        DVector<double> bi = DMatrix<double>::Zero(2*n_basis(), 1);
        DVector<double> xi0 = DMatrix<double>::Zero(2*n_basis(), 1); 
        DMatrix<double> U_i = DMatrix<double>::Zero(2*n_basis(), q());  // U = PsiTD * X
        DMatrix<double> V_i = DMatrix<double>::Zero(q(), 2*n_basis());  // V = X^T * Psi

        invP_.resize(data_.size());

        // auto _start = std::chrono::high_resolution_clock::now();   
        
        for (std::size_t i = 0; i < m_; i++){

            bi = DMatrix<double>::Zero(2*n_basis(), 1);
            xi0 = DMatrix<double>::Zero(2*n_basis(), 1); 

            // std::chrono::duration<double> duration = std::chrono::high_resolution_clock::now() - start;
            // std::cout << "- assemblamento della matrice di sistema A_: " << duration.count() << std::endl;

            U_i = DMatrix<double>::Zero(2*n_basis(), q());      // re-initialization
            V_i = DMatrix<double>::Zero(q(), 2*n_basis());

            // start = std::chrono::high_resolution_clock::now();
            
            P_ = SparseBlockMatrix<double, 2, 2>(
                -PsiTD_[i]*Psi_[i],                     lambda_D() * pde_.stiff().transpose(),
                lambda_D() * pde_.stiff(),             lambda_D() * pde_.mass()                 );
            
            // duration = std::chrono::high_resolution_clock::now() - start;
            // std::cout << "-		assemblamento P_: " << duration.count() << std::endl;
            // start = std::chrono::high_resolution_clock::now();

            invP_[i].compute(P_); 

            // duration = std::chrono::high_resolution_clock::now() - start;
            // std::cout << "-		inversione P_: " << duration.count() << std::endl;

            bi.block(0,0,n_basis(),1) = b_.block(i*n_basis(), 0, n_basis(), 1);
            
            U_i.block(0, 0, n_basis(), p) = PsiTD_[i]*Wg(i);
            U_i.block(0, p+i*qV, n_basis(), qV ) = PsiTD_[i]*Vp(i);
    
            V_i.block(0, 0, p, n_basis()) = Wg(i).transpose()*Psi_[i];
            V_i.block(p+i*qV, 0, qV, n_basis()) = Vp(i).transpose()*Psi_[i]; 
            
            // solve system (A_ + U_*(X^T*W_*X)*V_)x = b using woodbury formula from linear_algebra module
            xi0 = SMW<>().solve(invP_[i], U_i, XtWX(), V_i, bi);
            
            x_new.block(i*n_basis(), 0, n_basis(),1) = xi0.head(n_basis());
            x_new.block((i+m_)*n_basis(),0, n_basis(),1) = xi0.tail(n_basis()); 
            
            r.block(n_basis()*i,0, n_basis(),1) -=   ((-PsiTD_[i]*Psi_[i]) * xi0.head(n_basis()) +
                                           			  lambda_D()*pde_.stiff().transpose()*xi0.tail(n_basis()));
  
            // correzione cov fuori
            r.block(n_basis()*(m_+i),0, n_basis(),1) -= (lambda_D()*pde_.stiff()*xi0.head(n_basis()) +
                                                          lambda_D()*pde_.mass()*xi0.tail(n_basis()) );
        }
		
        // std::chrono::duration<double> _duration = std::chrono::high_resolution_clock::now() - _start;
        // std::cout << "-		inizializzazione: " << _duration.count() << std::endl;
		// _start = std::chrono::high_resolution_clock::now();
        
        // correzione covariate
		for(std::size_t i = 0; i < m_; i++){
			r.block(i*n_basis(),0, n_basis(), 1) -= aux(x_new, i); 
		}
	
		// _duration = std::chrono::high_resolution_clock::now() - _start;
        // std::cout << "-		residuo: " << _duration.count() << std::endl;
		
        // store result of smoothing
        f_ = x_new.head(m_*n_basis());      // f0        
        g_ = x_new.tail(m_*n_basis());      // g0

        beta_ = invXtWX().solve(X().transpose() * (y_ - mPsi_ * f_)); 
        
        // std::cout << "nu:\n" << beta_ << std::endl;
        beta_coeff_ = F_*beta_;
        // std::cout << "beta_coeff:\n" << beta_coeff_ << std::endl;
        alpha_coeff_ = T_*beta_.tail(m_*qV); 
        // std::cout << "alpha_coeff:\n" << alpha_coeff_ << std::endl;
        
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
      
        DVector<double> ri = DMatrix<double>::Zero(2*n_basis(),1);
        DVector<double> zi = DMatrix<double>::Zero(2*n_basis(),1);
        // _start = std::chrono::high_resolution_clock::now();

        // iterative scheme for minimization of functional 
        while (k < max_iter_ && !exit_) /* ischia pag 25 */ {
            // auto __start = std::chrono::high_resolution_clock::now();
            for(std::size_t i = 0; i < m_; i++){
                    bi = DMatrix<double>::Zero(2*n_basis(), 1);
                    xi0 = DMatrix<double>::Zero(2*n_basis(), 1); 

                    U_i = DMatrix<double>::Zero(2*n_basis(), q()); // re-initialization
                    V_i = DMatrix<double>::Zero(q(), 2*n_basis());
                                      
                    // valutare implementazione di lmbQ(yi)
                    bi.block(0,0,n_basis(),1) = r.block(i*n_basis(), 0, n_basis(), 1) ; 
                    bi.block(n_basis(), 0, n_basis(), 1) = r.block( (i+m_)*n_basis(), 0, n_basis(), 1) ;
                    
                    U_i.block(0, 0, n_basis(), p) = PsiTD_[i]*Wg(i);
                    U_i.block(0, p+i*qV, n_basis(), qV ) = PsiTD_[i]*Vp(i);

                    V_i.block(0, 0, p, n_basis()) =  Wg(i).transpose()*Psi_[i];
                    V_i.block(p+i*qV, 0, qV, n_basis()) = Vp(i).transpose()*Psi_[i]; 

                    // solve system (A_ + U_*(X^T*W_*X)*V_)x = b using woodbury formula from linear_algebra module
                    // auto ___start = std::chrono::high_resolution_clock::now();

                    zi = SMW<>().solve(invP_[i], U_i, XtWX(), V_i, bi);

                    // std::chrono::duration<double> ___duration = std::chrono::high_resolution_clock::now() - ___start;
        	        // std::cout << "-			costo SMW/unita: " << 
        			// 								___duration.count() << std::endl;
                    
                    x_new.block(n_basis()*i,0, n_basis(),1) += alpha(k)*zi.head(n_basis());  
                    x_new.block(n_basis()*(m_+i),0, n_basis(),1) += alpha(k)*zi.tail(n_basis());
                    
                    r.block(n_basis()*i,0, n_basis(),1) -=  alpha(k) * ((-PsiTD_[i]* Psi_[i]) * zi.head(n_basis()) +
                                                                         lambda_D()*pde_.stiff().transpose()*zi.tail(n_basis()));
                                                                                             
                    r.block(n_basis()*(m_+i),0, n_basis(),1) -= alpha(k)*(lambda_D()*pde_.stiff()*zi.head(n_basis()) +
                                                                            lambda_D()*pde_.mass()*zi.tail(n_basis()) );
                    
                    z.block(i*n_basis(),0, n_basis(),1) = zi.head(n_basis());  
                    z.block(n_basis()*(m_+i),0, n_basis(),1) = zi.tail(n_basis()); 
                    
            }

            // std::chrono::duration<double> __duration = std::chrono::high_resolution_clock::now() - __start;
        	// std::cout << "-		costo singola iter: " << __duration.count() << std::endl;
		
            f_ = x_new.topRows(n_basis()*m_);
            g_ = x_new.bottomRows(n_basis()*m_);
       
       		// __start = std::chrono::high_resolution_clock::now();

            beta_ = invXtWX().solve(X().transpose() * (y_ - mPsi_ * f_)); 

       		// __duration = std::chrono::high_resolution_clock::now() - __start;
        	// std::cout << "-		compute nu: " << __duration.count() << std::endl;
        	// __start = std::chrono::high_resolution_clock::now();

            beta_coeff_ = F_*beta_;
            alpha_coeff_ = T_*beta_.tail(m_*qV); 

            // __duration = std::chrono::high_resolution_clock::now() - __start;
        	// std::cout << "-		compute beta & alpha: " << __duration.count() << std::endl;

            // correzione covariate            
            for(std::size_t i = 0; i < m_; i++) {
                r.block(i*n_basis(),0, n_basis(), 1) -= aux(z, i); 
            }
            
            Jold = Jnew;
            Jnew = J(f_,g_);

            rcheck = r.norm() / b_.norm() < tol_res;
            Jcheck =  std::abs((Jnew-Jold)/Jnew) < tol_;
            exit_ = Jcheck && rcheck;

            k++;

        }

        // _duration = std::chrono::high_resolution_clock::now() - _start;
        // std::cout << "-		end while loop: " << _duration.count() << std::endl;
        
		std::cout << "iter: " << k << std::endl;
    
        // auto end = std::chrono::high_resolution_clock::now();
        // std::chrono::duration<double> duration = end - start;
        // std::cout << "- metodo solve(): " << duration.count() << std::endl;

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
    std::size_t q() const { return p+m_*qV; } // numero delle colonne di X
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
        DMatrix<double> Ip = {}; Ip.resize(qV,qV); Ip.setIdentity(); Ip = Ip/m_;
        DMatrix<double> Iqp = {}; Iqp.resize(p,p); Iqp.setIdentity();
        DMatrix<double> Z1 = DMatrix<double>::Zero(qV,p);
        DMatrix<double> Z2 = DMatrix<double>::Zero(p, m_*qV);
        DMatrix<double> Ip_loop = {}; Ip_loop.resize(qV, m_*qV);
        for (std::size_t i=0; i<m_; i++){ Ip_loop.block(0,i*qV,qV,qV) = Ip; }
        F_ = SparseBlockMatrix<double,2,2>(
            Z1.sparseView(), Ip_loop.sparseView(),
            Iqp.sparseView(), Z2.sparseView() ); 
        T_.resize(m_*qV,m_*qV); T_.setIdentity(); T_ = (m_-1.0)/m_ * T_;
        for (std::size_t i=0; i<m_; i++){
            for(std::size_t j=0; j<m_; j++){
                if(T_(i*qV,j*qV) == 0){ T_.block(i*qV,j*qV,qV,qV) = -Ip; }
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