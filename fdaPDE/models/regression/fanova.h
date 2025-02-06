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

#ifndef __FANOVA_H__
#define __FANOVA_H__

#include <fdaPDE/linear_algebra.h>
//#include <fdaPDE/linear_algebra/kronecker_product.h>
#include <fdaPDE/pde.h>
#include <fdaPDE/utils.h>
#include <Eigen/Dense> // for leftCols(r_); r_ # of cols
#include <Eigen/QR>

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

template <typename SolutionPolicy> class fANOVABase;

template <typename SolutionPolicy>
class fANOVABase : public RegressionBase<fANOVABase<SolutionPolicy>, SpaceOnly>{
    public:
        using Base = RegressionBase<fANOVABase<SolutionPolicy>, SpaceOnly>;
        using Base::model;
        using Base::y_mask_;
        using Base::W_;
        using Base::XtWX_;
        using Base::invXtWX_;
        using Base::nan_mask_;
        using Base::pde_;
        using Base::n_nan_;
        using Base::lambda_D;   // smoothing parameter in space
        using Base::n_basis;    // number of spatial basis
        using Base::runtime;    // runtime model status

        using RegularizationType = SpaceOnly;

        IMPORT_REGRESSION_SYMBOLS;

        fANOVABase() = default;
        fANOVABase(const pde_ptr& pde, Sampling s) : Base(pde, s) { };
        fANOVABase(const pde_ptr& pde, Sampling s, bool same_locs_value) : Base(pde, s, same_locs_value) { };

        void init_sampling(bool forced = true) {
        // switch (s) {
        // case Sampling::pointwise: {   // data sampled at general locations p_1, p_2, ... p_n
            // query pde to evaluate functional basis at given locations
            Psi_.resize(data_.size());
            PsiTD_.resize(data_.size());
            for(std::size_t i=0; i<data_.size(); i++){
                auto basis_evaluation = this->model().pde().eval_basis(core::eval::pointwise, data_[i].template get<double>(LOCS_BLOCK));
                Psi_[i] = basis_evaluation->Psi;
                PsiTD_[i] = Psi_[i].transpose();
            }
        // } break;
        // }
        return;
        }

        void init_X() {
            set_n_locs_cum();
            m_ = data_.size();                                              // m_: number of patients
            p_ = data_[0].template get<double>(V_BLOCK).cols();             // p_: patient-specific covariates
            if(data_[0].template get<double>(W_BLOCK).cols()){
                r_ = data_[0].template get<double>(W_BLOCK).cols();
            } else {
                r_ = 0;          // r_: group-specific covariates
            }         
            q_ = r_ + p_;
            X_ = DMatrix<double>::Zero(N, q()); // q = r_+m*p_

            y_.resize(N,1);
            for(std::size_t i=0; i<m_; i++){
                y_.block(n_locs_cum[i],0,n_locs(i),1) = data_[i].template get<double>(Y_BLOCK);
                if(r_){
                    X_.block(n_locs_cum[i], 0, n_locs(i), r_) = data_[i].template get<double>(W_BLOCK);
                    X_.block(n_locs_cum[i], r_+i*p_, n_locs(i), p_) = data_[i].template get<double>(V_BLOCK);
                } else {
                    X_.block(n_locs_cum[i], i*p_, n_locs(i), p_) = data_[i].template get<double>(V_BLOCK);
                }
            }
            set_F_T();
        }
    
        void analyze_data() {

            // build multi-domain model design matrix 
            init_X();

            // initialize empty masks
            if (!y_mask_.size()) y_mask_.resize(N);
            if (!nan_mask_.size()) nan_mask_.resize(N); 

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

        DMatrix<double> lmbQ(const DMatrix<double>& x) const {
            DMatrix<double> v = X_.transpose() * W_ * x;             // X^\top*W*x
            DMatrix<double> z = invXtWX_.solve(v);                   // (X^\top*W*X)^{-1}*X^\top*W*x
            // compute W*x - W*X*z = W*x - (W*X*(X^\top*W*X)^{-1}*X^\top*W)*x = W(I - H)*x = Q*x
            return W_ * x - W_ * X_ * z;
        }

        double norm(const DMatrix<double>& op1, const DMatrix<double>& op2) const { return (op1 - op2).squaredNorm(); }

        //setters
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
            N = sum;
            return;
        }

        virtual void init_model(){};

        virtual void solve(){};

        void set_data(const std::vector<BlockFrame<double, int>>& data, bool reindex = false) { data_ = data; }

        // getters
        std::size_t q() const { return r_+m_*p_; }
        std::size_t n_locs(std::size_t i) const { return data_[i].template get<double>(LOCS_BLOCK).rows(); }
        
        const DVector<double>& f() const { return f_; };
        const DVector<double> f(std::size_t i) const {return f_.block(i*n_basis(), 0, n_basis(), 1); }
        const DVector<double> alpha() const { return alpha_coeff_; }
        const DVector<double> beta() const { return beta_coeff_; }
        const DVector<double> betanp() const { return beta_; }

        const DMatrix<double>& Wg(std::size_t i) const { return data_[i].template get<double>(W_BLOCK); } 
        const DMatrix<double>& Vp(std::size_t i) const { return data_[i].template get<double>(V_BLOCK); } 
        const DMatrix<double>& y(std::size_t i) const { return data_[i].template get<double>(Y_BLOCK); } 
        const DMatrix<double>& X() const { return X_; }  

        virtual ~fANOVABase() = default;

    protected:
        
        DVector<double> b_ {};      // right hand side of problem's linear system (1 x 2N vector)
        DMatrix<double> y_ {}; 
        DMatrix<double> X_ {};      // N total obs * r_ group-specific covariates
        
        std::size_t N;              // N: total observations (N=n*m)
        std::size_t m_;             // m: number of levels
        std::size_t p_;             // p_: number of level-specific covariatess
        std::size_t r_;             // r_:  group-specific covariates
        std::size_t q_;             // q_: "input" DesignMatrix columns -> q_ - p_ = r_f

        SpMatrix<double> I_;        // N x N sparse identity matrix 

        SpMatrix<double> mPsi_;
        SpMatrix<double> mPsiTD_;

        std::vector<SpMatrix<double>> Psi_;             // override of Psi_
        std::vector<SpMatrix<double>> PsiTD_;           // override of Psi_
    
        DMatrix<double> alpha_coeff_;                   // coefficients
        DMatrix<double> beta_coeff_;                    // beta coefficients of the original model
        SparseBlockMatrix<double, 2, 2> F_ {};          // matrix used for linear transformation of coefficients
        DMatrix<double> T_;
        DVector<int> n_locs_cum;

        std::vector<BlockFrame<double, int>> data_;     // vector of dataframes

        void init_mPsi() { 

            mPsi_.resize(N, n_basis()*m_);

            std::vector<fdapde::Triplet<double>> triplet_list;
            triplet_list.reserve(N);
            std::size_t sum_row = 0;
            std::size_t sum_col = 0;
            for (std::size_t i = 0; i < m_; ++i){
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
};


// monolithic specialization

template <typename SolutionPolicy> class fANOVA;

template <> 
class fANOVA<monolithic> : public fANOVABase<fANOVA<monolithic>>{
    
    public:
        using fANOVABase::Base;
        using Base::model;
        using Base::y_mask_;
        using Base::XtWX_;
        using Base::invXtWX_;
        using Base::nan_mask_;
        using Base::pde_;
        using Base::n_nan_;
        using Base::lambda_D;   // smoothing parameter in space
        using Base::n_basis;    // number of spatial basis
        using Base::runtime;    // runtime model status

        using fANOVABase::RegularizationType;
        IMPORT_REGRESSION_SYMBOLS;

        using fANOVABase::n_locs;
        using fANOVABase::lmbQ;
        using fANOVABase::q;
        using fANOVABase::f;
        using fANOVABase::beta;
        using fANOVABase::alpha;
        using fANOVABase::init_mPsi;

        fANOVA() = default;
        fANOVA(const pde_ptr& pde, Sampling s) : fANOVABase(pde, s){};
        fANOVA(const pde_ptr& pde, Sampling s, bool same_locs = false) : fANOVABase(pde, s, same_locs){};

        const SpMatrix<double> R0() const { return Kronecker(I_, pde_.mass()); }
        const SpMatrix<double> R1() const { return Kronecker(I_, pde_.stiff()); }

        void init_model(){
            I_.resize(m_,m_);
            I_.setIdentity();
            
            // Kronecker products with the identity
            init_mPsi(); 
            mPsiTD_ = mPsi_.transpose();   

            if (runtime().query(runtime_status::is_lambda_changed)) {
                
                A_ = SparseBlockMatrix<double, 2, 2>(
                        -mPsiTD_  * W_ * mPsi_, lambda_D() * R1().transpose(), 
                        lambda_D() * R1(),        lambda_D() * R0()            );

                invA_.compute(A_);

                // prepare rhs of linear system 
                b_.resize(A_.rows());
                for( int i=0; i < m_; ++i){
                    b_.block((m_+i)*n_basis(), 0, n_basis(), 1) = lambda_D() * u(); 
                }
                return;
            }

            if (runtime().query(runtime_status::require_W_update)) {
                // adjust north-west block of matrix A_ only
                A_.block(0, 0) = -mPsiTD_ * W_ * mPsi_;     
                invA_.compute(A_);
                return;
            }
        }
        
        void solve(){

            fdapde_assert(y_.rows() != 0);
            DVector<double> sol;

            // parametric case
            // update rhs of SR-PDE linear system
            b_.block(0, 0, m_*n_basis(), 1) = -mPsiTD_ * W_ * lmbQ(y_);  
            
            // matrices U and V for application of woodbury formula
            U_ = DMatrix<double>::Zero(2 * m_ * n_basis(), q());
            U_.block(0, 0, m_* n_basis(), q()) =  mPsiTD_ * W_ * X_; 

            V_ = DMatrix<double>::Zero(q(), 2 * m_ * n_basis());
            V_.block(0, 0, q(), m_ * n_basis()) = X_.transpose() * W_ * mPsi_; 

            // solve system (A_ + U_*(X^T*W_*X)*V_)x = b using woodbury formula from linear_algebra module
            sol = SMW<>().solve(invA_, U_, XtWX_, V_, b_);
            
            // store result of smoothing
            f_ = sol.head(m_*n_basis());
            
            // nu
            beta_ = invXtWX_.solve(X_.transpose() * W_ * (y_ - mPsi_ * f_)); 

            beta_coeff_ = F_ * beta_;
            alpha_coeff_ = T_*beta_.tail(m_*p_); 

            // store PDE misfit
            g_ = sol.tail(m_*n_basis());
            return;
        }
    
    virtual ~fANOVA() = default;
    
    protected:

        SparseBlockMatrix<double, 2, 2> A_ {};         // system matrix of non-parametric problem (2N x 2N matrix) (in iter P_ deve diventare A_)
        fdapde::SparseLU<SpMatrix<double>> invA_ {};   // factorization of matrix A          
};

// CwiseNullaryExpr  
template<class ArgType> class U_view_functor;

template<class ArgType>
struct U_view_helper {
  typedef Eigen::Matrix<typename ArgType::Scalar,
                 ArgType::SizeAtCompileTime,
                 ArgType::SizeAtCompileTime,
                 Eigen::ColMajor,
                 ArgType::MaxSizeAtCompileTime,
                 ArgType::MaxSizeAtCompileTime> MatrixType;
};

template <class ArgType>
Eigen::CwiseNullaryOp< U_view_functor<ArgType>, typename U_view_helper<ArgType>::MatrixType >
    U_view(const Eigen::MatrixBase<ArgType>& arg, std::size_t j,  std::size_t q, std::size_t p)
{ 
    typedef typename U_view_helper<ArgType>::MatrixType MatrixType;
    return MatrixType::NullaryExpr(2*arg.rows(), std::floor((q-p) + p*(arg.cols()/q)), U_view_functor<ArgType>(arg.derived(), j, q, p));
};

template<class ArgType>
class U_view_functor {
  const ArgType &m_arg;
  std::size_t m_k;
  std::size_t m_q;
  std::size_t m_p;
public:
  U_view_functor(const ArgType& arg, std::size_t k, std::size_t q, std::size_t p) : m_arg(arg), m_k(k), m_q(q), m_p(p){}
  //const
  const typename ArgType::Scalar operator() (Eigen::Index row, Eigen::Index col) const { //&
    
    if(row > (m_arg.rows()-1) && row < 2*m_arg.rows()) return 0.;
    
    if (col < (m_q - m_p)) 
        return m_arg(row, col + m_k*m_q);
    else if( col > ( (m_q - m_p) + m_k*m_p - 1) &&
              col <  ( (m_q - m_p) + (m_k+1)*m_p )) 
         return   m_arg(row, col + m_k*(m_q-m_p));
    return 0.;
    }
};

// iterative specialization
template <>
class fANOVA<iterative> : public fANOVABase<fANOVA<iterative>> {

    public:
        using fANOVABase::Base;
        using Base::model;
        using Base::y_mask_;
        using Base::XtWX_;
        using Base::invXtWX_;
        using Base::nan_mask_;
        using Base::pde_;
        using Base::n_nan_;
        using Base::lambda_D;   // smoothing parameter in space
        using Base::n_basis;    // number of spatial basis
        using Base::runtime;    // runtime model status

        using fANOVABase::RegularizationType;
        IMPORT_REGRESSION_SYMBOLS;

        using fANOVABase::n_locs;
        using fANOVABase::lmbQ;
        using fANOVABase::q;
        using fANOVABase::f;
        using fANOVABase::beta;
        using fANOVABase::alpha;
        using fANOVABase::Vp;
        using fANOVABase::Wg;
        using fANOVABase::init_mPsi;
        using fANOVABase::X;

        fANOVA() = default;
        fANOVA(const pde_ptr& pde, Sampling s, bool same_locs = false) : fANOVABase(pde, s, same_locs) {};

        void init_model(){

            init_mPsi(); 
            mPsiTD_ = mPsi_.transpose();   

            b_ = DMatrix<double>::Zero(2*n_basis()*m_, 1);

            invA_.resize(data_.size());
            if(mem_){A_v.resize(data_.size());}
            _U = DMatrix<double>::Zero(n_basis(), m_*q_);   // U_tilde in 2*n_basis x ( (q-p) + p ) * m = 2*n_basis x m q
                                                       
            invG.resize(data_.size());
            
            // initialization
            for (std::size_t i = 0; i < m_; i++){

                if(!same_locs_value || (same_locs_value && i==0)){
                    A_ = SparseBlockMatrix<double, 2, 2>(
                    -PsiTD_[i]*W(i)*Psi_[i],                     lambda_D() * pde_.stiff().transpose(),
                    lambda_D() * pde_.stiff(),             lambda_D() * pde_.mass()                 );
                }   

                if(mem_){A_v[i] = A_;}
                
                if(same_locs_value && i!=0){
                    invA_[i] = invA_[0];
                }else{
                    invA_[i].compute(A_);                    
                }
                
                if(r_){
                    _U.block(0, i*q_ , n_basis(), r_) = PsiTD_[i]*Wg(i);
                }
                _U.block(0, (i+1)*q_ - p_, n_basis(), p_) = PsiTD_[i]*Vp(i);
                
                invG[i].compute(XtWX_ + U_view(_U, i, q_, p_).transpose() * invA_[i].solve(U_view(_U, i, q_, p_)));  
            }

            set_F_T();
            
            return;
        }

    
        void solve(){
        
            fdapde_assert(y_.rows() != 0);
            
            DVector<double> x_new = DMatrix<double>::Zero(2*n_basis()*m_, 1);
            DVector<double> x_old = DMatrix<double>::Zero(2*n_basis()*m_, 1);
            b_.block(0, 0, n_basis()*m_, 1) = -mPsiTD_ * lmbQ(y_); 
            
            DVector<double> r = b_; 
            double Jnew;
            double Jold;
            
            DVector<double> bi = DMatrix<double>::Zero(2*n_basis(), 1);
            DVector<double> ri = DMatrix<double>::Zero(2*n_basis(),1);
            DVector<double> zi = DMatrix<double>::Zero(2*n_basis(),1);
            x_new_vec.resize(2*n_basis()*m_); 
            
            for (std::size_t i = 0; i < m_; i++){

                bi.block(0,0,n_basis(),1) = b_.block(i*n_basis(), 0, n_basis(), 1);
                            
                // solve system (A_ + U_*(X^T*W_*X)*V_)x = b using woodbury formula from linear_algebra module               
                DMatrix<double> y = invA_[i].solve(bi);   
                DMatrix<double> t = invG[i].solve(U_view(_U, i, q_, p_).transpose()*y);
                zi = y - invA_[i].solve(U_view(_U, i, q_, p_) * t);

                x_new.block(i*n_basis(), 0, n_basis(),1) = zi.head(n_basis());
                x_new.block((i+m_)*n_basis(),0, n_basis(),1) = zi.tail(n_basis());
                x_new_vec[i] = zi;
                
            
                r.block(n_basis()*i,0, n_basis(),1) -=   ((-PsiTD_[i]*Psi_[i]) * zi.head(n_basis()) +
                                                        lambda_D()*pde_.stiff().transpose()*zi.tail(n_basis()));
    
                r.block(n_basis()*(m_+i),0, n_basis(),1) -= (lambda_D()*pde_.stiff()*zi.head(n_basis()) +
                                                            lambda_D()*pde_.mass()*zi.tail(n_basis()) );
            }
            
            bi = DMatrix<double>::Zero(2*n_basis(), 1);

            // covariate correction
            DMatrix<double> u = DMatrix<double>::Zero(q(),1);
            for(std::size_t i = 0; i < m_; ++i){
                if(r_){
                    u.block(0, 0, r_, 1) += Wg(i).transpose()*Psi_[i]*x_new.block(i*n_basis(), 0, n_basis(), 1);    
                }
                u.block(i*p_+r_, 0, p_, 1) = Vp(i).transpose()*Psi_[i]*x_new.block(i*n_basis(), 0, n_basis(), 1);
            }
            DMatrix<double> w = invXtWX().solve(u);         
        
            for(std::size_t i = 0; i < m_; i++){
                if(r_){
                    r.block(i*n_basis(),0, n_basis(), 1) -= Psi_[i].transpose()*(Wg(i)*w.block(0, 0, r_, 1) + 
                                                                                Vp(i)*w.block(i*p_+r_, 0, p_, 1)); 
                } else {
                     r.block(i*n_basis(),0, n_basis(), 1) -= Psi_[i].transpose()*(Vp(i)*w.block(i*p_, 0, p_, 1)); 
                }
            }
        
            // store result of smoothing
            f_ = x_new.head(m_*n_basis());      // f0        
            g_ = x_new.tail(m_*n_basis());      // g0

            // nu
            beta_ = invXtWX().solve(X().transpose() * (y_ - mPsi_ * f_)); 
            
            beta_coeff_ = F_*beta_;
            alpha_coeff_ = T_*beta_.tail(m_*p_); 
    

            DVector<double> z;
            z = DMatrix<double>::Zero(r.rows(), 1);

            Jnew = J(f_,g_);
            Jold = 2 * Jnew;
        
            // iteration loop
            std::size_t k = 1;                                      // iteration number    
            bool rcheck = r.norm() / b_.norm() < tol_res;           // stop by residual    
            bool Jcheck =  std::abs((Jnew-Jold)/Jnew) < tol_;       // stop by J
            bool exit_ = Jcheck && rcheck;

            // 'memory' is the GMRES parameter memory, if memory == 0 then we have the base solver

            // iterative scheme for minimization of functional 
            while (k < max_iter_ && !exit_)  {
                for(std::size_t i = 0; i < m_; i++){       

                    bi.block(0,0,n_basis(),1) = r.block(i*n_basis(), 0, n_basis(), 1) ; 
                    bi.block(n_basis(), 0, n_basis(), 1) = r.block( (i+m_)*n_basis(), 0, n_basis(), 1);

                    // SMW 
                    DMatrix<double> y = invA_[i].solve(bi);   
                    DMatrix<double> t = invG[i].solve(U_view(_U, i, q_, p_).transpose()*y);
                    zi = y -  invA_[i].solve(U_view(_U, i, q_, p_) * t);

                    x = x_new_vec[i]; // this line is needed for mem_=0 optimization

                    if(mem_){

                        int A_rows = A_v[i].rows();
                        double res_norm = zi.norm();
                        x = x_new_vec[i];       // initial solution x0

                        // Hessenberg matrix and Krylov vector
                        DMatrix<double> V = DMatrix<double>::Zero(A_rows, mem_ + 1);        // orthonormal basis
                        DMatrix<double> H = DMatrix<double>::Zero(mem_ + 1, mem_);          // Hessenberg matrix

                        V.col(0) = zi / res_norm;           // first orthonormal vector

                        DVector<double> e1 = DMatrix<double>::Zero(mem_ + 1, 1); 
                        e1(0) = res_norm;

                        // GMRES iterations
                        for (int j_col = 0; j_col < mem_; ++j_col) {

                            // new Krylov vector
                            DVector<double> w = A_v[i] * V.col(j_col); 

                            // Arnoldi orthonormalization
                            for (int i_row = 0; i_row <= j_col; ++i_row) {
                                H(i_row, j_col) = V.col(i_row).dot(w);
                                w -= H(i_row, j_col) * V.col(i_row);
                            }
                            H(j_col + 1, j_col) = w.norm();

                            // break if vector is almost zero
                            if (H(j_col + 1, j_col) < 10e-6) break;

                            V.col(j_col + 1) = w / H(j_col + 1, j_col);

                            DVector<double> partial_sol = H.block(0, 0, j_col + 2, j_col + 1)
                                                    .colPivHouseholderQr()
                                                    .solve(e1.head(j_col + 2));

                            double res_norm = (e1.head(j_col + 2) - H.block(0, 0, j_col + 2, j_col + 1) * partial_sol).norm();
                            
                            // update
                            if (res_norm < 10e-2) {
                                x += V.leftCols(j_col + 1) * partial_sol; 
                            }
                        }

                        // solution after m iterations
                        DVector<double> partial_sol = H.block(0, 0, mem_ + 1, mem_)
                                                .colPivHouseholderQr()
                                                .solve(e1);

                        x += V.leftCols(mem_) * partial_sol;
                    }

                    x_new.block(n_basis()*i,0, n_basis(),1) = alpha(k) * x.head(n_basis());  
                    x_new.block(n_basis()*(m_+i),0, n_basis(),1) = alpha(k) * x.tail(n_basis());
                    
                    r.block(n_basis()*i,0, n_basis(),1) -=  alpha(k) * ((-PsiTD_[i]* Psi_[i]) * zi.head(n_basis()) +
                                                                        lambda_D()*pde_.stiff().transpose()*zi.tail(n_basis()));
                                                                                            
                    r.block(n_basis()*(m_+i),0, n_basis(),1) -= alpha(k)*(lambda_D()*pde_.stiff()*zi.head(n_basis()) +
                                                                            lambda_D()*pde_.mass()*zi.tail(n_basis()) );
                    
                    z.block(i*n_basis(),0, n_basis(),1) = zi.head(n_basis());  
                    z.block(n_basis()*(m_+i),0, n_basis(),1) = zi.tail(n_basis());
                        
                }

                f_ = x_new.topRows(n_basis()*m_);
                g_ = x_new.bottomRows(n_basis()*m_);

                beta_ = invXtWX().solve(X().transpose() * (y_ - mPsi_ * f_)); 

                beta_coeff_ = F_*beta_;
                alpha_coeff_ = T_*beta_.tail(m_*p_); 

                // covariate correction
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

            // number of iterations
            std::cout << "iter: " << k << std::endl;
        
            return;
        }

        // getters
        double alpha(std::size_t k) const { return alpha_; }        // fixed to 1 

        //setters
        void set_GMRES_params(int memory){ mem_ = memory; }
        void set_tolerance(double tol) { tol_ = tol; }
        void set_max_iter(std::size_t max_iter) { max_iter_ = max_iter; }

        virtual ~fANOVA() = default;
    
    protected:
        
        SparseBlockMatrix<double, 2, 2> A_ {};         // system matrix of non-parametric problem (2N x 2N matrix)
        std::vector<fdapde::SparseLU<SpMatrix<double>>> invA_ {};
        std::vector<DMatrix<double>> A_v {};   
        std::vector<DVector<double>> x_new_vec {}; 
        DVector<double> x; 
    
        using DenseSolver  = Eigen::PartialPivLU<DMatrix<double>>;
        std::vector<DenseSolver> invG;
        DMatrix<double> _U;

        // iterative scheme parameters 
        double tol_ = 1e-4;             // tolerance (stopping criterion)
        double tol_res = 1e-8;  
        std::size_t max_iter_ = 10;     // maximum number of iteration
        double alpha_ = 1.;             

        // Anderon's accelerator parameters
        int mem_ = 0; 

        double J(const DMatrix<double>& f, const DMatrix<double>& g) const{
            DMatrix<double> fhat = mPsi_ * f; 
            return (y_ - X() * beta_ - fhat).squaredNorm() + lambda_D()*g.squaredNorm(); 
        }

        const DiagMatrix<double> W(std::size_t i) const {return DVector<double>::Ones(n_locs(i)).asDiagonal(); };

};

}   // namespace models
}   // namespace fdapde


#endif   // __MIXED_SRPDE_H__