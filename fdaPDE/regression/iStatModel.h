#ifndef __I_STAT_MODEL__
#define __I_STAT_MODEL__

#include <memory>
#include <Eigen/LU>

#include "../core/utils/Symbols.h"
#include "../core/FEM/PDE.h"
using fdaPDE::core::FEM::PDE;
#include "Internals.h"
using fdaPDE::regression::internal::psi;

// abstract base class for any fdaPDE statistical model
template <unsigned int M, unsigned int N, unsigned int K, typename E>
class iStatModel {
protected:
  // helper to check if data member contains valid data
  template <typename T>
  bool isAlloc(const std::shared_ptr<T>& t) const { return t != nullptr && t->size() != 0; }

  // data of the problem
  const PDE<M,N,K,E>& pde_;                 // regularizing term and domain information
  double lambda_;                           // smoothing parameter
  std::shared_ptr<DVector<double>> z_{};    // vector of observations
  std::shared_ptr<DMatrix<double>> W_{};    // design matrix

  // notation:
  //   * n: number of observations
  //   * N: number of locations where data are observed
  //   * q: number of regressors
  
  // data coming from FEM module
  std::shared_ptr<SpMatrix<double>> R0_{};  // mass matrix, result of the discretization of the identity operator (N x N matrix)
  std::shared_ptr<SpMatrix<double>> R1_{};  // discretization matrix of the differential operator L (N x N matrix)
  std::shared_ptr<DMatrix<double>>  u_{};   // discretization of forcing term (1 x N vector)
  
  // n x N matrix \Psi = [\psi_{ij}] = \psi_j(p_i) whose ij-th entry is the evaluation of the j-th basis function at the i-th spatial location
  std::shared_ptr<SpMatrix<double>> Psi_{};
  
  // system matrix of non-parametric problem (2N x 2N matrix)
  //     | -\Psi^T*\Psi  \lambda*R1^T |
  // A = |                            |  
  //     | \lambda*R1    \lambda*R0   |
  std::shared_ptr<SpMatrix<double>> A_{};
  // right hand side of problem's linear system (1 x 2N vector)
  //     | -\Psi^T*Q*z |
  // b = |             |
  //     |  \lambda*u  |
  std::shared_ptr<DVector<double>>  b_{};

  std::shared_ptr<DMatrix<double>>  H_{};   // hat matrix of the problem: H_ = W*(W*T*W)^{-1}*W^T (n x n matrix)
  std::shared_ptr<DMatrix<double>>  WTW_{}; // q x q dense matrix W^T*W
  std::shared_ptr<DMatrix<double>>  Q_{};   // n x n projection matrix onto the orthogonal space of Im(W), Q_ = I - H_
  
  // partial LU (with pivoting) factorization of the dense (square invertible) q x q matrix W^T*W.
  Eigen::PartialPivLU<DMatrix<double>> invWTW_{};

  // problem solution
  std::shared_ptr<DVector<double>> f_;      // estimate of the spatial field (1 x N vector)
  std::shared_ptr<DVector<double>> beta_;   // estimate of the coefficient vector (1 x q vector)
  
public:
  // constructor
  iStatModel() = default;
  iStatModel(const PDE<M,N,K,E>& pde, double lambda) : pde_(pde), lambda_(lambda) {
    Psi_ = psi(pde_);
  };

  // set problem design matrix and precomputes all related quantites
  void setCovariates(const DMatrix<double>& W) {
    // store design matrix
    W_ = std::make_shared<DMatrix<double>>(W);
    // compute q x q dense matrix
    WTW_ = std::make_shared<DMatrix<double>>(W.transpose()*W);
    // compute the factorization of W^T*W
    invWTW_ = WTW_->partialPivLu();
    return;
  }
  // set observation vector
  void setObservations(const DVector<double>& z) {
    z_ = std::make_shared<DVector<double>>(z);
    return;
  }
  // set smoothing parameter
  void setLambda(double lambda) { lambda_ = lambda; }

  // getters
  std::size_t q() const { return isAlloc(W_) ? W_->cols() : 0; } // q, the number of covariates
  std::size_t loc() const { return pde_.domain().nodes(); }      // N, the number of locations
  std::size_t obs() const { return z_->rows(); }                 // n, the number of observations
  // pointers to problem data
  std::shared_ptr<DVector<double>> z() const { return z_; } // observation vector
  std::shared_ptr<DMatrix<double>> W() const { return W_; } // design matrix
  double lambda() const { return lambda_; } // smoothing parameter
  
  // pointer to projection matrix. Q is computed on demand only when it is needed (in general operations involving Q can be substituted
  // with the more efficient routine lmbQ())
  std::shared_ptr<DMatrix<double>> Q() {
    if(!isAlloc(Q_)){ // Q is computed on request since not needed in general
      // compute transpose of W once here
      DMatrix<double> Wt = W_->transpose();

      // compute hat matrix H = W*(W*W^T)^{-1}*W^T
      H_ = std::make_shared<DMatrix<double>>((*W_)*invWTW_.solve(Wt));
      // compute Q = I - H_
      Q_ = std::make_shared<DMatrix<double>>
	(DMatrix<double>::Identity(H_->rows(), H_->cols()) - *H_);
    }
    return Q_;
  }

  // pointer to q x q dense matrix W^T*W and its inverse
  std::shared_ptr<DMatrix<double>>  WTW() const { return WTW_; }
  Eigen::PartialPivLU<DMatrix<double>> invWTW() const { return invWTW_; }

  // pointers to FEM related quantites
  std::shared_ptr<SpMatrix<double>> R0() const { return pde_.R0(); }
  std::shared_ptr<SpMatrix<double>> R1() const { return pde_.R1(); }
  std::shared_ptr<DMatrix<double>>  u()  const { return pde_.force(); }
  
  std::shared_ptr<SpMatrix<double>> A()   const { return A_; }   // pointer to non-parametric part of the problem
  std::shared_ptr<DVector<double>>  b()   const { return b_; }   // pointer to rhs of the problem
  std::shared_ptr<SpMatrix<double>> Psi() const { return Psi_; } // pointer to N x N sparse matrix \Psi

  // pointers to problem solution
  std::shared_ptr<DVector<double>> f_hat() const { return f_; }
  std::shared_ptr<DVector<double>> beta_hat() const { return beta_; }
  
  // methods
  bool hasCovariates() const { return q() != 0; }

  // abstract part of the interface, must be implemented by concrete models

  // finds a solution to the smoothing problem.
  // After a call to smooth() all quantites related to the solution of the problem must contain valid data
  virtual void smooth() = 0;
  // computes \hat z, the fitted values at the observations' locations
  virtual DVector<double> fitted() const = 0;
  
  virtual ~iStatModel() = default;
};

// import all symbols from iStatModel interface in derived classes
#define IMPORT_STAT_MODEL_SYMBOLS(M,N,K,E)		\
  using iStatModel<M,N,K,E>::pde_;			\
  using iStatModel<M,N,K,E>::lambda_;			\
  using iStatModel<M,N,K,E>::z_;			\
  using iStatModel<M,N,K,E>::W_;			\
  using iStatModel<M,N,K,E>::R0_;			\
  using iStatModel<M,N,K,E>::R1_;			\
  using iStatModel<M,N,K,E>::u_;			\
  using iStatModel<M,N,K,E>::Psi_;			\
  using iStatModel<M,N,K,E>::A_;			\
  using iStatModel<M,N,K,E>::b_;			\
  using iStatModel<M,N,K,E>::H_;			\
  using iStatModel<M,N,K,E>::WTW_;			\
  using iStatModel<M,N,K,E>::Q_;			\
  using iStatModel<M,N,K,E>::invWTW_;			\
  using iStatModel<M,N,K,E>::f_;			\
  using iStatModel<M,N,K,E>::beta_;			\

#endif // __I_STAT_MODEL__