
#ifndef KGeneralizedMinimalResidual_HH__
#define KGeneralizedMinimalResidual_HH__

#include "KFMLinearAlgebraDefinitions.hh"
#include "KFMMatrixOperations.hh"
#include "KFMMatrixVectorOperations.hh"
#include "KFMVectorOperations.hh"
#include "KGeneralizedMinimalResidualState.hh"
#include "KMatrix.hh"
#include "KSimpleMatrix.hh"
#include "KSimpleVector.hh"
#include "KSquareMatrix.hh"
#include "KVector.hh"

#include <cmath>

namespace KEMField
{

/*
*
*@file KGeneralizedMinimalResidual.hh
*@class KGeneralizedMinimalResidual
*@brief
*@details: only implemented for real types, do not use with ValueType = std::complex<...>

* GMRES implementation with externally controlled restart
* Algorithm is from the paper:
*
* "GMRES: A Generalized Minimal Residual Algortihm For Solving Nonsymmetric Linear Systems"
* SIAM J. Sci. Stat. Comput. Vol 7, No 3, July 1986
* Youcef Sadd and Martin Schultz
*
*<b>Revision History:<b>
*Date Name Brief Description
*Sat Feb  1 10:20:58 EST 2014 J. Barrett (barrettj@mit.edu) First Version
*
*/

template<typename ValueType> class KGeneralizedMinimalResidual
{
  public:
    using Matrix = KSquareMatrix<ValueType>;
    using Vector = KVector<ValueType>;

    using KSimpleMatrixType = KSimpleMatrix<ValueType>;
    using KSimpleVectorType = KSimpleVector<ValueType>;

    KGeneralizedMinimalResidual(const Matrix& A, Vector& x, const Vector& b) : fDim(A.Dimension()), fA(A), fX(x), fB(b)
    {
        fR.resize(fDim);
        fW.resize(fDim);
        fJ = 0;
        fUseSVD = true;  //default is to use SVD for solution calculation
        fExternalStateSet = false;
    };

    virtual ~KGeneralizedMinimalResidual() = default;
    ;

    static std::string Name()
    {
        return std::string("gmres");
    }
    std::string NameLabel()
    {
        return std::string("gmres");
    }

    const KGeneralizedMinimalResidualState<ValueType>& GetState() const;
    void SetState(const KGeneralizedMinimalResidualState<ValueType>& state);

    void Initialize();
    void ResetAndInitialize();
    void AugmentKrylovSubspace();
    void UpdateSolution();

    void GetResidualNorm(double& norm)
    {
        norm = fResidualNorm;
    };

    void CoalesceData(){};
    void Finalize(){};

    unsigned int Dimension() const
    {
        return fDim;
    };

    void SetResidualVector(const Vector&);
    void GetResidualVector(Vector&) const;

    //user can indicate a preference for how the solution is calculated, default is SVD
    void UseSingularValueDecomposition()
    {
        fUseSVD = true;
    };
    void UseBackSubstitution()
    {
        fUseSVD = false;
    };

  private:
    //inner product for vectors
    double InnerProduct(const Vector& a, const Vector& b);
    void ReconstructState();

    //data
    unsigned int fDim;

    //matrix system
    const Matrix& fA;
    Vector& fX;
    const Vector& fB;

    //inner iteration count
    unsigned int fJ;
    double fResidualNorm;

    //intial residual vector
    KSimpleVectorType fR;

    //columns of H, the minimization matrix
    std::vector<KSimpleVectorType> fH;

    //the krylov subspace basis vectors
    std::vector<KSimpleVectorType> fV;

    KSimpleVectorType fY;  //solution vector to the minimization problem
    KSimpleVectorType fP;  //right hand side of minimization problem
    KSimpleVectorType fW;  //workspace

    KSimpleVectorType fC;  //Givens rotation cosines
    KSimpleVectorType fS;  //Givens rotation sines

    bool fUseSVD;

    bool fExternalStateSet;
    mutable KGeneralizedMinimalResidualState<ValueType> fState;
};

template<typename ValueType> void KGeneralizedMinimalResidual<ValueType>::Initialize()
{
    if (fExternalStateSet)  //we have data from a previous run of the same process
    {
        double inf_norm = 0;
        for (unsigned int i = 0; i < fDim; i++) {
            if (std::fabs(fX[i]) > inf_norm) {
                inf_norm = std::fabs(fX[i]);
            };
        };

        if (InnerProduct(fX, fX) < 1e-14) {
            //current solution guess is probably zero vector
            //so load up the old checkpoint and go
            ReconstructState();
            return;
        }
        else {
            //first compute the residual of current solution
            ResetAndInitialize();
            //compare residual to previous checkpoint
            if (fResidualNorm > fState.GetResidualNorm()) {
                //current residual is worse than old checkpointed solution, so load it into memory
                ReconstructState();
                return;
            }
        }
    }

    //no previous state to load, go ahead
    ResetAndInitialize();
}


template<typename ValueType> void KGeneralizedMinimalResidual<ValueType>::ResetAndInitialize()
{

    //clear out any old data
    fH.resize(0);
    fV.resize(0);
    fY.resize(0);
    fP.resize(0);
    fC.resize(0);
    fS.resize(0);

    //start iteration count at zero
    fJ = 0;

    fA.Multiply(fX, fR);
    for (unsigned int i = 0; i < fDim; i++) {
        fR[i] = fB(i) - fR[i];
    }

    //now compute residual norm
    double norm = std::sqrt(InnerProduct(fR, fR));
    double inv_norm = 1.0 / norm;

    //set v_0 = r/|r|
    fV.push_back(KSimpleVectorType());
    fV[fJ].resize(fDim);
    for (unsigned int i = 0; i < fDim; i++) {
        fV[fJ][i] = fR[i] * inv_norm;
    }

    //set p_0 = |r|
    fP.push_back(norm);
    fResidualNorm = norm;
}


template<typename ValueType> void KGeneralizedMinimalResidual<ValueType>::AugmentKrylovSubspace()
{
    //compute w = A*v_j
    fV.push_back(KSimpleVectorType());
    fV[fJ + 1].resize(fDim);
    fA.Multiply(fV[fJ], fW);

    //append new column to fH
    fH.push_back(KSimpleVectorType());
    fH[fJ].resize(fJ + 1);

    //fH: note the change in row-column index convention!!
    unsigned int row;
    //compute the fJ-th column of fH
    for (row = 0; row <= fJ; row++) {
        double h = InnerProduct(fW, fV[row]);
        fH[fJ][row] = h;

        //gram-schmidt process: subtract off component parallel to subspace vector v_i
        for (unsigned int n = 0; n < fDim; n++) {
            fW[n] -= h * fV[row][n];
        };
    }

    //set v_j+1 = w/h
    double beta = std::sqrt(InnerProduct(fW, fW));
    double inv_beta = 1.0 / beta;
    for (unsigned int n = 0; n < fDim; n++) {
        fV[fJ + 1][n] = fW[n] * inv_beta;
    }

    //now apply all previous Given's rotations to the new column of fH
    for (row = 0; row < fJ; row++) {
        double a = fH[fJ][row];
        double b = fH[fJ][row + 1];
        fH[fJ][row] = fC[row] * a + fS[row] * b;
        fH[fJ][row + 1] = -1.0 * fS[row] * a + fC[row] * b;
    }

    //compute the newest rotation
    double gamma = std::sqrt((fH[fJ][fJ]) * (fH[fJ][fJ]) + beta * beta);
    fC.push_back(fH[fJ][fJ] / gamma);
    fS.push_back(beta / gamma);

    //apply newest rotation to last element of column
    //note the element 1 past the end (beta) is zero-ed out by this rotation
    fH[fJ][fJ] = gamma;

    //apply givens rotation to fP
    fP.push_back(-1.0 * fS[fJ] * fP[fJ]);
    fP[fJ] = fC[fJ] * fP[fJ];

    //update residual norm
    fResidualNorm = std::fabs(fP[fJ + 1]);

    //increment iteration count
    fJ++;
}

template<typename ValueType> void KGeneralizedMinimalResidual<ValueType>::UpdateSolution()
{
    if (fJ != 0) {
        //clear out old values of fX
        for (unsigned int i = 0; i < fDim; i++) {
            fX[i] = 0.0;
        }

        //the controller has decided that a restart is needed, or we have converged
        //so we compute the solution vector

        //create workspace needed to solve the minimization problem with SVD
        kfm_matrix* fSVD_a = kfm_matrix_calloc(fJ, fJ);
        kfm_matrix* fSVD_u = kfm_matrix_calloc(fJ, fJ);
        kfm_matrix* fSVD_v = kfm_matrix_calloc(fJ, fJ);
        kfm_vector* fSVD_s = kfm_vector_calloc(fJ);
        kfm_vector* fSVD_b = kfm_vector_calloc(fJ);
        kfm_vector* fSVD_x = kfm_vector_calloc(fJ);

        for (unsigned int row = 0; row < fJ; row++) {
            kfm_vector_set(fSVD_b, row, fP[row]);

            for (unsigned int col = 0; col < fJ; col++) {
                if (row <= col) {
                    //note that fH is transposed because we store it by columns
                    kfm_matrix_set(fSVD_a, row, col, fH[col][row]);
                }
                else {
                    kfm_matrix_set(fSVD_a, row, col, 0.0);
                }
            }
        }

        if (fUseSVD) {
            //perform singular value decomposition of H
            kfm_matrix_svd(fSVD_a, fSVD_u, fSVD_s, fSVD_v);

            //solve minimization problem H*y = p with SVD of H
            kfm_matrix_svd_solve(fSVD_u, fSVD_s, fSVD_v, fSVD_b, fSVD_x);
        }
        else {
            //solve this upper triangular system using back-substitution
            //Note: this is currently more memory intensive than it needs to be,
            //since we have allocated an entire NxN matrix to store the columns of H.
            //this could be made more efficient.
            kfm_matrix_upper_triangular_solve(fSVD_a, fSVD_b, fSVD_x);
        }

        //now compute the solution vector x
        for (unsigned int row = 0; row < fJ; row++) {
            double y = kfm_vector_get(fSVD_x, row);
            for (unsigned int n = 0; n < fDim; n++) {
                fX[n] += y * fV[row][n];
            }
        }

        //deallocate SVD solver space
        kfm_matrix_free(fSVD_a);
        kfm_matrix_free(fSVD_u);
        kfm_matrix_free(fSVD_v);
        kfm_vector_free(fSVD_s);
        kfm_vector_free(fSVD_b);
        kfm_vector_free(fSVD_x);
    }
}

template<typename ValueType>
double KGeneralizedMinimalResidual<ValueType>::InnerProduct(const Vector& a, const Vector& b)
{
    double result = 0.;

    for (unsigned int i = 0; i < fDim; i++) {
        result += a(i) * b(i);
    }

    return result;
}

template<typename ValueType> void KGeneralizedMinimalResidual<ValueType>::SetResidualVector(const Vector& v)
{
    fR.resize(v.Dimension());

    for (unsigned int i = 0; i < v.Dimension(); i++)
        fR[i] = v(i);
}

template<typename ValueType> void KGeneralizedMinimalResidual<ValueType>::GetResidualVector(Vector& v) const
{
    for (unsigned int i = 0; i < fR.Dimension(); i++)
        v[i] = fR(i);
}

template<typename ValueType>
const KGeneralizedMinimalResidualState<ValueType>& KGeneralizedMinimalResidual<ValueType>::GetState() const
{
    fState.SetDimension(fDim);
    fState.SetIterationCount(fJ);
    fState.SetResidualNorm(fResidualNorm);

    //have to handle x and b specially
    //fill temp vector with x fState
    KSimpleVectorType temp;
    temp.resize(fDim);
    for (unsigned int i = 0; i < fDim; i++) {
        temp[i] = fX(i);
    }
    fState.SetSolutionVector(&temp);

    for (unsigned int i = 0; i < fDim; i++) {
        temp[i] = fB(i);
    }
    fState.SetRightHandSide(&temp);

    fState.SetResidualVector(&fR);
    fState.SetMinimizationMatrix(&fH);
    fState.SetKrylovSpaceBasis(&fV);
    fState.SetMinimizationRightHandSide(&fP);
    fState.SetGivensRotationCosines(&fC);
    fState.SetGivensRotationSines(&fS);

    return fState;
}

template<typename ValueType>
void KGeneralizedMinimalResidual<ValueType>::SetState(const KGeneralizedMinimalResidualState<ValueType>& state)
{
    fState.SetDimension(state.GetDimension());
    fState.SetIterationCount(state.GetIterationCount());
    fState.SetResidualNorm(state.GetResidualNorm());

    const KSimpleVector<ValueType>* temp;

    temp = state.GetSolutionVector();
    fState.SetSolutionVector(temp);

    temp = state.GetRightHandSide();
    fState.SetRightHandSide(temp);

    temp = state.GetResidualVector();
    fState.SetResidualVector(temp);

    const std::vector<KSimpleVector<ValueType>>* h_temp;
    h_temp = state.GetMinimizationMatrix();
    fState.SetMinimizationMatrix(h_temp);

    const std::vector<KSimpleVector<ValueType>>* v_temp;
    v_temp = state.GetKrylovSpaceBasis();
    fState.SetKrylovSpaceBasis(v_temp);

    temp = state.GetMinimizationRightHandSide();
    fState.SetMinimizationRightHandSide(temp);

    temp = state.GetGivensRotationCosines();
    fState.SetGivensRotationCosines(temp);

    temp = state.GetGivensRotationSines();
    fState.SetGivensRotationSines(temp);

    fExternalStateSet = true;
}


template<typename ValueType> void KGeneralizedMinimalResidual<ValueType>::ReconstructState()
{
    if (fExternalStateSet) {
        fDim = fState.GetDimension();
        fJ = fState.GetIterationCount();
        fResidualNorm = fState.GetResidualNorm();

        const KSimpleVectorType* temp;
        temp = fState.GetSolutionVector();
        for (unsigned int i = 0; i < temp->size(); i++) {
            fX[i] = (*temp)(i);
        };

        temp = fState.GetResidualVector();
        fR.resize(temp->size());
        for (unsigned int i = 0; i < temp->size(); i++) {
            fR[i] = (*temp)(i);
        };

        //clear out fH
        for (unsigned int i = 0; i < fH.size(); i++) {
            fH[i].clear();
        };
        //fill up with new data
        std::vector<KSimpleVectorType> h_temp;
        fState.GetMinimizationMatrix(&h_temp);
        fH.resize(h_temp.size());
        for (unsigned int i = 0; i < h_temp.size(); i++) {
            unsigned int s = h_temp[i].size();
            fH[i].resize(s);
            for (unsigned int j = 0; j < s; j++) {
                fH[i][j] = h_temp[i][j];
            }
        }

        //clear out fV
        for (unsigned int i = 0; i < fV.size(); i++) {
            fV[i].clear();
        };
        //fill up with new data
        std::vector<KSimpleVectorType> v_temp;
        fState.GetKrylovSpaceBasis(&v_temp);
        fV.resize(v_temp.size());
        for (unsigned int i = 0; i < v_temp.size(); i++) {
            unsigned int s = v_temp[i].size();
            fV[i].resize(s);
            for (unsigned int j = 0; j < s; j++) {
                fV[i][j] = v_temp[i][j];
            }
        }

        temp = fState.GetMinimizationRightHandSide();
        fP.resize(temp->size());
        for (unsigned int i = 0; i < temp->size(); i++) {
            fP[i] = (*temp)(i);
        };

        temp = fState.GetGivensRotationCosines();
        fC.resize(temp->size());
        for (unsigned int i = 0; i < temp->size(); i++) {
            fC[i] = (*temp)(i);
        };

        temp = fState.GetGivensRotationSines();
        fS.resize(temp->size());
        for (unsigned int i = 0; i < temp->size(); i++) {
            fS[i] = (*temp)(i);
        };
    }
}


template<typename ValueType, typename Stream>
Stream& operator>>(Stream& s, KGeneralizedMinimalResidual<ValueType>& aData)
{
    s.PreStreamInAction(aData);

    KGeneralizedMinimalResidualState<ValueType> state;
    s >> state;
    aData.SetState(state);

    s.PostStreamInAction(aData);
    return s;
}


template<typename ValueType, typename Stream>
Stream& operator<<(Stream& s, const KGeneralizedMinimalResidual<ValueType>& aData)
{
    s.PreStreamOutAction(aData);

    s << aData.GetState();

    s.PostStreamOutAction(aData);

    return s;
}


}  // namespace KEMField

#endif /* KGeneralizedMinimalResidual_H__ */
