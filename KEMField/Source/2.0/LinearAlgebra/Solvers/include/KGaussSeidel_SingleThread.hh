#ifndef KGAUSSSEIDEL_SINGLETHREAD_DEF
#define KGAUSSSEIDEL_SINGLETHREAD_DEF

#include "KSimpleVector.hh"

namespace KEMField
{
  template <typename ValueType>
  class KGaussSeidel_SingleThread
  {
  public:
    typedef KSquareMatrix<ValueType> Matrix;
    typedef KVector<ValueType> Vector;

    KGaussSeidel_SingleThread(const Matrix& A,Vector& x,const Vector& b);
    ~KGaussSeidel_SingleThread() {}

    void Initialize();
    void FindResidual();
    void FindResidualNorm(double& residualNorm);
    void IncrementIndex();
    void ComputeCorrection();
    void UpdateSolutionApproximation();
    void UpdateVectorApproximation();
    void CoalesceData() {}
    void Finalize() {}

    unsigned int Dimension() const { return fB.Dimension(); }

    void SetResidualVector(const Vector&);
    void GetResidualVector(Vector&) const;

  private:
    const Matrix& fA;
    Vector& fX;
    const Vector& fB;

    KSimpleVector<ValueType> fB_iterative;
    KSimpleVector<ValueType> fResidual;

    double fBInfinityNorm;

    unsigned int fIndex;

    ValueType fCorrection;
  };

  template <typename ValueType>
  KGaussSeidel_SingleThread<ValueType>::KGaussSeidel_SingleThread(const Matrix& A,Vector& x,const Vector& b) : fA(A), fX(x), fB(b)
  {
    fIndex = Dimension()-1;
  }

  template <typename ValueType>
  void KGaussSeidel_SingleThread<ValueType>::Initialize()
  {
    if (fResidual.Dimension()==0)
    {
      fB_iterative.resize(fB.Dimension(),0.);
      fResidual.resize(fB.Dimension(),0.);

      if (fX.InfinityNorm()>1.e-16)
	fA.Multiply(fX,fB_iterative);
    }

    fBInfinityNorm = fB.InfinityNorm();
  }

  template <typename ValueType>
  void KGaussSeidel_SingleThread<ValueType>::FindResidual()
  {
    for (unsigned int i=0;i<fB.Dimension();i++)
      fResidual[i] = fB(i) - fB_iterative(i);
  }

  template <typename ValueType>
  void KGaussSeidel_SingleThread<ValueType>::FindResidualNorm(double& residualNorm)
  {
    residualNorm = fResidual.InfinityNorm()/fBInfinityNorm;
  }

  template <typename ValueType>
  void KGaussSeidel_SingleThread<ValueType>::IncrementIndex()
  {
    fIndex = (fIndex+1)%Dimension();
  }

  template <typename ValueType>
  void KGaussSeidel_SingleThread<ValueType>::ComputeCorrection()
  {
    fCorrection = (fB(fIndex) - fB_iterative(fIndex))/fA(fIndex,fIndex);
  }

  template <typename ValueType>
  void KGaussSeidel_SingleThread<ValueType>::UpdateSolutionApproximation()
  {
    fX[fIndex] += fCorrection;
  }

  template <typename ValueType>
  void KGaussSeidel_SingleThread<ValueType>::UpdateVectorApproximation()
  {
    for (unsigned int i = 0;i<fB_iterative.Dimension();i++)
      fB_iterative[i] += fA(i,fIndex)*fCorrection;
  }

  template <typename ValueType>
  void KGaussSeidel_SingleThread<ValueType>::SetResidualVector(const Vector& v)
  {
    fResidual.resize(v.Dimension());
    fB_iterative.resize(v.Dimension());

    for (unsigned int i = 0;i<v.Dimension();i++)
    {
      fResidual[i] = v(i);
      fB_iterative[i] = fB(i) - fResidual(i);
    }
  }

  template <typename ValueType>
  void KGaussSeidel_SingleThread<ValueType>::GetResidualVector(Vector& v) const
  {
    for (unsigned int i = 0;i<fResidual.Dimension();i++)
      v[i] = fResidual(i);
  }
}

#endif /* KGAUSSSEIDEL_SINGLETHREAD_DEF */
