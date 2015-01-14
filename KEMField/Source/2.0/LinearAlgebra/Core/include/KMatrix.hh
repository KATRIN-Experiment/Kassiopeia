#ifndef KMATRIX_DEF
#define KMATRIX_DEF

#include "KVector.hh"
#include "KSimpleVector.hh"

namespace KEMField
{
  template <typename ValueType>
  class KMatrix
  {
  public:
    KMatrix() : fRow(*this) {}
    virtual ~KMatrix() {}

    virtual unsigned int Dimension(unsigned int) const = 0;
    virtual const ValueType& operator()(unsigned int,unsigned int) const = 0;

    class KMatrixRow : public KVector<ValueType>
    {
    public:
      friend class KMatrix;
      KMatrixRow(KMatrix& m,unsigned int row=0) :
	KVector<ValueType>(), fParent(m), i(row) {}
      virtual ~KMatrixRow() {}

      const ValueType& operator()(unsigned int j) const { return fParent(i,j); }

      unsigned int Dimension() const { return fParent.Dimension(1); }

    private:
      // We disable this method by making it private.
      virtual ValueType& operator[](unsigned int)
      { static ValueType dummy; return dummy; }

      KMatrix& fParent;
      mutable unsigned int i;
    };

  public:
    class KMatrixColumn : public KVector<ValueType>
    {
    public:
      friend class KMatrix;
      KMatrixColumn(KMatrix& m,unsigned int column=0) :
	KVector<ValueType>(), fParent(m), j(column) {}
      virtual ~KMatrixColumn() {}

      const ValueType& operator()(unsigned int i) const { return fParent(i,j); }

      unsigned int Dimension() const { return fParent.Dimension(0); }

    private:
      // We disable this method by making it private.
      virtual ValueType& operator[](unsigned int)
      { static ValueType dummy; return dummy; }

      KMatrix& fParent;
      mutable unsigned int j;
    };

  private:
    const KMatrixRow fRow;

  public:
    const KMatrixRow& operator()(unsigned int i) const
    { fRow.i = i; return fRow; }

    virtual void Multiply(const KVector<ValueType>& x,
			  KVector<ValueType>& y) const;

    virtual void MultiplyTranspose(const KVector<ValueType>& x,
				   KVector<ValueType>& y) const;
  };

  template <typename ValueType>
  void KMatrix<ValueType>::Multiply(const KVector<ValueType>& x,
				    KVector<ValueType>& y) const
  {
    // Computes vector y in the equation A*x = y
    for (unsigned int i=0;i<Dimension(0);i++)
    {
      y[i] = 0.;
      for (unsigned int j=0;j<Dimension(1);j++)
	y[i] += this->operator()(i,j)*x(j);
    }
  }

  template <typename ValueType>
  void KMatrix<ValueType>::MultiplyTranspose(const KVector<ValueType>& x,
					     KVector<ValueType>& y) const
  {
    // Computes vector y in the equation A^T*x = y
    for (unsigned int i=0;i<Dimension(1);i++)
    {
      y[i] = 0.;
      for (unsigned int j=0;j<Dimension(0);j++)
	y[i] += this->operator()(j,i)*x(j);
    }
  }
}

#endif /* KMATRIX_DEF */
