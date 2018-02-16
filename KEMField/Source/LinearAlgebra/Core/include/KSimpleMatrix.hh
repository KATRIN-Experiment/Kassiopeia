#ifndef KSIMPLEMATRIX_DEF
#define KSIMPLEMATRIX_DEF

#include "KMatrix.hh"

#include <vector>

namespace KEMField
{
  template <typename ValueType>
  class KSimpleMatrix : public KMatrix<ValueType>
  {
  public:
    KSimpleMatrix(unsigned int nrows,unsigned int ncols);
    virtual ~KSimpleMatrix() {}

    unsigned int Dimension(unsigned int) const;
    const ValueType& operator()(unsigned int,unsigned int) const;
    ValueType& operator()(unsigned int,unsigned int);

  protected:
    KSimpleMatrix() {}

    std::vector<std::vector<ValueType> > fElements;
  };

  template <typename ValueType>
  KSimpleMatrix<ValueType>::KSimpleMatrix(unsigned int nrows,unsigned int ncols)
  {
    fElements.resize(nrows);
    for (unsigned int i = 0;i<nrows;i++)
      fElements.at(i).resize(ncols);
  }

  template <typename ValueType>
  unsigned int KSimpleMatrix<ValueType>::Dimension(unsigned int i) const
  {
    if (i == 0)
      return fElements.size();
    else
      return fElements.at(0).size();
  }

  template <typename ValueType>
  const ValueType& KSimpleMatrix<ValueType>::operator()(unsigned int i, unsigned int j) const
  {
    return fElements.at(i).at(j);
  }

  template <typename ValueType>
  ValueType& KSimpleMatrix<ValueType>::operator()(unsigned int i, unsigned int j)
  {
    return fElements.at(i).at(j);
  }

}

#endif /* KSIMPLEMATRIX_DEF */
