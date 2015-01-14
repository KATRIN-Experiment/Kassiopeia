#ifndef KSimpleSquareMatrix_HH__
#define KSimpleSquareMatrix_HH__

#include "KSquareMatrix.hh"

/*
*
*@file KSimpleSquareMatrix.hh
*@class KSimpleSquareMatrix
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Wed Jun  4 13:53:28 EDT 2014 J. Barrett (barrettj@mit.edu) First Version
*
*/

namespace KEMField
{


    template <typename ValueType>
    class KSimpleSquareMatrix : public KSquareMatrix<ValueType>
    {
        public:
        KSimpleSquareMatrix(unsigned int size);
        virtual ~KSimpleSquareMatrix() {}

        unsigned int Dimension() const;
        const ValueType& operator()(unsigned int,unsigned int) const;
        ValueType& operator()(unsigned int,unsigned int);

        protected:
        KSimpleSquareMatrix() {}
        std::vector<std::vector<ValueType> > fElements;
    };

    template <typename ValueType>
    KSimpleSquareMatrix<ValueType>::KSimpleSquareMatrix(unsigned int size)
    {
        fElements.resize(size);
        for (unsigned int i = 0;i<size;i++)
          fElements.at(i).resize(size);
    }

    template <typename ValueType>
    unsigned int KSimpleSquareMatrix<ValueType>::Dimension() const
    {
        return fElements.size();
    }

    template <typename ValueType>
    const ValueType& KSimpleSquareMatrix<ValueType>::operator()(unsigned int i, unsigned int j) const
    {
        return fElements.at(i).at(j);
    }

    template <typename ValueType>
    ValueType& KSimpleSquareMatrix<ValueType>::operator()(unsigned int i, unsigned int j)
    {
        return fElements.at(i).at(j);
    }

}

#endif /* KSimpleSquareMatrix_H__ */
