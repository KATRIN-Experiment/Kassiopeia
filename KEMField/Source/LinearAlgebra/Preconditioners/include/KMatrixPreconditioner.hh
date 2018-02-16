/*
 * KMatrixPreconditioner.hh
 *
 *  Created on: 13 Aug 2015
 *      Author: wolfgang
 */

#ifndef KEMFIELD_SOURCE_2_0_LINEARALGEBRA_PRECONDITIONERS_INCLUDE_KMATRIXPRECONDITIONER_HH_
#define KEMFIELD_SOURCE_2_0_LINEARALGEBRA_PRECONDITIONERS_INCLUDE_KMATRIXPRECONDITIONER_HH_

#include "KPreconditioner.hh"
#include "KSmartPointer.hh"

namespace KEMField {

/**
 *  Matrix Preconditioner allows any KSquareMatrix to be used as KPreconditioner.
 *  The KPreconditioner specific functions IsStationary and Name are implemented
 *  with default values.
 */

template< typename ValueType>
class KMatrixPreconditioner : public KPreconditioner<ValueType> {
public:
	explicit KMatrixPreconditioner(KSmartPointer<const KSquareMatrix<ValueType> > matrix);
	virtual ~KMatrixPreconditioner(){}

	virtual std::string Name() {return "Created from unnamed matrix";}

	virtual bool IsStationary() { return false;}

	//from KSquareMatrix
	virtual unsigned int Dimension() const {
		return fMatrix->Dimension();
	}

	virtual const ValueType& operator()(unsigned int i, unsigned int j) const {
		return fMatrix->operator()(i,j);
	}

	//from KMatrix
	virtual void Multiply(const KVector<ValueType>& x,
			KVector<ValueType>& y) const {
		fMatrix->Multiply(x,y);
	}

	virtual void MultiplyTranspose(const KVector<ValueType>& x,
			KVector<ValueType>& y) const {
		fMatrix->MultiplyTranspose(x,y);
	}

private:
	KSmartPointer<const KSquareMatrix<ValueType> > fMatrix;
};

template<typename ValueType>
KMatrixPreconditioner<ValueType>::KMatrixPreconditioner(
		KSmartPointer<const KSquareMatrix<ValueType> > matrix) : fMatrix(matrix) {}

} /* namespace KEMField */

#endif /* KEMFIELD_SOURCE_2_0_LINEARALGEBRA_PRECONDITIONERS_INCLUDE_KMATRIXPRECONDITIONER_HH_ */
