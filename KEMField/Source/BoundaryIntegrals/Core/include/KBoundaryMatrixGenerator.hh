/*
 * KBoundaryMatrixGenerator.hh
 *
 *  Created on: 11 Aug 2015
 *      Author: wolfgang
 */

#ifndef KEMFIELD_SOURCE_2_0_BOUNDARYINTEGRALS_CORE_INCLUDE_KBOUNDARYMATRIXGENERATOR_HH_
#define KEMFIELD_SOURCE_2_0_BOUNDARYINTEGRALS_CORE_INCLUDE_KBOUNDARYMATRIXGENERATOR_HH_

#include "KSmartPointer.hh"
#include "KSquareMatrix.hh"
#include "KSurfaceContainer.hh"

namespace KEMField {

template< typename ValueType >
class KBoundaryMatrixGenerator
{

public:
	KBoundaryMatrixGenerator() {}
	virtual ~KBoundaryMatrixGenerator() {}

	virtual KSmartPointer<KSquareMatrix<ValueType> > Build(const KSurfaceContainer& container) const
	= 0;

};

} /* namespace KEMField */

#endif /* KEMFIELD_SOURCE_2_0_BOUNDARYINTEGRALS_CORE_INCLUDE_KBOUNDARYMATRIXGENERATOR_HH_ */
