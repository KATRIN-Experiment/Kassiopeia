/*
 * KFMElectrostaticMatrixGenerator.hh
 *
 *  Created on: 11 Aug 2015
 *      Author: wolfgang
 */

#ifndef KEMFIELD_SOURCE_2_0_FASTMULTIPOLE_INTERFACE_BOUNDARYINTEGRALS_KFMELECTROSTATICMATRIXGENERATOR_HH_
#define KEMFIELD_SOURCE_2_0_FASTMULTIPOLE_INTERFACE_BOUNDARYINTEGRALS_KFMELECTROSTATICMATRIXGENERATOR_HH_

#include "KBoundaryMatrixGenerator.hh"

#include "KFMElectrostaticTypes.hh"

namespace KEMField {

class KFMElectrostaticMatrixGenerator
		: public KBoundaryMatrixGenerator<KFMElectrostaticTypes::ValueType>
{
public:

	KSmartPointer< KSquareMatrix<KFMElectrostaticTypes::ValueType> >
	Build(const KSurfaceContainer* container) const;


private:
};

} /* namespace KEMField */

#endif /* KEMFIELD_SOURCE_2_0_FASTMULTIPOLE_INTERFACE_BOUNDARYINTEGRALS_KFMELECTROSTATICMATRIXGENERATOR_HH_ */
