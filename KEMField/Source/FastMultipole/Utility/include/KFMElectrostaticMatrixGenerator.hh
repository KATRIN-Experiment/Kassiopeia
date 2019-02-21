/*
 * KFMElectrostaticMatrixGenerator.hh
 *
 *  Created on: 11 Aug 2015
 *      Author: wolfgang
 */

#ifndef KFMELECTROSTATICMATRIXGENERATOR_HH_
#define KFMELECTROSTATICMATRIXGENERATOR_HH_

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

#endif /* KFMELECTROSTATICMATRIXGENERATOR_HH_ */
