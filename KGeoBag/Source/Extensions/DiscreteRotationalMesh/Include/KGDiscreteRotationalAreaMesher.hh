/*
 * KGDiscreteRotationalAreaMesher.hh
 *
 *  Created on: 15.10.2015
 *      Author: hilk
 */

#ifndef KGEOBAG_KGDISCRETEROTATIONALAREAMESHER_HH_
#define KGEOBAG_KGDISCRETEROTATIONALAREAMESHER_HH_


#include "KGConicalWireArraySurface.hh"


namespace KGeoBag {

	class KGDiscreteRotationalAreaMesher :
			public KGConicalWireArrayDiscreteRotationalMesher
	{
		public:
			KGDiscreteRotationalAreaMesher(){};
			virtual ~KGDiscreteRotationalAreaMesher(){};

			using KGConicalWireArrayDiscreteRotationalMesher::VisitWrappedSurface;
	};

}

#endif /* KGEOBAG_KGDISCRETEROTATIONALAREAMESHER_HH_ */
