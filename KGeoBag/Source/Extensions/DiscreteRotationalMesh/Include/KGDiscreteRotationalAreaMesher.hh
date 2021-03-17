/*
 * KGDiscreteRotationalAreaMesher.hh
 *
 *  Created on: 15.10.2015
 *      Author: hilk
 */

#ifndef KGEOBAG_KGDISCRETEROTATIONALAREAMESHER_HH_
#define KGEOBAG_KGDISCRETEROTATIONALAREAMESHER_HH_


#include "KGConicalWireArrayDiscreteRotationalMesher.hh"


namespace KGeoBag
{

class KGDiscreteRotationalAreaMesher : public KGConicalWireArrayDiscreteRotationalMesher
{
  public:
    KGDiscreteRotationalAreaMesher() = default;
    ~KGDiscreteRotationalAreaMesher() override = default;

    using KGConicalWireArrayDiscreteRotationalMesher::VisitWrappedSurface;
};

}  // namespace KGeoBag

#endif /* KGEOBAG_KGDISCRETEROTATIONALAREAMESHER_HH_ */
