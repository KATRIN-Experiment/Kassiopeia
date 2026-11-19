#ifndef Kassiopeia_KSTrajInterpolatorFastBuilder_h_
#define Kassiopeia_KSTrajInterpolatorFastBuilder_h_

#include "KComplexElement.hh"
#include "KSTrajInterpolatorFast.h"

using namespace Kassiopeia;
namespace katrin
{

//! this interpolator uses a linear interpolation internally.
//! it is fast but can be innaccurate, especially when particles are changing direction in magnetic mirrors.
//! @example <kstraj_interpolator_fast name="interpolator_fast"/>
typedef KComplexElement<KSTrajInterpolatorFast> KSTrajInterpolatorFastBuilder;

template<> inline bool KSTrajInterpolatorFastBuilder::AddAttribute(KContainer* aContainer)
{
    if (aContainer->GetName() == "name") {
        aContainer->CopyTo(fObject, &KNamed::SetName);
        return true;
    }
    return false;
}

}  // namespace katrin

#endif
