#ifndef _KSGenPositionSurfaceAdjustmentStepBuilder_H_
#define _KSGenPositionSurfaceAdjustmentStepBuilder_H_

/*
 * KSGenPositionSurfaceAdjustmentStep.h
 *
 *  Created on: 28.01.2015
 *      Author: Nikolaus Trost
 */

#include "KSGenPositionSurfaceAdjustmentStep.h"
#include "KComplexElement.hh"
#include "KSRootBuilder.h"

using namespace Kassiopeia;
namespace katrin
{
    typedef KComplexElement<KSGenPositionSurfaceAdjustmentStep> KSGenPositionSurfaceAdjustmentStepBuilder;

    template<>
    inline bool KSGenPositionSurfaceAdjustmentStepBuilder::AddAttribute(KContainer* aContainer)
    {
        if(aContainer->GetName() == "name")
        {
            aContainer->CopyTo(fObject, &KNamed::SetName);
            return true;
        }

        if(aContainer->GetName() == "length")
        {
            aContainer->CopyTo(fObject, &KSGenPositionSurfaceAdjustmentStep::SetLength );
            return true;
        }

        return false;
    }
}

#endif /* _KSGenPositionSurfaceAdjustmentStepBuilder_H_ */
