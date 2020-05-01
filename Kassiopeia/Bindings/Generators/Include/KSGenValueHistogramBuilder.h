#ifndef Kassiopeia_KSGenValueHistogramBuilder_h_
#define Kassiopeia_KSGenValueHistogramBuilder_h_

#include "KComplexElement.hh"
#include "KSGenValueHistogram.h"

using namespace Kassiopeia;
namespace katrin
{

typedef KComplexElement<KSGenValueHistogram> KSGenValueHistogramBuilder;

template<> inline bool KSGenValueHistogramBuilder::AddAttribute(KContainer* aContainer)
{
    if (aContainer->GetName() == "name") {
        aContainer->CopyTo(fObject, &KNamed::SetName);
        return true;
    }
    if (aContainer->GetName() == "base") {
        aContainer->CopyTo(fObject, &KSGenValueHistogram::SetBase);
        return true;
    }
    if (aContainer->GetName() == "path") {
        aContainer->CopyTo(fObject, &KSGenValueHistogram::SetPath);
        return true;
    }
    if (aContainer->GetName() == "histogram") {
        aContainer->CopyTo(fObject, &KSGenValueHistogram::SetHistogram);
        return true;
    }
    if (aContainer->GetName() == "formula") {
        aContainer->CopyTo(fObject, &KSGenValueHistogram::SetFormula);
        return true;
    }
    return false;
}

}  // namespace katrin

#endif
