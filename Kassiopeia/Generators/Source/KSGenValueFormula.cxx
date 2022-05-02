#include "KSGenValueFormula.h"

#include "KRandomRootInterface.h"
#include "KSGeneratorsMessage.h"

#include "RVersion.h"

namespace Kassiopeia
{

KSGenValueFormula::KSGenValueFormula() : fValueMin(0.), fValueMax(0.), fValueFormula("x"), fValueFunction(nullptr) {}
KSGenValueFormula::KSGenValueFormula(const KSGenValueFormula& aCopy) :
    KSComponent(aCopy),
    fValueMin(aCopy.fValueMin),
    fValueMax(aCopy.fValueMax),
    fValueFormula(aCopy.fValueFormula),
    fValueFunction(nullptr)
{}
KSGenValueFormula* KSGenValueFormula::Clone() const
{
    return new KSGenValueFormula(*this);
}
KSGenValueFormula::~KSGenValueFormula() = default;

void KSGenValueFormula::DiceValue(std::vector<double>& aDicedValues)
{
    double tValue;

#if ROOT_VERSION_CODE < ROOT_VERSION(6,24,0)
#pragma message "Using ROOT's standard RNG (TRandom3) instead of the common Kasper interface (KRandom)"
    tValue = fValueFunction->GetRandom(fValueMin, fValueMax);
#else
    auto rng = katrin::Kommon::KRandomRootInterface<katrin::KRandom::engine_type>();
    tValue = fValueFunction->GetRandom(fValueMin, fValueMax, &rng);
#endif
    aDicedValues.push_back(tValue);

    return;
}

void KSGenValueFormula::InitializeComponent()
{
#if ROOT_VERSION_CODE < ROOT_VERSION(6,24,0)
    genmsg(eWarning) << "KSGenValueFormula will produce random numbers independent of the user-specified seed!" << eom;
#endif
    fValueFunction = new TF1("function", fValueFormula.c_str(), fValueMin, fValueMax);
    return;
}
void KSGenValueFormula::DeinitializeComponent()
{
    delete fValueFunction;
    fValueFunction = nullptr;
    return;
}

}  // namespace Kassiopeia
