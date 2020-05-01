#include "KGLinearCongruentialGenerator.hh"

namespace KGeoBag
{
KGLinearCongruentialGenerator::KGLinearCongruentialGenerator() :
    fModulus(714025),
    fMultiplier(1366),
    fIncrement(150889),
    fValue(0)
{}

double KGLinearCongruentialGenerator::Random() const
{
    fValue = (fMultiplier * fValue + fIncrement) % fModulus;
    return fValue / double(fModulus - 1);
}

}  // namespace KGeoBag
