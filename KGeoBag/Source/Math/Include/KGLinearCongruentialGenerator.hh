#ifndef KGKGLINEARCONGRUENTIALGENERATOR_HH_
#define KGKGLINEARCONGRUENTIALGENERATOR_HH_

namespace KGeoBag
{
class KGLinearCongruentialGenerator
{
  public:
    KGLinearCongruentialGenerator();
    virtual ~KGLinearCongruentialGenerator() = default;

    double Random() const;

  private:
    unsigned int fModulus;
    unsigned int fMultiplier;
    unsigned int fIncrement;
    mutable unsigned int fValue;
};
}  // namespace KGeoBag

#endif
