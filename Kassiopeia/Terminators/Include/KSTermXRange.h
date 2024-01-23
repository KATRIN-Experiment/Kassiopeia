
#ifndef Kassiopeia_KSTermXRange_h_
#define Kassiopeia_KSTermXRange_h_

#include "KSTerminator.h"

namespace Kassiopeia
{

class KSParticle;

class KSTermXRange : public KSComponentTemplate<KSTermXRange, KSTerminator>
{
  public:
    KSTermXRange();
    KSTermXRange(const KSTermXRange& aCopy);
    KSTermXRange* Clone() const override;
    ~KSTermXRange() override;

  public:
    void CalculateTermination(const KSParticle& anInitialParticle, bool& aFlag) override;
    void ExecuteTermination(const KSParticle& anInitialParticle, KSParticle& aFinalParticle,
                            KSParticleQueue& aParticleQueue) const override;

  public:
    void SetMaxX(const double& aValue);
    void SetMinX(const double& aValue);

  private:
    double fMinX;
    double fMaxX;
};

inline void KSTermXRange::SetMinX(const double& aValue)
{
    fMinX = aValue;
}
inline void KSTermXRange::SetMaxX(const double& aValue)
{
    fMaxX = aValue;
}

}  // namespace Kassiopeia

#endif
