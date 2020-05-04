#ifndef Kassiopeia_KSTrajIntegratorRK8_h_
#define Kassiopeia_KSTrajIntegratorRK8_h_

#include "KSComponentTemplate.h"
#include "KSMathRK8.h"
#include "KSTrajAdiabaticSpinTypes.h"
#include "KSTrajAdiabaticTypes.h"
#include "KSTrajElectricTypes.h"
#include "KSTrajExactSpinTypes.h"
#include "KSTrajExactTrappedTypes.h"
#include "KSTrajExactTypes.h"
#include "KSTrajMagneticTypes.h"

namespace Kassiopeia
{

class KSTrajIntegratorRK8 :
    public KSComponentTemplate<KSTrajIntegratorRK8>,
    public KSMathRK8<KSTrajExactSystem>,
    public KSMathRK8<KSTrajExactSpinSystem>,
    public KSMathRK8<KSTrajAdiabaticSpinSystem>,
    public KSMathRK8<KSTrajAdiabaticSystem>,
    public KSMathRK8<KSTrajExactTrappedSystem>,
    public KSMathRK8<KSTrajElectricSystem>,
    public KSMathRK8<KSTrajMagneticSystem>
{
  public:
    KSTrajIntegratorRK8();
    KSTrajIntegratorRK8(const KSTrajIntegratorRK8& aCopy);
    KSTrajIntegratorRK8* Clone() const override;
    ~KSTrajIntegratorRK8() override;
};

}  // namespace Kassiopeia

#endif
