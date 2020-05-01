#ifndef Kassiopeia_KSTrajIntegratorRK65_h_
#define Kassiopeia_KSTrajIntegratorRK65_h_

#include "KSComponentTemplate.h"
#include "KSMathRK65.h"
#include "KSTrajAdiabaticSpinTypes.h"
#include "KSTrajAdiabaticTypes.h"
#include "KSTrajElectricTypes.h"
#include "KSTrajExactSpinTypes.h"
#include "KSTrajExactTypes.h"
#include "KSTrajMagneticTypes.h"

namespace Kassiopeia
{

class KSTrajIntegratorRK65 :
    public KSComponentTemplate<KSTrajIntegratorRK65>,
    public KSMathRK65<KSTrajExactSystem>,
    public KSMathRK65<KSTrajExactSpinSystem>,
    public KSMathRK65<KSTrajAdiabaticSpinSystem>,
    public KSMathRK65<KSTrajAdiabaticSystem>,
    public KSMathRK65<KSTrajElectricSystem>,
    public KSMathRK65<KSTrajMagneticSystem>
{
  public:
    KSTrajIntegratorRK65();
    KSTrajIntegratorRK65(const KSTrajIntegratorRK65& aCopy);
    KSTrajIntegratorRK65* Clone() const override;
    ~KSTrajIntegratorRK65() override;
};

}  // namespace Kassiopeia

#endif
