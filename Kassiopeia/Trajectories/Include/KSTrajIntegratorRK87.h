#ifndef Kassiopeia_KSTrajIntegratorRK87_h_
#define Kassiopeia_KSTrajIntegratorRK87_h_

#include "KSComponentTemplate.h"
#include "KSMathRK87.h"
#include "KSTrajAdiabaticSpinTypes.h"
#include "KSTrajAdiabaticTypes.h"
#include "KSTrajElectricTypes.h"
#include "KSTrajExactSpinTypes.h"
#include "KSTrajExactTypes.h"
#include "KSTrajMagneticTypes.h"

namespace Kassiopeia
{

class KSTrajIntegratorRK87 :
    public KSComponentTemplate<KSTrajIntegratorRK87>,
    public KSMathRK87<KSTrajExactSystem>,
    public KSMathRK87<KSTrajExactSpinSystem>,
    public KSMathRK87<KSTrajAdiabaticSpinSystem>,
    public KSMathRK87<KSTrajAdiabaticSystem>,
    public KSMathRK87<KSTrajElectricSystem>,
    public KSMathRK87<KSTrajMagneticSystem>
{
  public:
    KSTrajIntegratorRK87();
    KSTrajIntegratorRK87(const KSTrajIntegratorRK87& aCopy);
    KSTrajIntegratorRK87* Clone() const override;
    ~KSTrajIntegratorRK87() override;
};

}  // namespace Kassiopeia

#endif
