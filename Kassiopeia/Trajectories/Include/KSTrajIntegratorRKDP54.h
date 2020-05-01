#ifndef Kassiopeia_KSTrajIntegratorRKDP54_h_
#define Kassiopeia_KSTrajIntegratorRKDP54_h_

#include "KSComponentTemplate.h"
#include "KSMathRKDP54.h"
#include "KSTrajAdiabaticSpinTypes.h"
#include "KSTrajAdiabaticTypes.h"
#include "KSTrajElectricTypes.h"
#include "KSTrajExactSpinTypes.h"
#include "KSTrajExactTypes.h"
#include "KSTrajMagneticTypes.h"

namespace Kassiopeia
{

class KSTrajIntegratorRKDP54 :
    public KSComponentTemplate<KSTrajIntegratorRKDP54>,
    public KSMathRKDP54<KSTrajExactSystem>,
    public KSMathRKDP54<KSTrajExactSpinSystem>,
    public KSMathRKDP54<KSTrajAdiabaticSpinSystem>,
    public KSMathRKDP54<KSTrajAdiabaticSystem>,
    public KSMathRKDP54<KSTrajElectricSystem>,
    public KSMathRKDP54<KSTrajMagneticSystem>
{
  public:
    KSTrajIntegratorRKDP54();
    KSTrajIntegratorRKDP54(const KSTrajIntegratorRKDP54& aCopy);
    KSTrajIntegratorRKDP54* Clone() const override;
    ~KSTrajIntegratorRKDP54() override;
};

}  // namespace Kassiopeia

#endif
