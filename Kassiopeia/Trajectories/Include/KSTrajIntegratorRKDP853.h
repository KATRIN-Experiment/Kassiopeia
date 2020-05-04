#ifndef Kassiopeia_KSTrajIntegratorRKDP853_h_
#define Kassiopeia_KSTrajIntegratorRKDP853_h_

#include "KSComponentTemplate.h"
#include "KSMathRKDP853.h"
#include "KSTrajAdiabaticSpinTypes.h"
#include "KSTrajAdiabaticTypes.h"
#include "KSTrajElectricTypes.h"
#include "KSTrajExactSpinTypes.h"
#include "KSTrajExactTypes.h"
#include "KSTrajMagneticTypes.h"

namespace Kassiopeia
{

class KSTrajIntegratorRKDP853 :
    public KSComponentTemplate<KSTrajIntegratorRKDP853>,
    public KSMathRKDP853<KSTrajExactSystem>,
    public KSMathRKDP853<KSTrajExactSpinSystem>,
    public KSMathRKDP853<KSTrajAdiabaticSpinSystem>,
    public KSMathRKDP853<KSTrajAdiabaticSystem>,
    public KSMathRKDP853<KSTrajElectricSystem>,
    public KSMathRKDP853<KSTrajMagneticSystem>
{
  public:
    KSTrajIntegratorRKDP853();
    KSTrajIntegratorRKDP853(const KSTrajIntegratorRKDP853& aCopy);
    KSTrajIntegratorRKDP853* Clone() const override;
    ~KSTrajIntegratorRKDP853() override;
};

}  // namespace Kassiopeia

#endif
