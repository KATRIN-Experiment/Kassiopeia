#ifndef Kassiopeia_KSTrajIntegratorSym4_h_
#define Kassiopeia_KSTrajIntegratorSym4_h_

#include "KSComponentTemplate.h"
#include "KSMathSym4.h"
#include "KSTrajExactTrappedTypes.h"

namespace Kassiopeia
{

class KSTrajIntegratorSym4 :
    public KSComponentTemplate<KSTrajIntegratorSym4>,
    public KSMathSym4<KSTrajExactTrappedSystem>
{
  public:
    KSTrajIntegratorSym4();
    KSTrajIntegratorSym4(const KSTrajIntegratorSym4& aCopy);
    KSTrajIntegratorSym4* Clone() const override;
    ~KSTrajIntegratorSym4() override;
};

}  // namespace Kassiopeia

#endif
