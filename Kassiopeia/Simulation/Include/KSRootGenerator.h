#ifndef Kassiopeia_KSRootGenerator_h_
#define Kassiopeia_KSRootGenerator_h_

#include "KSEvent.h"
#include "KSGenerator.h"

namespace Kassiopeia
{

class KSRootGenerator : public KSComponentTemplate<KSRootGenerator, KSGenerator>
{
  public:
    KSRootGenerator();
    KSRootGenerator(const KSRootGenerator& aCopy);
    KSRootGenerator* Clone() const override;
    ~KSRootGenerator() override;

    //*********
    //generator
    //*********

  protected:
    void ExecuteGeneration(KSParticleQueue& aPrimaries) override;

    //***********
    //composition
    //***********

  public:
    void SetGenerator(KSGenerator* aGenerator);
    void ClearGenerator(KSGenerator* aGenerator);

  private:
    KSGenerator* fGenerator;

    //******
    //action
    //******

  public:
    void SetEvent(KSEvent* anEvent);
    KSEvent* GetEvent() const;

    void ExecuteGeneration();

  private:
    KSEvent* fEvent;
};

}  // namespace Kassiopeia

#endif
