#ifndef Kassiopeia_KSWriter_h_
#define Kassiopeia_KSWriter_h_

#include "KSComponentTemplate.h"

namespace Kassiopeia
{

class KSWriter : public KSComponentTemplate<KSWriter>
{
  public:
    KSWriter();
    ~KSWriter() override;

  public:
    virtual void ExecuteRun() = 0;
    virtual void ExecuteEvent() = 0;
    virtual void ExecuteTrack() = 0;
    virtual void ExecuteStep() = 0;
};

}  // namespace Kassiopeia

#endif
