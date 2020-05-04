#ifndef KSROOTQUANTITY_H_
#define KSROOTQUANTITY_H_

#include "KSList.h"
#include "KSWriter.h"

namespace Kassiopeia
{

class KSRootWriter : public KSComponentTemplate<KSRootWriter, KSWriter>
{
  public:
    KSRootWriter();
    KSRootWriter(const KSRootWriter& aCopy);
    KSRootWriter* Clone() const override;
    ~KSRootWriter() override;

    //******
    //writer
    //******

  public:
    void ExecuteRun() override;
    void ExecuteEvent() override;
    void ExecuteTrack() override;
    void ExecuteStep() override;

    //***********
    //composition
    //***********

  public:
    void AddWriter(KSWriter* aWriter);
    void RemoveWriter(KSWriter* aWriter);

  private:
    KSList<KSWriter> fWriters;
};
}  // namespace Kassiopeia

#endif
