#ifndef Kassiopeia_KSGenValueAngleCosine_h_
#define Kassiopeia_KSGenValueAngleCosine_h_

#include "KField.h"
#include "KSGenValue.h"

namespace Kassiopeia
{
class KSGenValueAngleCosine : public KSComponentTemplate<KSGenValueAngleCosine, KSGenValue>
{
  public:
    KSGenValueAngleCosine();
    KSGenValueAngleCosine(const KSGenValueAngleCosine& aCopy);
    KSGenValueAngleCosine* Clone() const override;
    ~KSGenValueAngleCosine() override;

  public:
    void DiceValue(std::vector<double>& aDicedValues) override;

  public:
    enum EDistributionMode {
        Classic,
        MolecularFlow
    };
    enum EDirection {
        Forward,
        Backward
    };

    K_SET_GET(double, AngleMin)
    K_SET_GET(double, AngleMax)
    K_SET_GET(EDistributionMode, Mode)
    K_SET_GET(EDirection, Direction)
};

}  // namespace Kassiopeia

#endif
