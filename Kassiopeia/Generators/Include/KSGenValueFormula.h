#ifndef Kassiopeia_KSGenValueFormula_h_
#define Kassiopeia_KSGenValueFormula_h_

#include "KField.h"
#include "KSGenValue.h"
#include "TF1.h"

namespace Kassiopeia
{
class KSGenValueFormula : public KSComponentTemplate<KSGenValueFormula, KSGenValue>
{
  public:
    KSGenValueFormula();
    KSGenValueFormula(const KSGenValueFormula& aCopy);
    KSGenValueFormula* Clone() const override;
    ~KSGenValueFormula() override;

  public:
    void DiceValue(std::vector<double>& aDicedValues) override;

  public:
    ;
    K_SET_GET(double, ValueMin);
    ;
    K_SET_GET(double, ValueMax);
    ;
    K_SET_GET(std::string, ValueFormula);

  public:
    void InitializeComponent() override;
    void DeinitializeComponent() override;

  protected:
    TF1* fValueFunction;
};

}  // namespace Kassiopeia

#endif
