#ifndef Kassiopeia_KSGenValueList_h_
#define Kassiopeia_KSGenValueList_h_

#include "KField.h"
#include "KSGenValue.h"

namespace Kassiopeia
{

class KSGenValueList : public KSComponentTemplate<KSGenValueList, KSGenValue>
{
  public:
    KSGenValueList();
    KSGenValueList(const KSGenValueList& aCopy);
    KSGenValueList* Clone() const override;
    ~KSGenValueList() override;

  public:
    void DiceValue(std::vector<double>& aDicedValues) override;
    void AddValue(double aValue);
    void SetRandomize(bool aFlag);

  public:
    std::vector<double> fValues;
    bool fRandomize;
};

}  // namespace Kassiopeia

#endif
