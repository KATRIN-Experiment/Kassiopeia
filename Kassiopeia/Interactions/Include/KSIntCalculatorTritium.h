/**
 * @file KSIntCalculatorTritium.h
 *
 * @date 01.12.2015
 * @author Marco Kleesiek <marco.kleesiek@kit.edu>
 */
#ifndef KASSIOPEIA_INTERACTIONS_INCLUDE_KSINTCALCULATORTRITIUM_H_
#define KASSIOPEIA_INTERACTIONS_INCLUDE_KSINTCALCULATORTRITIUM_H_

#include "KSIntCalculatorHydrogen.h"

namespace Kassiopeia
{
/////////////////////////////////////
/////       Elastic Base        /////
/////////////////////////////////////

class KSIntCalculatorTritiumElasticBase :
    public KSComponentTemplate<KSIntCalculatorTritiumElasticBase, KSIntCalculatorHydrogenElasticBase>
{
  public:
    KSIntCalculatorTritiumElasticBase() {}
    ~KSIntCalculatorTritiumElasticBase() override {}
};

/////////////////////////////////
/////       Elastic         /////
/////////////////////////////////

class KSIntCalculatorTritiumElastic :
    public KSComponentTemplate<KSIntCalculatorTritiumElastic, KSIntCalculatorTritiumElasticBase>
{
  public:
    KSIntCalculatorTritiumElastic() {}
    KSIntCalculatorTritiumElastic* Clone() const override
    {
        return new KSIntCalculatorTritiumElastic(*this);
    }
    ~KSIntCalculatorTritiumElastic() override {}

  public:
    void CalculateCrossSection(const double anEnergy, double& aCrossSection) override;
    void CalculateEloss(const double anEnergy, const double aTheta, double& anEloss) override;
};

/////////////////////////////////
/////       Vibration       /////
/////////////////////////////////

class KSIntCalculatorTritiumVib : public KSComponentTemplate<KSIntCalculatorTritiumVib, KSIntCalculatorHydrogenVib>
{
  public:
    KSIntCalculatorTritiumVib() {}
    KSIntCalculatorTritiumVib* Clone() const override
    {
        return new KSIntCalculatorTritiumVib(*this);
    }
    ~KSIntCalculatorTritiumVib() override {}
};

/////////////////////////////////
/////       Rot02           /////
/////////////////////////////////

class KSIntCalculatorTritiumRot02 :
    public KSComponentTemplate<KSIntCalculatorTritiumRot02, KSIntCalculatorHydrogenRot02>
{
  public:
    KSIntCalculatorTritiumRot02() {}
    KSIntCalculatorTritiumRot02* Clone() const override
    {
        return new KSIntCalculatorTritiumRot02(*this);
    }
    ~KSIntCalculatorTritiumRot02() override {}
};

/////////////////////////////////
/////       Rot13           /////
/////////////////////////////////

class KSIntCalculatorTritiumRot13 :
    public KSComponentTemplate<KSIntCalculatorTritiumRot13, KSIntCalculatorHydrogenRot13>
{
  public:
    KSIntCalculatorTritiumRot13() {}
    KSIntCalculatorTritiumRot13* Clone() const override
    {
        return new KSIntCalculatorTritiumRot13(*this);
    }
    ~KSIntCalculatorTritiumRot13() override {}
};

/////////////////////////////////
/////       Rot20           /////
/////////////////////////////////

class KSIntCalculatorTritiumRot20 :
    public KSComponentTemplate<KSIntCalculatorTritiumRot20, KSIntCalculatorHydrogenRot20>
{
  public:
    KSIntCalculatorTritiumRot20() {}
    KSIntCalculatorTritiumRot20* Clone() const override
    {
        return new KSIntCalculatorTritiumRot20(*this);
    }
    ~KSIntCalculatorTritiumRot20() override {}
};

} /* namespace Kassiopeia */

#endif /* KASSIOPEIA_INTERACTIONS_INCLUDE_KSINTCALCULATORTRITIUM_H_ */
