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
    KSIntCalculatorTritiumElasticBase() = default;
    ~KSIntCalculatorTritiumElasticBase() override = default;
};

/////////////////////////////////
/////       Elastic         /////
/////////////////////////////////

class KSIntCalculatorTritiumElastic :
    public KSComponentTemplate<KSIntCalculatorTritiumElastic, KSIntCalculatorTritiumElasticBase>
{
  public:
    KSIntCalculatorTritiumElastic() = default;
    KSIntCalculatorTritiumElastic* Clone() const override
    {
        return new KSIntCalculatorTritiumElastic(*this);
    }
    ~KSIntCalculatorTritiumElastic() override = default;

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
    KSIntCalculatorTritiumVib() = default;
    KSIntCalculatorTritiumVib* Clone() const override
    {
        return new KSIntCalculatorTritiumVib(*this);
    }
    ~KSIntCalculatorTritiumVib() override = default;
};

/////////////////////////////////
/////       Rot02           /////
/////////////////////////////////

class KSIntCalculatorTritiumRot02 :
    public KSComponentTemplate<KSIntCalculatorTritiumRot02, KSIntCalculatorHydrogenRot02>
{
  public:
    KSIntCalculatorTritiumRot02() = default;
    KSIntCalculatorTritiumRot02* Clone() const override
    {
        return new KSIntCalculatorTritiumRot02(*this);
    }
    ~KSIntCalculatorTritiumRot02() override = default;
};

/////////////////////////////////
/////       Rot13           /////
/////////////////////////////////

class KSIntCalculatorTritiumRot13 :
    public KSComponentTemplate<KSIntCalculatorTritiumRot13, KSIntCalculatorHydrogenRot13>
{
  public:
    KSIntCalculatorTritiumRot13() = default;
    KSIntCalculatorTritiumRot13* Clone() const override
    {
        return new KSIntCalculatorTritiumRot13(*this);
    }
    ~KSIntCalculatorTritiumRot13() override = default;
};

/////////////////////////////////
/////       Rot20           /////
/////////////////////////////////

class KSIntCalculatorTritiumRot20 :
    public KSComponentTemplate<KSIntCalculatorTritiumRot20, KSIntCalculatorHydrogenRot20>
{
  public:
    KSIntCalculatorTritiumRot20() = default;
    KSIntCalculatorTritiumRot20* Clone() const override
    {
        return new KSIntCalculatorTritiumRot20(*this);
    }
    ~KSIntCalculatorTritiumRot20() override = default;
};

} /* namespace Kassiopeia */

#endif /* KASSIOPEIA_INTERACTIONS_INCLUDE_KSINTCALCULATORTRITIUM_H_ */
