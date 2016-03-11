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
    public KSComponentTemplate< KSIntCalculatorTritiumElasticBase, KSIntCalculatorHydrogenElasticBase >
{
public:
    KSIntCalculatorTritiumElasticBase() { }
    virtual ~KSIntCalculatorTritiumElasticBase() { }
};

/////////////////////////////////
/////       Elastic         /////
/////////////////////////////////

class KSIntCalculatorTritiumElastic :
    public KSComponentTemplate< KSIntCalculatorTritiumElastic, KSIntCalculatorTritiumElasticBase >
{
public:
    KSIntCalculatorTritiumElastic() { }
    KSIntCalculatorTritiumElastic* Clone() const { return new KSIntCalculatorTritiumElastic( *this ); }
    virtual ~KSIntCalculatorTritiumElastic() { }

public:
    virtual void CalculateCrossSection( const double anEnergy, double& aCrossSection );
    virtual void CalculateEloss( const double anEnergy, const double aTheta, double& anEloss );

};

/////////////////////////////////
/////       Vibration       /////
/////////////////////////////////

class KSIntCalculatorTritiumVib :
    public KSComponentTemplate< KSIntCalculatorTritiumVib, KSIntCalculatorHydrogenVib >
{
public:
    KSIntCalculatorTritiumVib() { }
    KSIntCalculatorTritiumVib* Clone() const { return new KSIntCalculatorTritiumVib( *this ); }
    virtual ~KSIntCalculatorTritiumVib() { }
};

/////////////////////////////////
/////       Rot02           /////
/////////////////////////////////

class KSIntCalculatorTritiumRot02 :
    public KSComponentTemplate< KSIntCalculatorTritiumRot02, KSIntCalculatorHydrogenRot02 >
{
public:
    KSIntCalculatorTritiumRot02() { }
    KSIntCalculatorTritiumRot02* Clone() const { return new KSIntCalculatorTritiumRot02( *this ); }
    virtual ~KSIntCalculatorTritiumRot02() { }
};

/////////////////////////////////
/////       Rot13           /////
/////////////////////////////////

class KSIntCalculatorTritiumRot13 :
    public KSComponentTemplate< KSIntCalculatorTritiumRot13, KSIntCalculatorHydrogenRot13 >
{
public:
    KSIntCalculatorTritiumRot13() { }
    KSIntCalculatorTritiumRot13* Clone() const { return new KSIntCalculatorTritiumRot13( *this ); }
    virtual ~KSIntCalculatorTritiumRot13() { }
};

/////////////////////////////////
/////       Rot20           /////
/////////////////////////////////

class KSIntCalculatorTritiumRot20 :
    public KSComponentTemplate< KSIntCalculatorTritiumRot20, KSIntCalculatorHydrogenRot20 >
{
public:
    KSIntCalculatorTritiumRot20() { }
    KSIntCalculatorTritiumRot20* Clone() const { return new KSIntCalculatorTritiumRot20( *this ); }
    virtual ~KSIntCalculatorTritiumRot20() { }
};

} /* namespace Kassiopeia */

#endif /* KASSIOPEIA_INTERACTIONS_INCLUDE_KSINTCALCULATORTRITIUM_H_ */
