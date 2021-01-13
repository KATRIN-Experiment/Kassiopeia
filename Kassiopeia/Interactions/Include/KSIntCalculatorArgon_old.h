/*
 * KSIntCalculatorArgon.h
 *
 *  Created on: 04.12.2013
 *      Author: oertlin
 */

#ifndef KINTCALCULATORARGON_H_
#define KINTCALCULATORARGON_H_

#include "KSIntCalculator.h"

#include <map>
#include <vector>

namespace Kassiopeia
{
/////////////////////////////////
/////		Data Reader 	/////
/////////////////////////////////
class KSIntCalculatorArgonDataReader
{
  private:
    std::map<double, double>* fData;
    std::vector<double>* fParameters;
    unsigned int fNumOfParameters;
    std::istream* fStream;

  public:
    KSIntCalculatorArgonDataReader(std::istream* aStream, unsigned int numOfParameters);
    ~KSIntCalculatorArgonDataReader();

    /**
             * \brief Read the data file. In case of an error it returns false. Otherwise true.
             * First, this method reads the parameters, e.g. for extrapolation. The number of parameters
             * is defined by numOfParameters. Then it reads data points (energy and cross-section) as many as available.
             */
    bool Read();

    /**
             * \brief Returns the data map. First parameter of this map is the energy, the second
             * the cross section.
             */
    std::map<double, double>* GetData();

    /**
             * \brief Returns a std::vector with read in parameters. The length of this std::vector is
             * defined by numOfParameters.
             */
    std::vector<double>* GetParameters();
};

/////////////////////////////////
/////		Mother			/////
/////////////////////////////////
class KSIntCalculatorArgon : public KSComponentTemplate<KSIntCalculatorArgon, KSIntCalculator>
{
  protected:
    std::string fDataFile;

  protected:
    /**
             * \brief Calculates the supporting points for interpolation/extrapolation/... The
             * result is stored in fSupportingPoints and fParameters. Currently it reads the
             * data file and stores the data and the parameters in they corresponding fields.
             *
             * \parameter numOfParameters The number of parameters which have to read in before data
             */
    virtual void ComputeSupportingPoints(unsigned int numOfParameters);

    /**
             * \brief Calculates the cross-section for the interpolation region. The parameter "point" is
             * the data point which is the first element in the map which goes after "anEnergy".
             * The current implementation executes a linear interpolation.
             *
             * \return Cross-section
             */
    virtual double GetInterpolation(const double& anEnergy, std::map<double, double>::iterator& point) const;

    /**
             * \brief Calculates the cross-section for extrapolation at high energies. The parameter "point" is
             * the data point which is the first element in the map which goes after "anEnergy".
             * The current implementation executes a power law extrapolation with parameters fParameters[0]
             * and fParameters[1].
             *
             * \return Cross-section
             */
    virtual double GetUpperExtrapolation(const double& anEnergy, std::map<double, double>::iterator& point) const;

    /**
             * \brief Calculates the cross-section for extrapolation at low energies. The parameter "point" is
             * the data point which is the first element in the map which goes after "anEnergy".
             * The current implementation returns zero.
             *
             * \return Cross-section
             */
    virtual double GetLowerExtrapolation(const double& anEnergy, std::map<double, double>::iterator& point) const;

    /**
             * \brief Calculates the cross-section for the given energy. It splits up
             * the energy region into three parts: 1. Energy lower than the lowest data point. Then
             * it calls the virtual method GetLowerExtrapolation. 2. Region of data points. So
             * we can do an interpolation. So it calls GetInterpolation. 3. Extrapolation for higher
             * energies. It calls GetUpperExtrapolation.
             *
             * \return Cross-section
             */
    double GetCrossSectionAt(const double& anEnergy) const;

  public:
    KSIntCalculatorArgon();
    virtual ~KSIntCalculatorArgon();
    std::map<double, double>* DEBUG_GetSupportingPoints();

  public:
    virtual void CalculateCrossSection(const KSParticle& aParticle, double& aCrossSection);
    virtual void CalculateCrossSection(const double anEnergie, double& aCrossSection);

  protected:
    std::map<double, double>* fSupportingPoints;
    std::vector<double>* fParameters;
};

/////////////////////////////////////
/////		Elastic Child		/////
/////////////////////////////////////
class KSIntCalculatorArgonElastic : public KSComponentTemplate<KSIntCalculatorArgonElastic, KSIntCalculatorArgon>
{
  public:
    KSIntCalculatorArgonElastic();
    KSIntCalculatorArgonElastic(const KSIntCalculatorArgonElastic& aCopy);
    KSIntCalculatorArgonElastic* Clone() const;
    virtual ~KSIntCalculatorArgonElastic();

  public:
    void ExecuteInteraction(const KSParticle& anInitialParticle, KSParticle& aFinalParticle,
                            KSParticleQueue& aSecondaries);

  protected:
    virtual void InitializeComponent();
};

/////////////////////////////////////
/////		Excited Child		/////
/////////////////////////////////////
class KSIntCalculatorArgonExcitation : public KSComponentTemplate<KSIntCalculatorArgonExcitation, KSIntCalculatorArgon>
{
  public:
    KSIntCalculatorArgonExcitation();
    KSIntCalculatorArgonExcitation(const KSIntCalculatorArgonExcitation& aCopy);
    KSIntCalculatorArgonExcitation* Clone() const;
    virtual ~KSIntCalculatorArgonExcitation();

  public:
    void ExecuteInteraction(const KSParticle& anInitialParticle, KSParticle& aFinalParticle,
                            KSParticleQueue& aSecondaries);
    void SetExcitationState(unsigned int aState);

  protected:
    virtual void InitializeComponent();
    virtual double GetUpperExtrapolation(const double& anEnergy, std::map<double, double>::iterator& point) const;

  protected:
    unsigned int fExcitationState;
};

/////////////////////////////////////
/////		Ionized Child		/////
/////////////////////////////////////
class KSIntCalculatorArgonSingleIonisation :
    public KSComponentTemplate<KSIntCalculatorArgonSingleIonisation, KSIntCalculatorArgon>
{
  protected:
    double fIonizationEnergy;

  public:
    KSIntCalculatorArgonSingleIonisation();
    KSIntCalculatorArgonSingleIonisation(const KSIntCalculatorArgonSingleIonisation& aCopy);
    KSIntCalculatorArgonSingleIonisation* Clone() const;
    virtual ~KSIntCalculatorArgonSingleIonisation();

  public:
    void ExecuteInteraction(const KSParticle& anInitialParticle, KSParticle& aFinalParticle,
                            KSParticleQueue& aSecondaries);

  protected:
    virtual void InitializeComponent();
};

/////////////////////////////////////////////
/////		Double Ionized Child		/////
/////////////////////////////////////////////
class KSIntCalculatorArgonDoubleIonisation :
    public KSComponentTemplate<KSIntCalculatorArgonDoubleIonisation, KSIntCalculatorArgon>
{
  protected:
    std::vector<double>* fIonizationEnergy;

  public:
    KSIntCalculatorArgonDoubleIonisation();
    KSIntCalculatorArgonDoubleIonisation(const KSIntCalculatorArgonDoubleIonisation& aCopy);
    KSIntCalculatorArgonDoubleIonisation* Clone() const;
    virtual ~KSIntCalculatorArgonDoubleIonisation();

  public:
    void ExecuteInteraction(const KSParticle& anInitialParticle, KSParticle& aFinalParticle,
                            KSParticleQueue& aSecondaries);

  protected:
    virtual void InitializeComponent();
};
} /* namespace Kassiopeia */

/////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////
/////                                                   /////
/////  BBBB   U   U  IIIII  L      DDDD   EEEEE  RRRR   /////
/////  B   B  U   U    I    L      D   D  E      R   R  /////
/////  BBBB   U   U    I    L      D   D  EE     RRRR   /////
/////  B   B  U   U    I    L      D   D  E      R   R  /////
/////  BBBB    UUU   IIIII  LLLLL  DDDD   EEEEE  R   R  /////
/////                                                   /////
/////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////

#include "KField.h"

class KSIntCalculatorArgonData
{
    ;
    K_SET_GET(string, Name);
    ;
    K_SET_GET(bool, SingleIonisation);
    ;
    K_SET_GET(bool, DoubleIonisation);
    ;
    K_SET_GET(bool, Excitation);
    ;
    K_SET_GET(bool, Elastic);
};

#include "KComplexElement.hh"
#include "KSInteractionsMessage.h"
#include "KToolbox.h"

namespace katrin
{
typedef KComplexElement<KSIntCalculatorArgonData> KSIntCalculatorArgonBuilder;

template<> inline bool KSIntCalculatorArgonBuilder::Begin()
{
    fObject = new KSIntCalculatorArgonData;

    fObject->SetElastic(true);
    fObject->SetExcitation(true);
    fObject->SetSingleIonisation(true);
    fObject->SetDoubleIonisation(true);

    return true;
}

template<> inline bool KSIntCalculatorArgonBuilder::AddAttribute(KContainer* aContainer)
{
    if (aContainer->GetName() == "name") {
        aContainer->CopyTo(fObject, &KSIntCalculatorArgonData::SetName);
        return true;
    }
    else if (aContainer->GetName() == "elastic") {
        aContainer->CopyTo(fObject, &KSIntCalculatorArgonData::SetElastic);
        return true;
    }
    else if (aContainer->GetName() == "excitation") {
        aContainer->CopyTo(fObject, &KSIntCalculatorArgonData::SetExcitation);
        return true;
    }
    else if (aContainer->GetName() == "single_ionisation") {
        aContainer->CopyTo(fObject, &KSIntCalculatorArgonData::SetSingleIonisation);
        return true;
    }
    else if (aContainer->GetName() == "double_ionisation") {
        aContainer->CopyTo(fObject, &KSIntCalculatorArgonData::SetDoubleIonisation);
        return true;
    }

    return false;
}

template<> bool KSIntCalculatorArgonBuilder::End()
{
    KToolbox& tToolBox = KToolbox::GetInstance();
    KSIntCalculator* aIntCalculator;

    if (fObject->GetElastic()) {
        aIntCalculator = new KSIntCalculatorArgonElastic();
        aIntCalculator->SetName(fObject->GetName() + "_elastic");
        aIntCalculator->SetTag(fObject->GetName());
        tToolBox.AddObject(aIntCalculator);
    }

    if (fObject->GetSingleIonisation()) {
        aIntCalculator = new KSIntCalculatorArgonSingleIonisation();
        aIntCalculator->SetName(fObject->GetName() + "_single_ionisation");
        aIntCalculator->SetTag(fObject->GetName());
        tToolBox.AddObject(aIntCalculator);
    }

    if (fObject->GetDoubleIonisation()) {
        aIntCalculator = new KSIntCalculatorArgonDoubleIonisation();
        aIntCalculator->SetName(fObject->GetName() + "_double_ionisation");
        aIntCalculator->SetTag(fObject->GetName());
        tToolBox.AddObject(aIntCalculator);
    }

    if (fObject->GetExcitation()) {
        for (unsigned int i = 0; i < 25; ++i) {
            std::stringstream tmp;
            tmp << (i + 1);
            aIntCalculator = new KSIntCalculatorArgonExcitation();
            aIntCalculator->SetName(fObject->GetName() + "_excitation_state_" + tmp.str());
            aIntCalculator->SetTag(fObject->GetName());
            static_cast<KSIntCalculatorArgonExcitation*>(aIntCalculator)->SetExcitationState(i + 1);
            tToolBox.AddObject(aIntCalculator);
        }
    }

    delete fObject;
    return true;
}
}  // namespace katrin

#endif
