#include "KSIntCalculatorArgon.h"
#include "KSInteractionsMessage.h"
#include "KSParticleFactory.h"
#include "KTextFile.h"

#include <cmath>
#include <fstream>
#include <limits>
#include <sstream>
using katrin::KTextFile;

#include "KThreeVector.hh"
using KGeoBag::KThreeVector;

#include "KConst.h"
#include "KRandom.h"
using katrin::KRandom;

namespace Kassiopeia
{
/////////////////////////////////////
/////		Excited Child		/////
/////////////////////////////////////
KSIntCalculatorArgonExcitation::KSIntCalculatorArgonExcitation()
{
    fExcitationState = 0;
    fDataFile = std::string("No file selected");
}

KSIntCalculatorArgonExcitation::KSIntCalculatorArgonExcitation(const KSIntCalculatorArgonExcitation& aCopy)
{
    delete fSupportingPoints;
    delete fParameters;

    fSupportingPoints = new std::map<double, double>(*aCopy.fSupportingPoints);
    fParameters = new std::vector<double>(*aCopy.fParameters);
    fDataFile = aCopy.fDataFile;
    fExcitationState = aCopy.fExcitationState;
}

KSIntCalculatorArgonExcitation::~KSIntCalculatorArgonExcitation() {}

KSIntCalculatorArgonExcitation* KSIntCalculatorArgonExcitation::Clone() const
{
    return new KSIntCalculatorArgonExcitation(*this);
}

double KSIntCalculatorArgonExcitation::GetUpperExtrapolation(const double&, std::map<double, double>::iterator&) const
{
    return 0;
}

void KSIntCalculatorArgonExcitation::InitializeComponent()
{
    // Use SetExcitationState to change fExcitationState
    if (fExcitationState > 0) {
        // Build file name for excitation state
        std::stringstream fileName;
        fileName << "argon_excitation_state-" << this->fExcitationState << "_cross-section.txt";

        // Set file name
        fDataFile = fileName.str();

        // Read file and create supporting points
        ComputeSupportingPoints(0);
    }
}

void KSIntCalculatorArgonExcitation::SetExcitationState(unsigned int aState)
{
    fExcitationState = aState;
}

void KSIntCalculatorArgonExcitation::ExecuteInteraction(const KSParticle& anInitialParticle, KSParticle& aFinalParticle,
                                                        KSParticleQueue&)
{
    double tInitialKineticEnergy = anInitialParticle.GetKineticEnergy_eV();
    KThreeVector tInitialDirection = anInitialParticle.GetMomentum();

    // outgoing primary

    // tLostKineticEnergy is equal to the threshold energy, which currently is the first
    // element in the data map.
    double tLostKineticEnergy = fSupportingPoints->begin()->first;
    double tPhi = KRandom::GetInstance().Uniform(0., 2. * katrin::KConst::Pi());
    // ToDo: tTheta?
    double tTheta = 0;

    //todo:://here now some fancy formulas from ferencs EH2scat

    KThreeVector tOrthogonalOne = tInitialDirection.Orthogonal();
    KThreeVector tOrthogonalTwo = tInitialDirection.Cross(tOrthogonalOne);
    KThreeVector tFinalDirection =
        tInitialDirection.Magnitude() *
        (sin(tTheta) * (cos(tPhi) * tOrthogonalOne.Unit() + sin(tPhi) * tOrthogonalTwo.Unit()) +
         cos(tTheta) * tInitialDirection.Unit());

    aFinalParticle = anInitialParticle;
    aFinalParticle.SetTime(anInitialParticle.GetTime());
    aFinalParticle.SetPosition(anInitialParticle.GetPosition());
    aFinalParticle.SetMomentum(tFinalDirection);
    aFinalParticle.SetKineticEnergy_eV(tInitialKineticEnergy - tLostKineticEnergy);
    aFinalParticle.AddLabel(GetName());

    return;
}

/////////////////////////////////
/////		Elastic			/////
/////////////////////////////////
KSIntCalculatorArgonElastic::KSIntCalculatorArgonElastic()
{
    fDataFile = std::string("argon_total_elastic_cross-section.txt");
}

KSIntCalculatorArgonElastic::KSIntCalculatorArgonElastic(const KSIntCalculatorArgonElastic& aCopy)
{
    delete fSupportingPoints;
    delete fParameters;

    fSupportingPoints = new std::map<double, double>(*aCopy.fSupportingPoints);
    fParameters = new std::vector<double>(*aCopy.fParameters);
    fDataFile = aCopy.fDataFile;
}

KSIntCalculatorArgonElastic* KSIntCalculatorArgonElastic::Clone() const
{
    return new KSIntCalculatorArgonElastic(*this);
}

KSIntCalculatorArgonElastic::~KSIntCalculatorArgonElastic() {}

void KSIntCalculatorArgonElastic::InitializeComponent()
{
    // numOfParameters: two for extrapolation.
    ComputeSupportingPoints(2);
}

void KSIntCalculatorArgonElastic::ExecuteInteraction(const KSParticle& anInitialParticle, KSParticle& aFinalParticle,
                                                     KSParticleQueue&)
{
    double tInitialKineticEnergy = anInitialParticle.GetKineticEnergy_eV();
    KThreeVector tInitialDirection = anInitialParticle.GetMomentum();

    // outgoing primary

    double tLostKineticEnergy = 0;
    double tTheta = 0;
    double tPhi;

    //todo:://here now some fancy formulas from ferencs EH2scat

    tPhi = KRandom::GetInstance().Uniform(0., 2. * katrin::KConst::Pi());

    KThreeVector tOrthogonalOne = tInitialDirection.Orthogonal();
    KThreeVector tOrthogonalTwo = tInitialDirection.Cross(tOrthogonalOne);
    KThreeVector tFinalDirection =
        tInitialDirection.Magnitude() *
        (sin(tTheta) * (cos(tPhi) * tOrthogonalOne.Unit() + sin(tPhi) * tOrthogonalTwo.Unit()) +
         cos(tTheta) * tInitialDirection.Unit());

    aFinalParticle = anInitialParticle;
    aFinalParticle.SetTime(anInitialParticle.GetTime());
    aFinalParticle.SetPosition(anInitialParticle.GetPosition());
    aFinalParticle.SetMomentum(tFinalDirection);
    aFinalParticle.SetKineticEnergy_eV(tInitialKineticEnergy - tLostKineticEnergy);
    aFinalParticle.AddLabel(GetName());

    return;
}

/////////////////////////////
/////		Mother		/////
/////////////////////////////
KSIntCalculatorArgon::KSIntCalculatorArgon()
{
    fSupportingPoints = new std::map<double, double>();
    fParameters = new std::vector<double>();
}

KSIntCalculatorArgon::~KSIntCalculatorArgon()
{
    delete fSupportingPoints;
    delete fParameters;
}

std::map<double, double>* KSIntCalculatorArgon::DEBUG_GetSupportingPoints()
{
    return fSupportingPoints;
}

double KSIntCalculatorArgon::GetInterpolation(const double& anEnergy, std::map<double, double>::iterator& point) const
{
    std::map<double, double>::iterator firstPoint = point;
    --firstPoint;

    double slope = (point->second - firstPoint->second) / (point->first - firstPoint->first);
    double y0 = firstPoint->second - slope * firstPoint->first;

    return slope * anEnergy + y0;
}

double KSIntCalculatorArgon::GetUpperExtrapolation(const double& anEnergy, std::map<double, double>::iterator&) const
{
    return fParameters->at(0) * pow(anEnergy, fParameters->at(1));
}

double KSIntCalculatorArgon::GetLowerExtrapolation(const double&, std::map<double, double>::iterator&) const
{
    return 0;
}

double KSIntCalculatorArgon::GetCrossSectionAt(const double& anEnergy) const
{
    // Zero? Interpolation? Extrapolation? Make a decision and execute it...

    std::map<double, double>::iterator point = fSupportingPoints->lower_bound(anEnergy);

    if (point == fSupportingPoints->begin()) {
        // Lower extrapolation
        return GetLowerExtrapolation(anEnergy, point);
    }
    else if (point == fSupportingPoints->end()) {
        // Upper extrapolation
        return GetUpperExtrapolation(anEnergy, point);
    }
    else {
        // Linear interpolation
        return GetInterpolation(anEnergy, point);
    }
}

void KSIntCalculatorArgon::ComputeSupportingPoints(unsigned int numOfParameters)
{
    // Clear supporting points
    fSupportingPoints->clear();

    // New read in
    KTextFile* tInputFile = katrin::CreateDataTextFile(fDataFile);

    if (tInputFile->Open(katrin::KFile::eRead)) {
        KSIntCalculatorArgonDataReader reader(tInputFile->File(), numOfParameters);
        if (!reader.Read()) {
            intmsg(eError) << "KSIntCalculatorArgon::ComputeSupportingPoints " << ret;
            intmsg(eError) << " Error while reading < " << tInputFile->GetName() << " > " << eom;
        }
        else {
            // First, copy parameters for extrapolation:
            fParameters->operator=(*reader.GetParameters());

            if (reader.GetData()->size() > 0) {
                // Copy data for interpolation.
                // It is maybe better to calculate equidistant energy points to find the right
                // interval earlier...
                fSupportingPoints->operator=(*reader.GetData());
            }
            else {
                intmsg(eError) << "KSIntCalculatorArgon::ComputeSupportingPoints " << ret;
                intmsg(eError) << " No found data in < " << tInputFile->GetName() << " > " << eom;
            }
        }

        tInputFile->Close();
    }
    else {
        intmsg(eError) << "KSIntCalculatorArgon::ComputeSupportingPoints " << ret;
        intmsg(eError) << " Cant open inputfile < " << tInputFile->GetName() << " > " << eom;
    }
}

void KSIntCalculatorArgon::CalculateCrossSection(const KSParticle& aParticle, double& aCrossSection)
{
    CalculateCrossSection(aParticle.GetKineticEnergy_eV(), aCrossSection);
    return;
}

void KSIntCalculatorArgon::CalculateCrossSection(const double anEnergy, double& aCrossSection)
{
    aCrossSection = this->GetCrossSectionAt(anEnergy);
}

/////////////////////////////////
/////	Single Ionization	/////
/////////////////////////////////
KSIntCalculatorArgonSingleIonisation::KSIntCalculatorArgonSingleIonisation()
{
    fDataFile = std::string("argon_total_single_ionization_cross-section.txt");
    fIonizationEnergy = 0;
}

KSIntCalculatorArgonSingleIonisation::KSIntCalculatorArgonSingleIonisation(
    const KSIntCalculatorArgonSingleIonisation& aCopy)
{
    delete fSupportingPoints;
    delete fParameters;

    fIonizationEnergy = aCopy.fIonizationEnergy;
    fSupportingPoints = new std::map<double, double>(*aCopy.fSupportingPoints);
    fParameters = new std::vector<double>(*aCopy.fParameters);
    fDataFile = aCopy.fDataFile;
}

KSIntCalculatorArgonSingleIonisation* KSIntCalculatorArgonSingleIonisation::Clone() const
{
    return new KSIntCalculatorArgonSingleIonisation(*this);
}

KSIntCalculatorArgonSingleIonisation::~KSIntCalculatorArgonSingleIonisation() {}

void KSIntCalculatorArgonSingleIonisation::InitializeComponent()
{
    // numOfParameters: two for extrapolation power law plus one for the ionization energy.
    ComputeSupportingPoints(3);

    // Get ionization energy:
    fIonizationEnergy = fParameters->back();

    // Remove this energy from parameter list:
    fParameters->pop_back();
}

void KSIntCalculatorArgonSingleIonisation::ExecuteInteraction(const KSParticle& anInitialParticle,
                                                              KSParticle& aFinalParticle, KSParticleQueue& aSecondaries)
{
    double tInitialKineticEnergy = anInitialParticle.GetKineticEnergy_eV();
    KThreeVector tInitialDirection = anInitialParticle.GetMomentum();

    // outgoing primary
    double tLostKineticEnergy = fIonizationEnergy + 0;
    double tTheta = 0;
    double tPhi;

    // ToDo: tLostKineticEnergy?

    //todo:://here now some fancy formulas from ferencs EH2scat

    // ToDo: tTheta?
    tPhi = KRandom::GetInstance().Uniform(0., 2. * katrin::KConst::Pi());

    KThreeVector tOrthogonalOne = tInitialDirection.Orthogonal();
    KThreeVector tOrthogonalTwo = tInitialDirection.Cross(tOrthogonalOne);
    KThreeVector tFinalDirection =
        tInitialDirection.Magnitude() *
        (sin(tTheta) * (cos(tPhi) * tOrthogonalOne.Unit() + sin(tPhi) * tOrthogonalTwo.Unit()) +
         cos(tTheta) * tInitialDirection.Unit());

    aFinalParticle = anInitialParticle;
    aFinalParticle.SetMomentum(tFinalDirection);
    aFinalParticle.SetKineticEnergy_eV(tInitialKineticEnergy - tLostKineticEnergy);
    aFinalParticle.AddLabel(GetName());

    // Outgoing secondary
    tTheta = acos(KRandom::GetInstance().Uniform(-1., 1.));
    tPhi = KRandom::GetInstance().Uniform(0., 2. * katrin::KConst::Pi());

    tOrthogonalOne = tInitialDirection.Orthogonal();
    tOrthogonalTwo = tInitialDirection.Cross(tOrthogonalOne);
    tFinalDirection = tInitialDirection.Magnitude() *
                      (sin(tTheta) * (cos(tPhi) * tOrthogonalOne.Unit() + sin(tPhi) * tOrthogonalTwo.Unit()) +
                       cos(tTheta) * tInitialDirection.Unit());

    KSParticle* tSecondary = KSParticleFactory::GetInstance().Create(11);
    (*tSecondary) = anInitialParticle;
    tSecondary->SetMomentum(tFinalDirection);
    tSecondary->SetKineticEnergy_eV(tLostKineticEnergy - fIonizationEnergy);
    tSecondary->AddLabel(GetName());

    aSecondaries.push_back(tSecondary);
}

/////////////////////////////////
/////	Double Ionization	/////
/////////////////////////////////
KSIntCalculatorArgonDoubleIonisation::KSIntCalculatorArgonDoubleIonisation()
{
    fIonizationEnergy = new std::vector<double>(2, 0);
    fDataFile = std::string("argon_total_double_ionization_cross-section.txt");
}

KSIntCalculatorArgonDoubleIonisation::KSIntCalculatorArgonDoubleIonisation(
    const KSIntCalculatorArgonDoubleIonisation& aCopy)
{
    delete fSupportingPoints;
    delete fParameters;

    fIonizationEnergy = new std::vector<double>(*aCopy.fIonizationEnergy);
    fSupportingPoints = new std::map<double, double>(*aCopy.fSupportingPoints);
    fParameters = new std::vector<double>(*aCopy.fParameters);
    fDataFile = aCopy.fDataFile;
}

KSIntCalculatorArgonDoubleIonisation* KSIntCalculatorArgonDoubleIonisation::Clone() const
{
    return new KSIntCalculatorArgonDoubleIonisation(*this);
}

KSIntCalculatorArgonDoubleIonisation::~KSIntCalculatorArgonDoubleIonisation()
{
    delete fIonizationEnergy;
}

void KSIntCalculatorArgonDoubleIonisation::InitializeComponent()
{
    // numOfParameters: two for extrapolation power law plus two for the ionization energies.
    ComputeSupportingPoints(4);

    // Get second ionization energy
    (*fIonizationEnergy)[1] = fParameters->back();

    // Remove this one
    this->fParameters->pop_back();

    // Get first ionization energy
    (*fIonizationEnergy)[0] = fParameters->back();

    // Remove this one
    fParameters->pop_back();
}

void KSIntCalculatorArgonDoubleIonisation::ExecuteInteraction(const KSParticle& anInitialParticle,
                                                              KSParticle& aFinalParticle, KSParticleQueue& aSecondaries)
{
    double tInitialKineticEnergy = anInitialParticle.GetKineticEnergy_eV();
    KThreeVector tInitialDirection = anInitialParticle.GetMomentum();

    // outgoing primary
    double tLostKineticEnergy = fIonizationEnergy->at(0) + fIonizationEnergy->at(1) + 0;
    double tTheta = 0;
    double tPhi;

    // ToDo: tLostKineticEnergy?

    //todo:://here now some fancy formulas from ferencs EH2scat

    // ToDo: tTheta?
    tPhi = KRandom::GetInstance().Uniform(0., 2. * katrin::KConst::Pi());

    KThreeVector tOrthogonalOne = tInitialDirection.Orthogonal();
    KThreeVector tOrthogonalTwo = tInitialDirection.Cross(tOrthogonalOne);
    KThreeVector tFinalDirection =
        tInitialDirection.Magnitude() *
        (sin(tTheta) * (cos(tPhi) * tOrthogonalOne.Unit() + sin(tPhi) * tOrthogonalTwo.Unit()) +
         cos(tTheta) * tInitialDirection.Unit());

    aFinalParticle = anInitialParticle;
    aFinalParticle.SetMomentum(tFinalDirection);
    aFinalParticle.SetKineticEnergy_eV(tInitialKineticEnergy - tLostKineticEnergy);
    aFinalParticle.AddLabel(GetName());

    // Outgoing secondaries
    for (int i = 0; i < 2; ++i) {
        tTheta = acos(KRandom::GetInstance().Uniform(-1., 1.));
        tPhi = KRandom::GetInstance().Uniform(0., 2. * katrin::KConst::Pi());

        tOrthogonalOne = tInitialDirection.Orthogonal();
        tOrthogonalTwo = tInitialDirection.Cross(tOrthogonalOne);
        tFinalDirection = tInitialDirection.Magnitude() *
                          (sin(tTheta) * (cos(tPhi) * tOrthogonalOne.Unit() + sin(tPhi) * tOrthogonalTwo.Unit()) +
                           cos(tTheta) * tInitialDirection.Unit());

        KSParticle* tSecondary = KSParticleFactory::GetInstance().Create(11);
        (*tSecondary) = anInitialParticle;
        tSecondary->SetMomentum(tFinalDirection);
        tSecondary->SetKineticEnergy_eV(tLostKineticEnergy - fIonizationEnergy->at(i));
        tSecondary->AddLabel(GetName());

        aSecondaries.push_back(tSecondary);
    }
}

/////////////////////////////////
/////		Data Reader 	/////
/////////////////////////////////

KSIntCalculatorArgonDataReader::KSIntCalculatorArgonDataReader(std::istream* aStream, unsigned int numOfParameters)
{
    fStream = aStream;
    fNumOfParameters = numOfParameters;
    fData = new std::map<double, double>();
    fParameters = new std::vector<double>();
}

KSIntCalculatorArgonDataReader::~KSIntCalculatorArgonDataReader()
{
    delete fData;
    delete fParameters;
}

std::vector<double>* KSIntCalculatorArgonDataReader::GetParameters()
{
    return fParameters;
}

std::map<double, double>* KSIntCalculatorArgonDataReader::GetData()
{
    return fData;
}

bool KSIntCalculatorArgonDataReader::Read()
{
    if (!fStream->good()) {
        return false;
    }

    enum Mode
    {
        READ_PARAMETER,
        READ_DATA
    };

    Mode mode = fNumOfParameters > 0 ? READ_PARAMETER : READ_DATA;
    int pos;
    double value, secondValue;
    bool secondValueReaded = false;
    char c;
    while ((*fStream) >> c) {
        // Skip comments
        if ('#' == c) {
            fStream->ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        }
        else {
            pos = fStream->tellg();
            fStream->seekg(pos - 1);

            // Read number
            (*fStream) >> value;

            switch (mode) {
                // The first fNumOfParameters numbers in file are parameters
                // for extrapolation or something. After that it will read data
                // points for interpolation.
                case READ_PARAMETER:
                    fParameters->push_back(value);

                    if (fParameters->size() >= fNumOfParameters) {
                        mode = READ_DATA;
                    }
                    break;

                    // Read data
                case READ_DATA:
                    if (!secondValueReaded) {
                        secondValue = value;
                        secondValueReaded = true;
                    }
                    else {
                        // Now we have two values. "value" stands for the cross section
                        // and "secondValue" for the energy. Now we could store this data
                        // point into a map:
                        fData->insert(std::pair<double, double>(secondValue, value));
                        secondValueReaded = false;
                    }
                    break;
            }
        }
    }

    if (secondValueReaded) {
        return false;
    }

    return true;
}

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

#include "KToolbox.h"

using namespace Kassiopeia;
namespace katrin
{

template<> KSIntCalculatorArgonBuilder::~KComplexElement() {}

static int sKSIntCalculatorArgonStructure = KSIntCalculatorArgonBuilder::Attribute<std::string>("name") +
                                            KSIntCalculatorArgonBuilder::Attribute<bool>("elastic") +
                                            KSIntCalculatorArgonBuilder::Attribute<bool>("excitation") +
                                            KSIntCalculatorArgonBuilder::Attribute<bool>("single_ionisation") +
                                            KSIntCalculatorArgonBuilder::Attribute<bool>("double_ionisation");

static int sToolboxKSIntCalculatorArgon =
    KToolboxBuilder::ComplexElement<KSIntCalculatorArgonData>("ksint_calculator_argon");

}  // namespace katrin
