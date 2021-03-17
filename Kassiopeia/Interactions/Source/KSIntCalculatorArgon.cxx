#include "KSIntCalculatorArgon.h"

#include "KSInteractionsMessage.h"
#include "KSParticleFactory.h"
#include "KTextFile.h"
#include "KThreeVector.hh"

#include <cassert>
#include <cmath>
#include <fstream>
#include <limits>
#include <sstream>
using KGeoBag::KThreeVector;

#include "KConst.h"
#include "KRandom.h"

using namespace katrin;

namespace Kassiopeia
{

/////////////////////////////
/////		Mother		/////
/////////////////////////////
KSIntCalculatorArgon::KSIntCalculatorArgon()
{
    fSupportingPointsTotalCrossSection = new std::map<double, double>();
    fParametersTotalCrossSection = new std::vector<double>();
    fDifferentialCrossSectionInterpolator = new KMathBilinearInterpolator<double>();
}

KSIntCalculatorArgon::~KSIntCalculatorArgon()
{
    delete fSupportingPointsTotalCrossSection;
    delete fParametersTotalCrossSection;
    delete fDifferentialCrossSectionInterpolator;
}

double KSIntCalculatorArgon::GetInterpolationForTotalCrossSection(const double& anEnergy,
                                                                  std::map<double, double>::iterator& point) const
{
    assert(anEnergy == anEnergy);
    assert(anEnergy >= 0.);
    auto firstPoint = point;
    --firstPoint;

    double slope = (point->second - firstPoint->second) / (point->first - firstPoint->first);
    double y0 = firstPoint->second - slope * firstPoint->first;

    return slope * anEnergy + y0;
}

double KSIntCalculatorArgon::GetUpperExtrapolationForTotalCrossSection(const double& anEnergy,
                                                                       std::map<double, double>::iterator&) const
{
    assert(anEnergy == anEnergy);
    assert(anEnergy >= 0.);
    return fParametersTotalCrossSection->at(0) * pow(anEnergy, fParametersTotalCrossSection->at(1));
}

double KSIntCalculatorArgon::GetLowerExtrapolationForTotalCrossSection(const double&,
                                                                       std::map<double, double>::iterator&) const
{
    return 0;
}

double KSIntCalculatorArgon::GetTheta(const double& anEnergy) const
{
    assert(anEnergy == anEnergy);
    assert(anEnergy >= 0.);
    double sigma_max = GetDifferentialCrossSectionAt(anEnergy, 5.);
    sigma_max *= 1.5;
    double sigma_min = GetDifferentialCrossSectionAt(anEnergy, 180.);
    sigma_min *= 0.9;
    double tTheta_deg;

    while (true) {
        tTheta_deg = KRandom::GetInstance().Uniform(0., 180., false, true);
        if (KRandom::GetInstance().Uniform(sigma_min, sigma_max, false, true) <
            GetDifferentialCrossSectionAt(anEnergy, tTheta_deg)

        )
            break;
    }

    return tTheta_deg * KConst::Pi() / 180.;
}

double KSIntCalculatorArgon::GetTotalCrossSectionAt(const double& anEnergy) const
{
    assert(anEnergy == anEnergy);
    assert(anEnergy >= 0.);
    // Zero? Interpolation? Extrapolation? Make a decision and execute it...

    auto point = fSupportingPointsTotalCrossSection->lower_bound(anEnergy);

    if (point == fSupportingPointsTotalCrossSection->begin()) {
        // Lower extrapolation
        return GetLowerExtrapolationForTotalCrossSection(anEnergy, point);
    }
    else if (point == fSupportingPointsTotalCrossSection->end()) {
        // Upper extrapolation
        return GetUpperExtrapolationForTotalCrossSection(anEnergy, point);
    }
    else {
        // Linear interpolation
        return GetInterpolationForTotalCrossSection(anEnergy, point);
    }
}

double KSIntCalculatorArgon::GetDifferentialCrossSectionAt(const double& anEnergy, const double& anAngle) const
{
    assert(anEnergy == anEnergy);
    assert(anEnergy >= 0.);
    assert(anAngle == anAngle);
    assert(anAngle >= 0.);
    double maxEnergy = (--fDifferentialCrossSectionInterpolator->GetPoints()->end())->first;

    if (anEnergy > maxEnergy) {
        double t = anEnergy / (2. * KConst::ERyd_eV());
        double c = cos(anAngle * KConst::Pi() / 180.);
        double k2 = 4. * (t * (1. - c));

        return 1.85099E-19 * 4.0 * (8.0 + k2) * (8.0 + k2) / pow(4.0 + k2, 4.0);
    }
    else {
        return fDifferentialCrossSectionInterpolator->GetValue(anEnergy, anAngle);
    }
}

void KSIntCalculatorArgon::InitializeDifferentialCrossSection(unsigned int numOfParameters)
{
    intmsg(eNormal) << this->GetName() << " Initialize Diff CrossSection" << eom;
    fDifferentialCrossSectionInterpolator->Reset();

    // Now read in
    KTextFile* tInputFile = KTextFile::CreateDataTextFile(fDataFileDifferentialCrossSection);

    if (tInputFile->Open(KFile::eRead)) {
        KSIntCalculatorArgonDifferentialCrossSectionReader reader(tInputFile->File(), numOfParameters);
        if (!reader.Read()) {
            intmsg(eError) << "KIntCalculatorArgon::InitializeDifferentialCrossSection " << ret;
            intmsg(eError) << " Error while reading < " << tInputFile->GetName() << " > " << eom;
        }
        else {
            // First, copy parameters for extrapolation:
            //fParametersTotalCrossSection->operator=(*reader.GetParameters());

            if (reader.GetData()->size() > 0) {
                // Copy data to the interpolation
                std::map<double*, double>* data = reader.GetData();

                for (auto& i : *data) {
                    fDifferentialCrossSectionInterpolator->AddPoint(i.first[0], i.first[1], i.second);
                }
            }
            else {
                intmsg(eError) << "KIntCalculatorArgon::InitializeDifferentialCrossSection " << ret;
                intmsg(eError) << " No found data in < " << tInputFile->GetName() << " > " << eom;
            }
        }

        tInputFile->Close();
    }
    else {
        intmsg(eError) << "KIntCalculatorArgon::InitializeTotalCrossSection " << ret;
        intmsg(eError) << " Cant open inputfile < " << tInputFile->GetName() << " > " << eom;
    }
}

void KSIntCalculatorArgon::InitializeTotalCrossSection(unsigned int numOfParameters)
{
    intmsg(eNormal) << this->GetName() << " Initialize Total CrossSection" << eom;
    // Clear supporting points
    fSupportingPointsTotalCrossSection->clear();

    // New read in
    KTextFile* tInputFile = KTextFile::CreateDataTextFile(fDataFileTotalCrossSection);

    if (tInputFile->Open(KFile::eRead)) {
        KSIntCalculatorArgonTotalCrossSectionReader reader(tInputFile->File(), numOfParameters);
        if (!reader.Read()) {
            intmsg(eError) << "KIntCalculatorArgon::InitializeTotalCrossSection " << ret;
            intmsg(eError) << " Error while reading < " << tInputFile->GetName() << " > " << eom;
        }
        else {
            // First, copy parameters for extrapolation:
            fParametersTotalCrossSection->operator=(*reader.GetParameters());

            if (reader.GetData()->size() > 0) {
                // Copy data for interpolation.
                // It is maybe better to calculate equidistant energy points to find the right
                // interval earlier...
                fSupportingPointsTotalCrossSection->operator=(*reader.GetData());
            }
            else {
                intmsg(eError) << "KIntCalculatorArgon::InitializeTotalCrossSection " << ret;
                intmsg(eError) << " No found data in < " << tInputFile->GetName() << " > " << eom;
            }
        }

        tInputFile->Close();
    }
    else {
        intmsg(eError) << "KIntCalculatorArgon::InitializeTotalCrossSection " << ret;
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
    assert(anEnergy == anEnergy);
    assert(anEnergy >= 0.);
    aCrossSection = this->GetTotalCrossSectionAt(anEnergy);
}

/////////////////////////////////////
/////		Excited Child		/////
/////////////////////////////////////
KSIntCalculatorArgonExcitation::KSIntCalculatorArgonExcitation()
{
    fSupportingPointsTotalCrossSection = new std::map<double, double>();
    fParametersTotalCrossSection = new std::vector<double>();
    fDifferentialCrossSectionInterpolator = new KMathBilinearInterpolator<double>();
    fExcitationState = 0;
    fDataFileTotalCrossSection = std::string("No file selected");

    fDataFileDifferentialCrossSection = std::string("No file selected");
}

KSIntCalculatorArgonExcitation::KSIntCalculatorArgonExcitation(const KSIntCalculatorArgonExcitation& aCopy) :
    KSComponent(aCopy)
{
    delete fSupportingPointsTotalCrossSection;
    delete fParametersTotalCrossSection;

    fSupportingPointsTotalCrossSection = new std::map<double, double>(*aCopy.fSupportingPointsTotalCrossSection);
    fParametersTotalCrossSection = new std::vector<double>(*aCopy.fParametersTotalCrossSection);
    fDataFileTotalCrossSection = aCopy.fDataFileTotalCrossSection;
    fExcitationState = aCopy.fExcitationState;
    fDataFileDifferentialCrossSection = aCopy.fDataFileDifferentialCrossSection;
}

KSIntCalculatorArgonExcitation::~KSIntCalculatorArgonExcitation() = default;

KSIntCalculatorArgonExcitation* KSIntCalculatorArgonExcitation::Clone() const
{
    return new KSIntCalculatorArgonExcitation(*this);
}

void KSIntCalculatorArgonExcitation::InitializeComponent()
{
    // Use SetExcitationState to change fExcitationState
    if (fExcitationState > 0) {
        // Build file name for excitation state
        std::stringstream TotalCrossSectionFileName;
        TotalCrossSectionFileName << "argon_excitation_state-" << this->fExcitationState << "_cross-section.txt";

        // Set file name
        fDataFileTotalCrossSection = TotalCrossSectionFileName.str();

        // Read file and create supporting points
        InitializeTotalCrossSection(2);

        std::stringstream DiffCrossSectionFileName;
        DiffCrossSectionFileName << "argon_differential_cross-section_excitation_state." << this->fExcitationState
                                 << ".txt";

        fDataFileDifferentialCrossSection = DiffCrossSectionFileName.str();
        InitializeDifferentialCrossSection(0);
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

    intmsg_debug("< " << this->GetName() << " >:ExecuteInteraction() " << ret);
    intmsg_debug("tInitialKineticEnergy: " << tInitialKineticEnergy << ret);
    intmsg_debug("tInitialDirection X: " << tInitialDirection.X() << " Y: " << tInitialDirection.Y()
                                         << " Z: " << tInitialDirection.Z() << ret)
        assert(tInitialKineticEnergy == tInitialKineticEnergy);
    assert(tInitialKineticEnergy >= 0.);

    // outgoing primary
    double tTheta = GetTheta(tInitialKineticEnergy);
    intmsg_debug("tTheta: " << tTheta << ret);
    assert(tTheta == tTheta);
    assert(tTheta >= 0.);
    double tLostKineticEnergy = GetEnergyLoss(tInitialKineticEnergy, tTheta);
    intmsg_debug("tLostKineticEnergy: " << tLostKineticEnergy << ret);
    assert(tLostKineticEnergy == tLostKineticEnergy);
    assert(tLostKineticEnergy >= 0.);
    double tPhi = KRandom::GetInstance().Uniform(0., 2. * KConst::Pi());
    intmsg_debug("tPhi: " << tPhi << ret << ret);

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

    fStepNInteractions = 1;
    fStepEnergyLoss = tLostKineticEnergy;
    fStepAngularChange = tTheta * 180. / KConst::Pi();

    intmsg_debug("FinalParticle: " << ret);
    intmsg_debug("Time: " << aFinalParticle.GetTime() << ret);
    intmsg_debug("Position: " << aFinalParticle.GetPosition().X() << " " << aFinalParticle.GetPosition().Y() << " "
                              << aFinalParticle.GetPosition().Z() << ret);
    intmsg_debug("Momentum: " << aFinalParticle.GetMomentum().X() << " " << aFinalParticle.GetMomentum().Y() << " "
                              << aFinalParticle.GetMomentum().Z() << ret);
    intmsg_debug("KineticEnergy: " << aFinalParticle.GetKineticEnergy_eV() << ret);
    intmsg_debug("=============<" << this->GetName() << ">:ExecuteInteraction()" << eom)

        return;
}

double KSIntCalculatorArgonExcitation::GetEnergyLoss(const double&, const double&) const
{
    // The lost kinetic energy is equal to the threshold energy, which currently is the first
    // element in the data map.
    return fSupportingPointsTotalCrossSection->begin()->first;
}

double KSIntCalculatorArgonExcitation::GetDifferentialCrossSectionAt(const double& anEnergy,
                                                                     const double& anAngle) const
{
    assert(anEnergy == anEnergy);
    assert(anEnergy >= 0.);
    assert(anAngle == anAngle);
    assert(anAngle >= 0.);

    double maxEnergy = (--fDifferentialCrossSectionInterpolator->GetPoints()->end())->first;

    if (anEnergy > maxEnergy) {
        return fDifferentialCrossSectionInterpolator->GetValue(maxEnergy - 1., anAngle) /
               this->GetTotalCrossSectionAt(maxEnergy - 1.) * (this->GetTotalCrossSectionAt(anEnergy));
    }
    else {
        return fDifferentialCrossSectionInterpolator->GetValue(anEnergy, anAngle);
    }
}

void KSIntCalculatorArgonExcitation::InitializeDifferentialCrossSection(unsigned int numOfParameters)
{
    //This Method differs from the Base Class method only in the use of Readlx() instead of Read()
    //to use the files from the LxCat data base
    fDifferentialCrossSectionInterpolator->Reset();

    // Now read in
    KTextFile* tInputFile = KTextFile::CreateDataTextFile(fDataFileDifferentialCrossSection);

    if (tInputFile->Open(KFile::eRead)) {
        KSIntCalculatorArgonDifferentialCrossSectionReader reader(tInputFile->File(), numOfParameters);
        if (!reader.Readlx()) {
            intmsg(eError) << "KIntCalculatorArgon::InitializeDifferentialCrossSection " << ret;
            intmsg(eError) << " Error while reading < " << tInputFile->GetName() << " > " << eom;
        }
        else {
            // First, copy parameters for extrapolation:
            //fParametersTotalCrossSection->operator=(*reader.GetParameters());

            if (reader.GetData()->size() > 0) {
                // Copy data to the interpolation
                std::map<double*, double>* data = reader.GetData();

                for (auto& i : *data) {
                    fDifferentialCrossSectionInterpolator->AddPoint(i.first[0], i.first[1], i.second);
                }
            }
            else {
                intmsg(eError) << "KIntCalculatorArgon::InitializeDifferentialCrossSection " << ret;
                intmsg(eError) << " No found data in < " << tInputFile->GetName() << " > " << eom;
            }
        }

        tInputFile->Close();
    }
    else {
        intmsg(eError) << "KIntCalculatorArgon::InitializeTotalCrossSection " << ret;
        intmsg(eError) << " Cant open inputfile < " << tInputFile->GetName() << " > " << ret;
        intmsg(eError) << "fDataFileDifferentialCrossSection: " << fDataFileDifferentialCrossSection << eom;
    }
}

/////////////////////////////////
/////		Elastic			/////
/////////////////////////////////
KSIntCalculatorArgonElastic::KSIntCalculatorArgonElastic()
{
    fDataFileTotalCrossSection = std::string("argon_total_elastic_cross-section.txt");
    fDataFileDifferentialCrossSection = std::string("argon_differential_elastic_cross-section.txt");
}

KSIntCalculatorArgonElastic::KSIntCalculatorArgonElastic(const KSIntCalculatorArgonElastic& aCopy) : KSComponent(aCopy)
{
    delete fSupportingPointsTotalCrossSection;
    delete fParametersTotalCrossSection;

    fSupportingPointsTotalCrossSection = new std::map<double, double>(*aCopy.fSupportingPointsTotalCrossSection);
    fParametersTotalCrossSection = new std::vector<double>(*aCopy.fParametersTotalCrossSection);
    fDataFileTotalCrossSection = aCopy.fDataFileTotalCrossSection;
    fDataFileDifferentialCrossSection = aCopy.fDataFileDifferentialCrossSection;
}

KSIntCalculatorArgonElastic* KSIntCalculatorArgonElastic::Clone() const
{
    return new KSIntCalculatorArgonElastic(*this);
}

KSIntCalculatorArgonElastic::~KSIntCalculatorArgonElastic() = default;

void KSIntCalculatorArgonElastic::InitializeComponent()
{
    intmsg_debug("initializing argon elastic calculator" << eom);
    // numOfParameters: two for extrapolation.
    InitializeTotalCrossSection(2);
    InitializeDifferentialCrossSection(0);
}

double KSIntCalculatorArgonElastic::GetDifferentialCrossSectionAt(const double& anEnergy, const double& anAngle) const
{
    assert(anEnergy == anEnergy);
    assert(anEnergy >= 0.);
    assert(anAngle == anAngle);
    assert(anAngle >= 0.);
    double maxEnergy = (--fDifferentialCrossSectionInterpolator->GetPoints()->end())->first;

    if (anEnergy > maxEnergy) {
        double k2 = 4. * anEnergy / (2. * KConst::ERyd_eV()) * (1. - cos(anAngle * KConst::Pi() / 180.));

        return 1.85099E-19 * 4.0 * (8.0 + k2) * (8.0 + k2) / pow(4.0 + k2, 4.0);
    }
    else {
        return fDifferentialCrossSectionInterpolator->GetValue(anEnergy, anAngle);
    }
}

double KSIntCalculatorArgonElastic::GetEnergyLoss(const double& anEnergy, const double& aTheta) const
{
    assert(anEnergy == anEnergy);
    assert(anEnergy >= 0.);
    assert(aTheta == aTheta);
    assert(aTheta >= 0.);

    double anEloss = 0;
    // ArMolMass = 1/alpha^2 * #nuclei * m_p/m_e
    double ArMolMass = 1.38e9;
    double emass = 1. / (KConst::Alpha() * KConst::Alpha());
    double cosTheta = cos(aTheta);

    anEloss = 2. * emass / ArMolMass * (1. - cosTheta) * anEnergy;

    assert(anEloss == anEloss);
    assert(anEloss >= 0.);

    return anEloss;
}

void KSIntCalculatorArgonElastic::ExecuteInteraction(const KSParticle& anInitialParticle, KSParticle& aFinalParticle,
                                                     KSParticleQueue&)
{

    double tInitialKineticEnergy = anInitialParticle.GetKineticEnergy_eV();
    KThreeVector tInitialDirection = anInitialParticle.GetMomentum();

    intmsg_debug("< " << this->GetName() << " >:ExecuteInteraction() " << ret);
    intmsg_debug("tInitialKineticEnergy: " << tInitialKineticEnergy << ret);
    intmsg_debug("tInitialDirection X: " << tInitialDirection.X() << " Y: " << tInitialDirection.Y()
                                         << " Z: " << tInitialDirection.Z() << ret)
        assert(tInitialKineticEnergy == tInitialKineticEnergy);
    assert(tInitialKineticEnergy >= 0.);

    // outgoing primary

    double tTheta = GetTheta(tInitialKineticEnergy);
    intmsg_debug("tTheta: " << tTheta << ret);
    double tLostKineticEnergy = GetEnergyLoss(tInitialKineticEnergy, tTheta);
    intmsg_debug("tLostKineticEnergy: " << tLostKineticEnergy << ret);
    double tPhi = KRandom::GetInstance().Uniform(0., 2. * KConst::Pi());
    intmsg_debug("tPhi: " << tPhi << ret << ret);

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

    intmsg_debug("FinalParticle: " << ret);
    intmsg_debug("Time: " << aFinalParticle.GetTime() << ret);
    intmsg_debug("Position: " << aFinalParticle.GetPosition().X() << " " << aFinalParticle.GetPosition().Y() << " "
                              << aFinalParticle.GetPosition().Z() << ret);
    intmsg_debug("Momentum: " << aFinalParticle.GetMomentum().X() << " " << aFinalParticle.GetMomentum().Y() << " "
                              << aFinalParticle.GetMomentum().Z() << ret);
    intmsg_debug("KineticEnergy: " << aFinalParticle.GetKineticEnergy_eV() << ret);
    intmsg_debug("=============<" << this->GetName() << ">:ExecuteInteraction()" << eom)

        fStepNInteractions = 1;
    fStepEnergyLoss = tLostKineticEnergy;

    return;
}

/////////////////////////////////
/////	Single Ionization	/////
/////////////////////////////////
KSIntCalculatorArgonSingleIonisation::KSIntCalculatorArgonSingleIonisation()
{
    fDataFileTotalCrossSection = std::string("argon_total_single_ionization_cross-section.txt");
    // Ionization energy: 15.759610
    // National Institut of Standards and Technology, Physical Meas. Laboratory,
    // http://physics.nist.gov/PhysRefData/Handbook/Tables/argontable1.htm, Nov. 22, 2013
    fIonizationEnergy = 0.;  // Is read from file and set in Initialize()
    DiffCrossCalculator = new KSIntCalculatorHydrogenIonisation();
}

KSIntCalculatorArgonSingleIonisation::KSIntCalculatorArgonSingleIonisation(
    const KSIntCalculatorArgonSingleIonisation& aCopy) :
    KSComponent(aCopy)
{
    delete fSupportingPointsTotalCrossSection;
    delete fParametersTotalCrossSection;

    fIonizationEnergy = aCopy.fIonizationEnergy;

    fSupportingPointsTotalCrossSection = new std::map<double, double>(*aCopy.fSupportingPointsTotalCrossSection);

    fParametersTotalCrossSection = new std::vector<double>(*aCopy.fParametersTotalCrossSection);

    fDataFileTotalCrossSection = aCopy.fDataFileTotalCrossSection;
}

KSIntCalculatorArgonSingleIonisation* KSIntCalculatorArgonSingleIonisation::Clone() const
{
    return new KSIntCalculatorArgonSingleIonisation(*this);
}

KSIntCalculatorArgonSingleIonisation::~KSIntCalculatorArgonSingleIonisation() = default;

void KSIntCalculatorArgonSingleIonisation::InitializeComponent()
{
    // numOfParameters: two for extrapolation power law plus one for the ionization energy.
    InitializeTotalCrossSection(3);

    // Get ionization energy:
    fIonizationEnergy = fParametersTotalCrossSection->back();

    // Remove this energy from parameter list:
    fParametersTotalCrossSection->pop_back();
}

void KSIntCalculatorArgonSingleIonisation::ExecuteInteraction(const KSParticle& anInitialParticle,
                                                              KSParticle& aFinalParticle, KSParticleQueue& aSecondaries)
{
    double tInitialKineticEnergy = anInitialParticle.GetKineticEnergy_eV();
    KThreeVector tInitialDirection = anInitialParticle.GetMomentum();

    intmsg_debug("< " << this->GetName() << " >:ExecuteInteraction() " << ret);
    intmsg_debug("tInitialKineticEnergy: " << tInitialKineticEnergy << ret);
    intmsg_debug("tInitialDirection X: " << tInitialDirection.X() << " Y: " << tInitialDirection.Y()
                                         << " Z: " << tInitialDirection.Z() << ret)

        // outgoing primary
        double tTheta = 0.;
    double tLostKineticEnergy = GetEnergyLoss(tInitialKineticEnergy, tTheta);
    intmsg_debug("tLostKineticEnergy: " << tLostKineticEnergy << ret);
    tTheta = GetTheta(tInitialKineticEnergy, tLostKineticEnergy);
    intmsg_debug("tTheta: " << tTheta << ret);
    double tPhi = KRandom::GetInstance().Uniform(0., 2. * KConst::Pi());
    intmsg_debug("tPhi: " << tPhi << ret << ret);

    KThreeVector tOrthogonalOne = tInitialDirection.Orthogonal();
    KThreeVector tOrthogonalTwo = tInitialDirection.Cross(tOrthogonalOne);
    KThreeVector tFinalDirection =
        tInitialDirection.Magnitude() *
        (sin(tTheta) * (cos(tPhi) * tOrthogonalOne.Unit() + sin(tPhi) * tOrthogonalTwo.Unit()) +
         cos(tTheta) * tInitialDirection.Unit());

    aFinalParticle = anInitialParticle;
    aFinalParticle.SetMomentum(tFinalDirection);
    aFinalParticle.SetKineticEnergy_eV(tInitialKineticEnergy - tLostKineticEnergy);

    intmsg_debug("FinalParticle: " << ret);
    intmsg_debug("Time: " << aFinalParticle.GetTime() << ret);
    intmsg_debug("Position: " << aFinalParticle.GetPosition().X() << " " << aFinalParticle.GetPosition().Y() << " "
                              << aFinalParticle.GetPosition().Z() << ret);
    intmsg_debug("Momentum: " << aFinalParticle.GetMomentum().X() << " " << aFinalParticle.GetMomentum().Y() << " "
                              << aFinalParticle.GetMomentum().Z() << ret);
    intmsg_debug("KineticEnergy: " << aFinalParticle.GetKineticEnergy_eV() << ret);

    // Outgoing secondary
    tTheta = acos(KRandom::GetInstance().Uniform(-1., 1.));
    tPhi = KRandom::GetInstance().Uniform(0., 2. * KConst::Pi());

    tOrthogonalOne = tInitialDirection.Orthogonal();
    tOrthogonalTwo = tInitialDirection.Cross(tOrthogonalOne);
    tFinalDirection = tInitialDirection.Magnitude() *
                      (sin(tTheta) * (cos(tPhi) * tOrthogonalOne.Unit() + sin(tPhi) * tOrthogonalTwo.Unit()) +
                       cos(tTheta) * tInitialDirection.Unit());

    KSParticle* tSecondary = KSParticleFactory::GetInstance().Create(11);
    (*tSecondary) = anInitialParticle;
    tSecondary->SetMomentum(tFinalDirection);
    tSecondary->SetKineticEnergy_eV(tLostKineticEnergy - fIonizationEnergy);
    tSecondary->SetLabel(GetName());

    intmsg_debug("SecondaryParticle: " << ret);
    intmsg_debug("Time: " << tSecondary->GetTime() << ret);
    intmsg_debug("Position: " << tSecondary->GetPosition().X() << " " << tSecondary->GetPosition().Y() << " "
                              << tSecondary->GetPosition().Z() << ret);
    intmsg_debug("Momentum: " << tSecondary->GetMomentum().X() << " " << tSecondary->GetMomentum().Y() << " "
                              << tSecondary->GetMomentum().Z() << ret);
    intmsg_debug("KineticEnergy: " << tSecondary->GetKineticEnergy_eV() << ret);
    intmsg_debug("=============<" << this->GetName() << ">:ExecuteInteraction()" << eom)
        aSecondaries.push_back(tSecondary);

    fStepNInteractions = 1;
    fStepEnergyLoss = tLostKineticEnergy;
    return;
}

double KSIntCalculatorArgonSingleIonisation::GetEnergyLoss(const double& anEnergy, const double&) const
{
    assert(anEnergy == anEnergy);
    assert(anEnergy > fIonizationEnergy);

    double tmpSecEnergy;
    double tmpMaxCross = KConst::BohrRadiusSquared() * 13.2 / anEnergy * log((anEnergy + 120.0 / anEnergy) / 15.76) *
                         10.3 * 10.3 / (pow(0. - 2. + 100. / (10. + anEnergy), 2) + 10.3 * 10.3);

    while (true) {
        tmpSecEnergy = KRandom::GetInstance().Uniform(0., (anEnergy - fIonizationEnergy) / 2., false, true);

        double tmpDiffCross = KConst::BohrRadiusSquared() * 13.2 / anEnergy *
                              log((anEnergy + 120.0 / anEnergy) / 15.76) * 10.3 * 10.3 /
                              (pow(tmpSecEnergy - 2. + 100. / (10. + anEnergy), 2) + 10.3 * 10.3);

        if (KRandom::GetInstance().Uniform(0., tmpMaxCross, false, true) < tmpDiffCross)
            break;
    }
    assert(tmpSecEnergy == tmpSecEnergy);
    assert(tmpSecEnergy >= 0.);

    return fIonizationEnergy + tmpSecEnergy;
}

double KSIntCalculatorArgonSingleIonisation::GetTheta(const double& anEnergy, const double& anEloss) const
{
    assert(anEnergy == anEnergy);
    assert(anEnergy >= 0.);
    assert(anEloss == anEloss);
    assert(anEloss >= 0.);
    double sigma_max = GetDifferentialCrossSectionAt(anEnergy, 0., anEloss);
    sigma_max *= 1.5;
    double sigma_min = GetDifferentialCrossSectionAt(anEnergy, 180., anEloss);
    sigma_min *= 0.9;
    double tTheta_deg;

    while (true) {
        tTheta_deg = KRandom::GetInstance().Uniform(0., 180., false, true);

        double tsigma = GetDifferentialCrossSectionAt(anEnergy, tTheta_deg, anEloss);

        if (KRandom::GetInstance().Uniform(sigma_min, sigma_max, false, true) < tsigma) {
            break;
        }
    }
    assert(tTheta_deg == tTheta_deg);
    assert(tTheta_deg >= 0. && tTheta_deg <= 180.);

    return tTheta_deg * KConst::Pi() / 180.;
}

double KSIntCalculatorArgonSingleIonisation::GetDifferentialCrossSectionAt(const double& anEnergy,
                                                                           const double& anAngle,
                                                                           const double& anEloss) const
{
    assert(anEnergy == anEnergy);
    assert(anEnergy >= 0.);
    assert(anEloss == anEloss);
    assert(anEloss >= 0.);
    assert(anAngle == anAngle);
    assert(anAngle >= 0.);

    double aCrossection;
    DiffCrossCalculator->CalculateDoublyDifferentialCrossSection(anEnergy / fIonizationEnergy,
                                                                 (anEnergy - anEloss) / fIonizationEnergy,
                                                                 cos(anAngle * KConst::Pi() / 180.),
                                                                 aCrossection);

    double wrongtotalcross;
    DiffCrossCalculator->CalculateCrossSection(anEnergy, wrongtotalcross);

    return aCrossection / wrongtotalcross * this->GetTotalCrossSectionAt(anEnergy);
}

/////////////////////////////////
/////	Double Ionization	/////
/////////////////////////////////
KSIntCalculatorArgonDoubleIonisation::KSIntCalculatorArgonDoubleIonisation()
{
    fIonizationEnergy = new std::vector<double>(2, 0);
    fDataFileTotalCrossSection = std::string("argon_total_double_ionization_cross-section.txt");
    DiffCrossCalculator = new KSIntCalculatorHydrogenIonisation();
}

KSIntCalculatorArgonDoubleIonisation::KSIntCalculatorArgonDoubleIonisation(
    const KSIntCalculatorArgonDoubleIonisation& aCopy) :
    KSComponent(aCopy)
{
    delete fSupportingPointsTotalCrossSection;
    delete fParametersTotalCrossSection;

    fIonizationEnergy = new std::vector<double>(*aCopy.fIonizationEnergy);

    fSupportingPointsTotalCrossSection = new std::map<double, double>(*aCopy.fSupportingPointsTotalCrossSection);

    fParametersTotalCrossSection = new std::vector<double>(*aCopy.fParametersTotalCrossSection);

    fDataFileTotalCrossSection = aCopy.fDataFileTotalCrossSection;
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
    InitializeTotalCrossSection(4);

    // Get second ionization energy
    (*fIonizationEnergy)[1] = fParametersTotalCrossSection->back();

    // Remove this one
    this->fParametersTotalCrossSection->pop_back();

    // Get first ionization energy
    (*fIonizationEnergy)[0] = fParametersTotalCrossSection->back();

    // Remove this one
    fParametersTotalCrossSection->pop_back();
}

void KSIntCalculatorArgonDoubleIonisation::ExecuteInteraction(const KSParticle& anInitialParticle,
                                                              KSParticle& aFinalParticle, KSParticleQueue& aSecondaries)
{
    double tInitialKineticEnergy = anInitialParticle.GetKineticEnergy_eV();
    KThreeVector tInitialDirection = anInitialParticle.GetMomentum();
    assert(tInitialKineticEnergy == tInitialKineticEnergy);
    assert(tInitialKineticEnergy >= 0.);

    intmsg_debug("< " << this->GetName() << " >:ExecuteInteraction() " << ret);
    intmsg_debug("tInitialKineticEnergy: " << tInitialKineticEnergy << ret);
    intmsg_debug("tInitialDirection X: " << tInitialDirection.X() << " Y: " << tInitialDirection.Y()
                                         << " Z: " << tInitialDirection.Z() << ret)

        // outgoing primary
        double tTheta = 0;
    double tLostKineticEnergy = GetEnergyLoss(tInitialKineticEnergy, tTheta);
    intmsg_debug("tLostKineticEnergy: " << tLostKineticEnergy << ret);
    tTheta = GetTheta(tInitialKineticEnergy, tLostKineticEnergy);
    intmsg_debug("tTheta: " << tTheta << ret);
    double tPhi = KRandom::GetInstance().Uniform(0., 2. * KConst::Pi());
    intmsg_debug("tPhi: " << tPhi << ret << ret);

    KThreeVector tOrthogonalOne = tInitialDirection.Orthogonal();
    KThreeVector tOrthogonalTwo = tInitialDirection.Cross(tOrthogonalOne);
    KThreeVector tFinalDirection =
        tInitialDirection.Magnitude() *
        (sin(tTheta) * (cos(tPhi) * tOrthogonalOne.Unit() + sin(tPhi) * tOrthogonalTwo.Unit()) +
         cos(tTheta) * tInitialDirection.Unit());

    aFinalParticle = anInitialParticle;
    aFinalParticle.SetMomentum(tFinalDirection);
    aFinalParticle.SetKineticEnergy_eV(tInitialKineticEnergy - tLostKineticEnergy);

    intmsg_debug("FinalParticle: " << ret);
    intmsg_debug("Time: " << aFinalParticle.GetTime() << ret);
    intmsg_debug("Position: " << aFinalParticle.GetPosition().X() << " " << aFinalParticle.GetPosition().Y() << " "
                              << aFinalParticle.GetPosition().Z() << ret);
    intmsg_debug("Momentum: " << aFinalParticle.GetMomentum().X() << " " << aFinalParticle.GetMomentum().Y() << " "
                              << aFinalParticle.GetMomentum().Z() << ret);
    intmsg_debug("KineticEnergy: " << aFinalParticle.GetKineticEnergy_eV() << ret);

    // Outgoing secondaries
    for (int i = 0; i < 2; ++i) {
        tTheta = acos(KRandom::GetInstance().Uniform(-1., 1.));
        tPhi = KRandom::GetInstance().Uniform(0., 2. * KConst::Pi());

        tOrthogonalOne = tInitialDirection.Orthogonal();
        tOrthogonalTwo = tInitialDirection.Cross(tOrthogonalOne);
        tFinalDirection = tInitialDirection.Magnitude() *
                          (sin(tTheta) * (cos(tPhi) * tOrthogonalOne.Unit() + sin(tPhi) * tOrthogonalTwo.Unit()) +
                           cos(tTheta) * tInitialDirection.Unit());

        KSParticle* tSecondary = KSParticleFactory::GetInstance().Create(11);
        (*tSecondary) = anInitialParticle;
        tSecondary->SetMomentum(tFinalDirection);
        tSecondary->SetKineticEnergy_eV(tLostKineticEnergy - fIonizationEnergy->at(i));
        tSecondary->SetLabel(GetName());

        aSecondaries.push_back(tSecondary);

        intmsg_debug("SecondaryParticle: " << ret);
        intmsg_debug("Time: " << tSecondary->GetTime() << ret);
        intmsg_debug("Position: " << tSecondary->GetPosition().X() << " " << tSecondary->GetPosition().Y() << " "
                                  << tSecondary->GetPosition().Z() << ret);
        intmsg_debug("Momentum: " << tSecondary->GetMomentum().X() << " " << tSecondary->GetMomentum().Y() << " "
                                  << tSecondary->GetMomentum().Z() << ret);
        intmsg_debug("KineticEnergy: " << tSecondary->GetKineticEnergy_eV() << ret);
    }
    intmsg_debug("=============<" << this->GetName() << ">:ExecuteInteraction()" << eom)

        fStepNInteractions = 1;
    fStepEnergyLoss = tLostKineticEnergy;

    return;
}

double KSIntCalculatorArgonDoubleIonisation::GetEnergyLoss(const double& anEnergy, const double&) const
{
    assert(anEnergy == anEnergy);
    assert(anEnergy >= 0.);

    double tmpSecEnergy;
    double tmpMaxCross = KConst::BohrRadiusSquared() * 13.2 / anEnergy * log((anEnergy + 120.0 / anEnergy) / 15.76) *
                         10.3 * 10.3 / (pow(0. - 2 + 100. / (10. + anEnergy), 2) + 10.3 * 10.3);

    while (true) {
        tmpSecEnergy =
            KRandom::GetInstance().Uniform(0.,
                                           (anEnergy - fIonizationEnergy->at(0) - fIonizationEnergy->at(1)) / 2.,
                                           false,
                                           true);

        double tmpDiffCross = KConst::BohrRadiusSquared() * 13.2 / anEnergy *
                              log((anEnergy + 120.0 / anEnergy) / 15.76) * 10.3 * 10.3 /
                              (pow(tmpSecEnergy - 2. + 100. / (10. + anEnergy), 2) + 10.3 * 10.3);

        if (KRandom::GetInstance().Uniform(0., tmpMaxCross, false, true) < tmpDiffCross)
            break;
    }
    assert(tmpSecEnergy == tmpSecEnergy);
    assert(tmpSecEnergy >= 0.);

    return fIonizationEnergy->at(0) + fIonizationEnergy->at(1) + tmpSecEnergy;
}

double KSIntCalculatorArgonDoubleIonisation::GetTheta(const double& anEnergy, const double& anEloss) const
{
    assert(anEnergy == anEnergy);
    assert(anEnergy >= 0.);
    assert(anEloss == anEloss);
    assert(anEloss >= 0.);

    double sigma_max = GetDifferentialCrossSectionAt(anEnergy, 0., anEloss);
    sigma_max *= 1.5;

    double sigma_min = GetDifferentialCrossSectionAt(anEnergy, 180., anEloss);
    sigma_min *= 0.9;

    double tTheta_deg;

    while (true) {
        tTheta_deg = KRandom::GetInstance().Uniform(0., 180., false, true);

        if (KRandom::GetInstance().Uniform(sigma_min, sigma_max, false, true) <
            GetDifferentialCrossSectionAt(anEnergy, tTheta_deg, anEloss))
            break;
    }
    assert(tTheta_deg == tTheta_deg);
    assert(tTheta_deg >= 0. && tTheta_deg <= 180.);

    return tTheta_deg * KConst::Pi() / 180.;
}

double KSIntCalculatorArgonDoubleIonisation::GetDifferentialCrossSectionAt(const double& anEnergy,
                                                                           const double& anAngle,
                                                                           const double& anEloss) const
{
    assert(anEnergy == anEnergy);
    assert(anEnergy >= 0.);
    assert(anEloss == anEloss);
    assert(anEloss >= 0.);
    assert(anAngle == anAngle);
    assert(anAngle >= 0.);

    double aCrossection;
    double wrongtotalcross;
    double aReducedInitialEnergy = anEnergy / (fIonizationEnergy->at(1));
    double aReducedFinalEnergy = (anEnergy - anEloss) / fIonizationEnergy->at(1);

    DiffCrossCalculator->CalculateDoublyDifferentialCrossSection(aReducedInitialEnergy,
                                                                 aReducedFinalEnergy,
                                                                 cos(anAngle * KConst::Pi() / 180.),
                                                                 aCrossection);

    DiffCrossCalculator->CalculateCrossSection(anEnergy, wrongtotalcross);

    return aCrossection / wrongtotalcross * this->GetTotalCrossSectionAt(anEnergy);
}

/////////////////////////////////
/////		Data Reader 	/////
/////////////////////////////////

KSIntCalculatorArgonTotalCrossSectionReader::KSIntCalculatorArgonTotalCrossSectionReader(std::istream* aStream,
                                                                                         unsigned int numOfParameters)
{
    fStream = aStream;
    fNumOfParameters = numOfParameters;
    fData = new std::map<double, double>();
    fParameters = new std::vector<double>();
}

KSIntCalculatorArgonTotalCrossSectionReader::~KSIntCalculatorArgonTotalCrossSectionReader()
{
    delete fData;
    delete fParameters;
}

std::vector<double>* KSIntCalculatorArgonTotalCrossSectionReader::GetParameters()
{
    return fParameters;
}

std::map<double, double>* KSIntCalculatorArgonTotalCrossSectionReader::GetData()
{
    return fData;
}

bool KSIntCalculatorArgonTotalCrossSectionReader::Read()
{
    intmsg_debug("reading total cross data" << eom);
    if (!fStream->good()) {
        return false;
    }

    enum Mode
    {
        READ_PARAMETER,
        READ_DATA
    };

    Mode mode = fNumOfParameters > 0 ? READ_PARAMETER : READ_DATA;

    fData->clear();

    int pos;
    double value, secondValue = 0;
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

    intmsg_debug("...success" << eom);
    return true;
}

KSIntCalculatorArgonDifferentialCrossSectionReader::KSIntCalculatorArgonDifferentialCrossSectionReader(
    std::istream* aStream, unsigned int numOfParameters)
{
    fStream = aStream;
    fNumOfParameters = numOfParameters;
    fData = new std::map<double*, double>();
    fParameters = new std::vector<double>();
}

KSIntCalculatorArgonDifferentialCrossSectionReader::~KSIntCalculatorArgonDifferentialCrossSectionReader()
{
    delete fData;
    delete fParameters;
}

std::vector<double>* KSIntCalculatorArgonDifferentialCrossSectionReader::GetParameters()
{
    return fParameters;
}

std::map<double*, double>* KSIntCalculatorArgonDifferentialCrossSectionReader::GetData()
{
    return fData;
}

bool KSIntCalculatorArgonDifferentialCrossSectionReader::Read()
{
    intmsg_debug("reading diff cross data" << eom);
    if (!fStream->good()) {
        return false;
    }

    enum Mode
    {
        READ_PARAMETER,
        READ_DATA
    };

    Mode mode = fNumOfParameters > 0 ? READ_PARAMETER : READ_DATA;

    fData->clear();

    int pos;
    double value;
    char c;
    unsigned int dimX = 0, dimY = 0;
    unsigned int currX = 0, currY = 0;

    double* point;

    std::vector<double> energies;
    std::vector<double> angles;

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

                    // Read data: The first two numbers gives the dimension of the matrix.
                    // #1(dimX) x #2(dimY)
                case READ_DATA:
                    if (0 == dimX) {
                        dimX = static_cast<unsigned int>(value);
                    }
                    else if (0 == dimY) {
                        dimY = static_cast<unsigned int>(value);
                    }
                    else if (energies.size() != dimX - 1) {
                        // The first line (except the first element) gives you the energies...
                        if (currX > 0) {
                            energies.push_back(value);
                        }

                        if (++currX >= dimX) {
                            currX = 0;
                            ++currY;
                        }
                    }
                    else {
                        if (0 == currX) {
                            // First, store the angle...
                            angles.push_back(value);
                        }
                        else if (value > 0) {
                            // ...and second the cross section:
                            point = new double[2];
                            point[0] = energies[currX - 1];
                            point[1] = angles[currY - 1];

                            fData->insert(std::pair<double*, double>(point, value));
                        }

                        if (++currX >= dimX) {
                            currX = 0;
                            ++currY;
                        }
                    }
                    break;
            }
        }
    }

    if (currY != dimY) {
        return false;
    }

    intmsg_debug("...success" << eom);
    return true;
}

bool KSIntCalculatorArgonDifferentialCrossSectionReader::Readlx()
{
    intmsg_debug("reading diff cross data" << eom);
    if (!fStream->good()) {
        return false;
    }

    enum Mode
    {
        READ_PARAMETER,
        READ_DATA
    };

    Mode mode = fNumOfParameters > 0 ? READ_PARAMETER : READ_DATA;

    fData->clear();

    double angle;
    double parameter;
    double energy;
    double diffcross;
    char c;

    double* point;

    while ((*fStream) >> c) {
        // Skip comments
        if ('#' == c) {
            fStream->ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        }
        else {
            //rewind input stream to previous position since c is not #
            int pos = fStream->tellg();
            fStream->seekg(pos - 1);

            switch (mode) {
                // The first fNumOfParameters numbers in file are parameters
                // for extrapolation or something. After that it will read data
                // points for interpolation.
                case READ_PARAMETER:
                    // Read number
                    (*fStream) >> parameter;
                    fParameters->push_back(parameter);

                    if (fParameters->size() >= fNumOfParameters) {
                        mode = READ_DATA;
                    }
                    break;

                    // Within the LXCat Database the differential cross sections
                    // are delivered as
                    // Angle (deg) | Energy (eV) | Differential cross section (m2)
                case READ_DATA:

                    (*fStream) >> angle;
                    (*fStream) >> energy;
                    (*fStream) >> diffcross;
                    point = new double[2];
                    point[0] = energy;
                    point[1] = angle;
                    fData->insert(std::pair<double*, double>(point, diffcross));

                    break;
            }
        }
    }
    intmsg_debug("...success" << eom);
    return true;
}

} /* namespace Kassiopeia */
