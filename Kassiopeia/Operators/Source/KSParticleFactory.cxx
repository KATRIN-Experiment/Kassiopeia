#include "KSParticleFactory.h"

#include "KSOperatorsMessage.h"

#include <limits>

namespace Kassiopeia
{

KSParticleFactory::KSParticleFactory() :
    fParticles(),
    fParticleString(),
    fSpace(nullptr),
    fMagneticField(nullptr),
    fElectricField(nullptr),
    fParticleIndex(-1)
{}
KSParticleFactory::~KSParticleFactory() {}

KSParticle* KSParticleFactory::Create(const long long& aPID)
{
    ParticleCIt tIter = fParticles.find(aPID);
    if (tIter == fParticles.end()) {
        tIter = fParticles.find(0);
        oprmsg(eError) << "could not find particle for pid <" << aPID << ">" << eom;
    }

    auto* tParticle = new KSParticle(*(tIter->second));
    tParticle->SetIndexNumber(++fParticleIndex);
    tParticle->SetCurrentSpace(fSpace);
    tParticle->SetMagneticFieldCalculator(fMagneticField);
    tParticle->SetElectricFieldCalculator(fElectricField);
    tParticle->RecalculateSpinBody();

    return tParticle;
}

KSParticleQueue KSParticleFactory::Create(const long long& aPID, const size_t& aCount)
{
    KSParticleQueue result;
    for (size_t i = 1; i < aCount; i++) {
        result.push_back(Create(aPID));
    }
    return result;
}

KSParticle* KSParticleFactory::StringCreate(const std::string& aStringID)
{
    ParticleStringCIt tIter = fParticleString.find(aStringID);
    if (tIter == fParticleString.end()) {
        tIter = fParticleString.find("ghost");
        oprmsg(eError) << "could not find particle for string id <" << aStringID << ">" << eom;
    }

    auto* tParticle = new KSParticle(*(tIter->second));
    tParticle->SetIndexNumber(++fParticleIndex);
    tParticle->SetCurrentSpace(fSpace);
    tParticle->SetMagneticFieldCalculator(fMagneticField);
    tParticle->SetElectricFieldCalculator(fElectricField);
    tParticle->RecalculateSpinBody();

    return tParticle;
}

KSParticleQueue KSParticleFactory::StringCreate(const std::string& aStringID, const size_t& aCount)
{
    KSParticleQueue result;
    for (size_t i = 1; i < aCount; i++) {
        result.push_back(StringCreate(aStringID));
    }
    return result;
}

int KSParticleFactory::Define(const long long& aPID, const std::string& aStringID,
                              const std::vector<std::string> aAltStringIDs, const double& aMass, const double& aCharge,
                              const double& aSpinMagnitude, const double& aGyromagneticRatio)
{
    auto tIter = fParticles.find(aPID);
    if (tIter != fParticles.end()) {
        oprmsg(eError) << "asked to add definition for pid <" << aPID << "> with one already defined" << eom;
        return -1;
    }
    auto tStringIter = fParticleString.find(aStringID);
    if (tStringIter != fParticleString.end()) {
        oprmsg(eError) << "asked to add definition for string_id <" << aStringID << "> with one already defined" << eom;
        return -1;
    }

    auto* tParticle = new KSParticle();
    tParticle->fPID = aPID;
    tParticle->fStringID = aStringID;
    tParticle->fMass = aMass;
    tParticle->fCharge = aCharge;
    tParticle->fSpinMagnitude = aSpinMagnitude;
    tParticle->fGyromagneticRatio = aGyromagneticRatio;

    fParticles.insert(ParticleEntry(aPID, tParticle));
    fParticleString.insert(ParticleStringEntry(aStringID, tParticle));
    //Add the alternative strings to the particle string list as well
    //See http://www.cplusplus.com/reference/vector/vector/begin/
    for (auto it = aAltStringIDs.begin(); it != aAltStringIDs.end(); ++it) {
        fParticleString.insert(ParticleStringEntry(*it, tParticle));
    }

    return 0;
}

void KSParticleFactory::SetSpace(KSSpace* aSpace)
{
    fSpace = aSpace;
    return;
}
KSSpace* KSParticleFactory::GetSpace()
{
    return fSpace;
}

void KSParticleFactory::SetMagneticField(KSMagneticField* aMagneticField)
{
    fMagneticField = aMagneticField;
    return;
}
KSMagneticField* KSParticleFactory::GetMagneticField()
{
    return fMagneticField;
}

void KSParticleFactory::SetElectricField(KSElectricField* anElectricField)
{
    fElectricField = anElectricField;
    return;
}
KSElectricField* KSParticleFactory::GetElectricField()
{
    return fElectricField;
}


// A "ghost" particle
STATICINT sGhostDefinition = KSParticleFactory::GetInstance().Define(0, "ghost", {}, std::numeric_limits<double>::min(),
                                                                     0., 0., 0.);  // needs to have non-zero mass

//electron
STATICINT sElectronDefinition = KSParticleFactory::GetInstance().Define(
    11, "e-", {"e^-"}, katrin::KConst::M_el_kg(), -1. * katrin::KConst::Q(), 0.5, -1.760859644e+11);

//positron
STATICINT sPositronDefinition = KSParticleFactory::GetInstance().Define(-11, "e+", {"e^+"}, katrin::KConst::M_el_kg(),
                                                                        katrin::KConst::Q(), 0.5, -1.760859644e+11);

//muon
STATICINT sMuMinusDefinition = KSParticleFactory::GetInstance().Define(13, "mu-", {"mu^-"}, katrin::KConst::M_mu_kg(),
                                                                       -1 * katrin::KConst::Q(), 0.5, -2.43318710e+7);

//anti-muon
STATICINT sMuPlusDefinition = KSParticleFactory::GetInstance().Define(-13, "mu+", {"mu^+"}, katrin::KConst::M_mu_kg(),
                                                                      katrin::KConst::Q(), 0.5, -2.43318710e+7);

//proton
STATICINT sProtonDefinition = KSParticleFactory::GetInstance().Define(
    2212, "p+", {"p^+", "H^+", "H+"}, katrin::KConst::M_prot_kg(), katrin::KConst::Q(), 0.5, 2.675221900e+8);

//anti-proton
STATICINT sAntiProtonDefinition = KSParticleFactory::GetInstance().Define(
    -2212, "p-", {"p^-"}, katrin::KConst::M_prot_kg(), -1 * katrin::KConst::Q(), 0.5, 2.675221900e+8);

//neutron
STATICINT sNeutronDefinition =
    KSParticleFactory::GetInstance().Define(2112, "n", {}, katrin::KConst::M_neut_kg(), 0., 0.5, -1.83247172e+8);

//deuterium plus
STATICINT sDeuteriumPlusDefinition = KSParticleFactory::GetInstance().Define(
    99041, "D^+", {"D+"}, katrin::KConst::M_deut_kg(), katrin::KConst::Q(), 0.5, 0);

//deuterium 2 plus
STATICINT sDeuteriumTwoPlusDefinition = KSParticleFactory::GetInstance().Define(
    99042, "D_2^+", {"D2^+", "D2+"}, 2.0 * katrin::KConst::M_deut_kg(), katrin::KConst::Q(), 0.5, 0);

//deuterium 3 plus

STATICINT sDeuteriumThreePlusDefinition = KSParticleFactory::GetInstance().Define(
    99043, "D_3^+", {"D3^+", "D3+"}, 3.0 * katrin::KConst::M_deut_kg(), katrin::KConst::Q(), 0.5, 0);

//deuterium minus
STATICINT sDeuteriumMinusDefinition = KSParticleFactory::GetInstance().Define(
    99051, "D^-", {"D-"}, katrin::KConst::M_DMinus_kg(), -1 * katrin::KConst::Q(), 0.5, 0);

//NOTE: Still need values for tritium

//Tritium triplet state
STATICINT sTTripletDefinition = KSParticleFactory::GetInstance().Define(
    99061, "T", {}, katrin::KConst::M_tPlus_kg() + katrin::KConst::M_el_kg(), 0, 0.5, -1.76e+11);

//T+
STATICINT sTPlusDefinition = KSParticleFactory::GetInstance().Define(99071, "T^+", {"T+"}, katrin::KConst::M_tPlus_kg(),
                                                                     katrin::KConst::Q(), 0.5, 2.853493e+8);

//T2+
STATICINT sT2PlusDefinition = KSParticleFactory::GetInstance().Define(
    99072, "T_2^+", {"T2^+", "T2+"}, katrin::KConst::M_T2Plus_kg(), 1 * katrin::KConst::Q(), 0.5, 0);

//T3+
STATICINT sT3PlusDefinition = KSParticleFactory::GetInstance().Define(
    99073, "T_3^+", {"T3^+", "T3+"}, katrin::KConst::M_T2_kg() + katrin::KConst::M_tPlus_kg(), 1 * katrin::KConst::Q(),
    0.5, 0);

//T5+
STATICINT sT5PlusDefinition = KSParticleFactory::GetInstance().Define(
    99075, "T_5^+", {"T5^+", "T5+"}, 2 * katrin::KConst::M_T2_kg() + katrin::KConst::M_tPlus_kg(),
    1 * katrin::KConst::Q(), 0.5, 0);

//T- (NOTE: mass should be corrected)
STATICINT sTMinusDefinition = KSParticleFactory::GetInstance().Define(
    99081, "T^-", {"T-"}, katrin::KConst::M_tPlus_kg(), -1 * katrin::KConst::Q(), 0.5, 0);

//rydberg states
STATICINT sRydbergDefinition_0 = KSParticleFactory::GetInstance().Define(
    99010, "H^*", {"H*"}, 1.008 * katrin::KConst::AtomicMassUnit_kg(), 0., 0.5, 2.67513e+6);

//H2+ (NOTE: One has to decide whether ortho or para for spin...)
STATICINT sH2PlusDefinition = KSParticleFactory::GetInstance().Define(
    99012, "H_2^+", {"H2^+", "H2+"}, katrin::KConst::M_H2Plus_kg(), 1 * katrin::KConst::Q(), 0.5, 0);

//H3+
STATICINT sH3PlusDefinition = KSParticleFactory::GetInstance().Define(
    99013, "H_3^+", {"H3^+", "H3+"}, katrin::KConst::M_H3Plus_kg(), 1 * katrin::KConst::Q(), 0.5, 0);

//H-
STATICINT sHMinusDefinition = KSParticleFactory::GetInstance().Define(
    99021, "H^-", {"H-"}, katrin::KConst::M_HMinus_kg(), -1 * katrin::KConst::Q(), 0.5, 0);

//^4He+ (NOTE: Using He+ mass; this should be improved)
STATICINT s4HeDefinition = KSParticleFactory::GetInstance().Define(
    99101, "^4He^+", {"4He^+", "^4He+", "4He+"}, katrin::KConst::M_HePlus_kg(), 1 * katrin::KConst::Q(), 0.5, 0);
}  // namespace Kassiopeia
