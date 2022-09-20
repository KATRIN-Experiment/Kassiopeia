#include "KSTermZHRadius.h"
#include "KSTerminatorsMessage.h"
#include "KSMagneticKEMField.h"
#include "KSElectricKEMField.h"
#include "KStaticElectromagnetField.hh"
#include "KZonalHarmonicMagnetostaticFieldSolver.hh"
#include "KElectrostaticBoundaryField.hh"
#include "KElectricZHFieldSolver.hh"

namespace Kassiopeia
{

KSTermZHRadius::KSTermZHRadius() :
    fMagneticFields(),
    fElectricFields(),
    fCheckCentralExpansion(true),
    fCheckRemoteExpansion(false)
{}
KSTermZHRadius::KSTermZHRadius(const KSTermZHRadius&) : KSComponent() {}
KSTermZHRadius* KSTermZHRadius::Clone() const
{
    return new KSTermZHRadius(*this);
}
KSTermZHRadius::~KSTermZHRadius() = default;

void KSTermZHRadius::AddMagneticField(KSMagneticField* field)
{
    fMagneticFields.push_back(field);
}

const std::vector<KSMagneticField*> KSTermZHRadius::GetMagneticFields() const
{
    return fMagneticFields;
}

void KSTermZHRadius::AddElectricField(KSElectricField* field)
{
    fElectricFields.push_back(field);
}

const std::vector<KSElectricField*> KSTermZHRadius::GetElectricFields() const
{
    return fElectricFields;
}

void KSTermZHRadius::SetCheckCentralExpansion(bool aFlag)
{
    fCheckCentralExpansion = aFlag;
}
bool KSTermZHRadius::GetCheckCentralExpansion() const
{
    return fCheckCentralExpansion;
}

void KSTermZHRadius::SetCheckRemoteExpansion(bool aFlag)
{
    fCheckRemoteExpansion = aFlag;
}
bool KSTermZHRadius::GetCheckRemoteExpansion() const
{
    return fCheckRemoteExpansion;
}

void KSTermZHRadius::CalculateTermination(const KSParticle& anInitialParticle, bool& aFlag)
{
    auto tPosition = anInitialParticle.GetPosition();

    for (auto & solver : fMagneticSolvers) {
        // terminate if ZH solver cannot use central expansion, i.e. position outside convergence radius
        if (fCheckCentralExpansion && ! solver->UseCentralExpansion(tPosition)) {
            termmsg_debug("magnetic field central expansion not valid at " << tPosition << eom);
            aFlag = true;
            return;
        }
        if (fCheckRemoteExpansion && ! solver->UseRemoteExpansion(tPosition)) {
            termmsg_debug("magnetic field remote expansion not valid at " << tPosition << eom);
            aFlag = true;
            return;
        }
    }

    for (auto & solver : fElectricSolvers) {
        // terminate if ZH solver cannot use central expansion, i.e. position outside convergence radius
        if (fCheckCentralExpansion && ! solver->UseCentralExpansion(tPosition)) {
            termmsg_debug("electric field central expansion not valid at " << tPosition << eom);
            aFlag = true;
            return;
        }
        if (fCheckRemoteExpansion && ! solver->UseRemoteExpansion(tPosition)) {
            termmsg_debug("electric field remote expansion not valid at " << tPosition << eom);
            aFlag = true;
            return;
        }
    }

    aFlag = false;
    return;
}

void KSTermZHRadius::ExecuteTermination(const KSParticle&, KSParticle& aFinalParticle, KSParticleQueue&) const
{
    aFinalParticle.SetActive(false);
    aFinalParticle.SetLabel(GetName());
    return;
}

void KSTermZHRadius::InitializeComponent()
{
    fMagneticSolvers.clear();
    for (auto & field : fMagneticFields) {
        auto tMagneticField = dynamic_cast<KSMagneticKEMField*>(field);
        if (! tMagneticField) {
            termmsg(eError) << "cannot initialize terminator <" << GetName() << ">: magnetic field <" << field->GetName() << "> is not an KEMField object." << eom;
            return;
        }

        auto tElectromagnetField = dynamic_cast<KEMField::KStaticElectromagnetField*>(tMagneticField->GetMagneticField());
        if (! tElectromagnetField) {
            termmsg(eError) << "cannot initialize terminator <" << GetName() << ">: magnetic field <" << field->GetName() << "> is not an electromagnet system." << eom;
            return;
        }

        auto tFieldSolver = tElectromagnetField->GetFieldSolver().get();

        auto tZonalHarmonicSolver = dynamic_cast<KEMField::KZonalHarmonicMagnetostaticFieldSolver*>(tFieldSolver);
        if (! tZonalHarmonicSolver) {
                    termmsg(eError) << "cannot initialize terminator <" << GetName() << ">: magnetic field <" << field->GetName() << "> does not use a zonal harmonic solver." << eom;
            return;
        }

        fMagneticSolvers.push_back(tZonalHarmonicSolver);
    }

    fElectricSolvers.clear();
    for (auto & field : fElectricFields) {
        auto tElectricField = dynamic_cast<KSElectricKEMField*>(field);
        if (! tElectricField) {
            termmsg(eError) << "cannot initialize terminator <" << GetName() << ">: electric field <" << field->GetName() << "> is not an KEMField object." << eom;
            return;
        }

        auto tElectrostaticField = dynamic_cast<KEMField::KElectrostaticBoundaryField*>(tElectricField->GetElectricField());
        if (! tElectrostaticField) {
            termmsg(eError) << "cannot initialize terminator <" << GetName() << ">: electric field <" << field->GetName() << "> is not an electrostatic system." << eom;
            return;
        }

        auto tFieldSolver = tElectrostaticField->GetFieldSolver().get();

        auto tZonalHarmonicSolver = dynamic_cast<KEMField::KElectricZHFieldSolver*>(tFieldSolver);
        if (! tZonalHarmonicSolver) {
                    termmsg(eError) << "cannot initialize terminator <" << GetName() << ">: electric field <" << field->GetName() << "> does not use a zonal harmonic solver." << eom;
            return;
        }

        fElectricSolvers.push_back(tZonalHarmonicSolver);
    }
}

}  // namespace Kassiopeia
