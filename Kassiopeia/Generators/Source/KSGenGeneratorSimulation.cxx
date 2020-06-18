#include "KSGenGeneratorSimulation.h"

#include "KSGeneratorsMessage.h"
#include "KSParticleFactory.h"

namespace Kassiopeia
{

KSGenGeneratorSimulation::KSGenGeneratorSimulation() :
    fBase(""),
    fPath(""),
    fPositionX(""),
    fPositionY(""),
    fPositionZ(""),
    fDirectionX(""),
    fDirectionY(""),
    fDirectionZ(""),
    fEnergy(""),
    fTime(""),
    fTerminator(""),
    fGenerator(""),
    fTrackGroupName("output_track_world"),
    fTerminatorName("terminator_name"),
    fGeneratorName("creator_name"),
    fPositionName("final_position"),
    fMomentumName("final_momentum"),
    fKineticEnergyName("final_kinetic_energy"),
    fTimeName("final_time"),
    fPIDName(""),
    fDefaultPosition(KThreeVector(0, 0, 0)),
    fDefaultDirection(KThreeVector(0, 0, 0)),
    fDefaultEnergy(1.),
    fDefaultTime(0.),
    fDefaultPID(11),  // electron
    fRootFile(nullptr),
    fFormulaPositionX(nullptr),
    fFormulaPositionY(nullptr),
    fFormulaPositionZ(nullptr),
    fFormulaDirectionX(nullptr),
    fFormulaDirectionY(nullptr),
    fFormulaDirectionZ(nullptr),
    fFormulaEnergy(nullptr),
    fFormulaTime(nullptr)
{}
KSGenGeneratorSimulation::KSGenGeneratorSimulation(const KSGenGeneratorSimulation& aCopy) :
    KSComponent(),
    fBase(aCopy.fBase),
    fPath(aCopy.fPath),
    fPositionX(aCopy.fPositionX),
    fPositionY(aCopy.fPositionY),
    fPositionZ(aCopy.fPositionZ),
    fDirectionX(aCopy.fDirectionX),
    fDirectionY(aCopy.fDirectionY),
    fDirectionZ(aCopy.fDirectionX),
    fEnergy(aCopy.fEnergy),
    fTime(aCopy.fTime),
    fTerminator(aCopy.fTerminator),
    fGenerator(aCopy.fGenerator),
    fTrackGroupName(aCopy.fTrackGroupName),
    fTerminatorName(aCopy.fTerminatorName),
    fGeneratorName(aCopy.fGeneratorName),
    fPositionName(aCopy.fPositionName),
    fMomentumName(aCopy.fMomentumName),
    fKineticEnergyName(aCopy.fKineticEnergyName),
    fTimeName(aCopy.fTimeName),
    fPIDName(aCopy.fPIDName),
    fRootFile(nullptr),
    fFormulaPositionX(nullptr),
    fFormulaPositionY(nullptr),
    fFormulaPositionZ(nullptr),
    fFormulaDirectionX(nullptr),
    fFormulaDirectionY(nullptr),
    fFormulaDirectionZ(nullptr),
    fFormulaEnergy(nullptr),
    fFormulaTime(nullptr)
{}
KSGenGeneratorSimulation* KSGenGeneratorSimulation::Clone() const
{
    return new KSGenGeneratorSimulation(*this);
}
KSGenGeneratorSimulation::~KSGenGeneratorSimulation() {}

void KSGenGeneratorSimulation::ExecuteGeneration(KSParticleQueue& aPrimaries)
{
    KSParticleQueue tParticleQueue;
    GenerateParticlesFromFile(tParticleQueue);

    aPrimaries.assign(tParticleQueue.begin(), tParticleQueue.end());

    // check if particle state is valid
    KSParticleIt tParticleIt;
    for (tParticleIt = aPrimaries.begin(); tParticleIt != aPrimaries.end(); tParticleIt++) {
        auto* tParticle = new KSParticle(**tParticleIt);
        if (!tParticle->IsValid()) {
            tParticle->Print();
            delete tParticle;
            genmsg(eError) << "invalid particle state in generator <" << GetName() << ">" << eom;
        }
        delete tParticle;
    }

    return;
}

void KSGenGeneratorSimulation::InitializeComponent()
{
    // prepare ROOT file
    fRootFile = KRootFile::CreateOutputRootFile(fBase);
    if (!fPath.empty()) {
        fRootFile->AddToPaths(fPath);
    }
    if (fRootFile->Open(KFile::eRead) == false) {
        genmsg(eError) << "simulation generator <" << GetName() << "> could not open file <" << fBase << "> at path <"
                       << fPath << ">" << eom;
    }

    // run consictency checks
    if (fTrackGroupName.empty())
        genmsg(eError) << "track group must be defined to access tracks from simulation output" << eom;

    if (!fTerminator.empty()) {
        if (fTerminatorName.empty())
            genmsg(eError) << "terminator field must be define to select tracks by terminator value" << eom;
        genmsg(eDebug) << "selecting tracks with terminator set to <" << fTerminator << ">" << eom;
    }
    if (!fGenerator.empty()) {
        if (fGeneratorName.empty())
            genmsg(eError) << "generator field must be define to select tracks by generator value" << eom;
        genmsg(eDebug) << "selecting tracks with generator set to <" << fGenerator << ">" << eom;
    }

    if (fPositionName.empty())
        genmsg(eWarning) << "final position will not be read from simulation output" << eom;
    if (fMomentumName.empty())
        genmsg(eWarning) << "final momentum will not be read from simulation output" << eom;
    if (fKineticEnergyName.empty())
        genmsg(eWarning) << "final kinetic energy will not be read from simulation output" << eom;
    if (fTimeName.empty())
        genmsg(eWarning) << "final time will not be read from simulation output" << eom;
    if (fPIDName.empty())
        genmsg(eWarning) << "particle id will not be read from simulation output" << eom;

    // create formula objects, if defined
    if (!fPositionX.empty())
        fFormulaPositionX = new TFormula("position_x", fPositionX.c_str());
    if (!fPositionY.empty())
        fFormulaPositionY = new TFormula("position_y", fPositionY.c_str());
    if (!fPositionZ.empty())
        fFormulaPositionZ = new TFormula("position_z", fPositionZ.c_str());
    if (!fDirectionX.empty())
        fFormulaDirectionX = new TFormula("direction_x", fDirectionX.c_str());
    if (!fDirectionY.empty())
        fFormulaDirectionY = new TFormula("direction_y", fDirectionY.c_str());
    if (!fDirectionZ.empty())
        fFormulaDirectionZ = new TFormula("direction_z", fDirectionZ.c_str());
    if (!fEnergy.empty())
        fFormulaEnergy = new TFormula("energy", fEnergy.c_str());
    if (!fTime.empty())
        fFormulaTime = new TFormula("time", fTime.c_str());

    return;
}
void KSGenGeneratorSimulation::DeinitializeComponent()
{
    if (fRootFile != nullptr)
        delete fRootFile;
    fRootFile = nullptr;

    if (fFormulaPositionX != nullptr)
        delete fFormulaPositionX;
    if (fFormulaPositionY != nullptr)
        delete fFormulaPositionY;
    if (fFormulaPositionZ != nullptr)
        delete fFormulaPositionZ;

    if (fFormulaDirectionX != nullptr)
        delete fFormulaDirectionX;
    if (fFormulaDirectionY != nullptr)
        delete fFormulaDirectionY;
    if (fFormulaDirectionZ != nullptr)
        delete fFormulaDirectionZ;

    if (fFormulaEnergy != nullptr)
        delete fFormulaEnergy;
    if (fFormulaTime != nullptr)
        delete fFormulaTime;

    fFormulaPositionX = nullptr;
    fFormulaPositionY = nullptr;
    fFormulaPositionZ = nullptr;
    fFormulaDirectionX = nullptr;
    fFormulaDirectionY = nullptr;
    fFormulaDirectionZ = nullptr;
    fFormulaEnergy = nullptr;
    fFormulaTime = nullptr;

    return;
}

void KSGenGeneratorSimulation::GenerateParticlesFromFile(KSParticleQueue& aParticleQueue)
{
    KSReadFileROOT tReader;
    tReader.OpenFile(fRootFile);

    KSReadRunROOT& tRunReader = tReader.GetRun();
    KSReadEventROOT& tEventReader = tReader.GetEvent();
    KSReadTrackROOT& tTrackReader = tReader.GetTrack();

    KSReadObjectROOT& tTrackGroup = tTrackReader.GetObject(fTrackGroupName);

    for (tRunReader = 0; tRunReader <= tRunReader.GetLastRunIndex(); tRunReader++) {
        for (tEventReader = tRunReader.GetFirstEventIndex(); tEventReader <= tRunReader.GetLastEventIndex();
             tEventReader++) {
            for (tTrackReader = tEventReader.GetFirstTrackIndex(); tTrackReader <= tEventReader.GetLastTrackIndex();
                 tTrackReader++) {
                if (!tTrackGroup.Valid())
                    continue;

                if (!fTerminator.empty()) {
                    string tTerminator = tTrackGroup.Get<KSString>(fTerminatorName).Value();
                    if (tTerminator != fTerminator)
                        continue;
                }

                if (!fGenerator.empty()) {
                    string tGenerator = tTrackGroup.Get<KSString>(fGeneratorName).Value();
                    if (tGenerator != fGenerator)
                        continue;
                }

                KThreeVector tPosition = fDefaultPosition;
                KThreeVector tDirection = fDefaultDirection;
                double tEnergy = fDefaultEnergy;
                double tTime = fDefaultTime;
                int tPID = fDefaultPID;

                if (!fPositionName.empty()) {
                    tPosition = tTrackGroup.Get<KSThreeVector>(fPositionName).Value();
                }
                if (!fMomentumName.empty()) {
                    tDirection = tTrackGroup.Get<KSThreeVector>(fMomentumName).Value().Unit();
                }
                if (!fKineticEnergyName.empty()) {
                    tEnergy = tTrackGroup.Get<KSDouble>(fKineticEnergyName).Value();
                }
                if (!fTimeName.empty()) {
                    tTime = tTrackGroup.Get<KSDouble>(fTimeName).Value();
                }
                if (!fPIDName.empty()) {
                    tPID = tTrackGroup.Get<KSLongLong>(fPIDName).Value();
                }

                if (fFormulaPositionX != nullptr)
                    tPosition.SetX(fFormulaPositionX->Eval(tPosition.X()));
                if (fFormulaPositionY != nullptr)
                    tPosition.SetY(fFormulaPositionY->Eval(tPosition.Y()));
                if (fFormulaPositionZ != nullptr)
                    tPosition.SetZ(fFormulaPositionZ->Eval(tPosition.Z()));
                if (fFormulaDirectionX != nullptr)
                    tDirection.SetX(fFormulaDirectionX->Eval(tDirection.X()));
                if (fFormulaDirectionY != nullptr)
                    tDirection.SetY(fFormulaDirectionY->Eval(tDirection.Y()));
                if (fFormulaDirectionZ != nullptr)
                    tDirection.SetZ(fFormulaDirectionZ->Eval(tDirection.Z()));
                if (fFormulaEnergy != nullptr)
                    tEnergy = fFormulaEnergy->Eval(tEnergy);
                if (fFormulaTime != nullptr)
                    tTime = fFormulaTime->Eval(tTime);

                KSParticle* tParticle = KSParticleFactory::GetInstance().Create(tPID);
                tParticle->SetPosition(tPosition);
                tParticle->SetMomentum(tDirection.Unit());  // normalize again here to be safe
                tParticle->SetKineticEnergy_eV(tEnergy);
                tParticle->SetTime(tTime);
                tParticle->AddLabel(GetName());
                tParticle->AddLabel(
                    std::to_string(tRunReader.GetRunIndex()));  // append run/event/track no. to creator name
                tParticle->AddLabel(std::to_string(tEventReader.GetEventIndex()));
                tParticle->AddLabel(std::to_string(tTrackReader.GetTrackIndex()));

                aParticleQueue.push_back(tParticle);
            }
        }
    }

    genmsg_debug("simulation generator <" << GetName() << "> creates " << aParticleQueue.size() << " particles" << eom);

    tReader.CloseFile();
    return;
}

}  // namespace Kassiopeia
