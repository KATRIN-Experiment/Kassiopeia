#include "KSGenGeneratorTextFile.h"

#include "KSGeneratorsMessage.h"
#include "KSParticleFactory.h"
#include "KBaseStringUtils.h"

using namespace std;
using namespace katrin;
using KGeoBag::KThreeVector;
using katrin::KTextFile;

namespace Kassiopeia
{

KSGenGeneratorTextFile::KSGenGeneratorTextFile() :
    fBase(""),
    fPath("./")
{}
KSGenGeneratorTextFile::KSGenGeneratorTextFile(const KSGenGeneratorTextFile& aCopy) :
    KSComponent(aCopy),
    fBase(aCopy.fBase),
    fPath(aCopy.fPath)
{}
KSGenGeneratorTextFile* KSGenGeneratorTextFile::Clone() const
{
    return new KSGenGeneratorTextFile(*this);
}
KSGenGeneratorTextFile::~KSGenGeneratorTextFile() = default;

void KSGenGeneratorTextFile::ExecuteGeneration(KSParticleQueue& aPrimaries)
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

void KSGenGeneratorTextFile::InitializeComponent()
{
    fTextFile = katrin::KTextFile::CreateOutputTextFile(fPath, fBase);
    fTextFile->AddToPaths(DATA_DEFAULT_DIR);
    fTextFile->AddToPaths(SCRATCH_DEFAULT_DIR);

    if (! fTextFile->Open()) {
        genmsg(eError) << "file generator <" << GetName() << " cannot open input file <" << fPath << ">" << eom;
    }

    return;
}
void KSGenGeneratorTextFile::DeinitializeComponent()
{
    return;
}

void KSGenGeneratorTextFile::GenerateParticlesFromFile(KSParticleQueue& aParticleQueue)
{
    auto ifs = fTextFile->File();
    if (ifs && ifs->is_open()) {
        string buf;
        while (getline(*ifs, buf)) {
            if (buf.empty() || buf[0] == '#')
                continue;

            vector<double> fields;
            fields = KBaseStringUtils::SplitTrimAndConvert<double>(buf, " \t");

            if (fields.size() < 10) {
                genmsg(eError) << "file generator <" << GetName() << " cannot parse input file with " << fields.size() << " columns (needs at least 10)" << eom;
            }

            // Format:
            //  # index_number	     pid	mass	charge	total_spin	gyromagnetic_ratio	time	position_x	position_y	position_z	momentum_x	momentum_y	momentum_z	kinetic_energy_ev	magnetic_field	electric_field	electric_potential	label

            size_t k = 0;  // used to iterate over colums

            uint32_t tIndex = fields[k+0];
            int32_t tPID = fields[k+1];
            k += 2;

            KSParticle* tParticle = KSParticleFactory::GetInstance().Create(tPID);
            tParticle->SetIndexNumber(tIndex);

            if (fields.size() >= 14) {
                // long fromat - contains mass, charge etc.
                // if ((tParticle->GetMass() != fields[2]) || (tParticle->GetCharge() != fields[3]) ||
                //     (tParticle->GetSpinMagnitude() != fields[4]) || (tParticle->GetGyromagneticRatio() == fields[5])) {
                //     genmsg(eWarning) << "The particle mass/charge/etc. does not match PID <" << tPID << "> for index <" << tIndex << ">" << eom;
                // }

                k += 4;  // skip four columns
            }

            tParticle->SetTime(fields[k+0]);
            tParticle->SetPosition(KThreeVector(fields[k+1], fields[k+2], fields[k+3]));
            tParticle->SetMomentum(KThreeVector(fields[k+4], fields[k+5], fields[k+6]).Unit());
            tParticle->SetKineticEnergy_eV(fields[k+7]);
            k += 7;

            if (fields.size() > k+1) {
                genmsg(eInfo) << "Ignored " << fields.size()-k << " extra columns for index <" << tIndex << ">" << eom;
            }

            tParticle->AddLabel(GetName());
            tParticle->AddLabel(std::to_string(tIndex));

            aParticleQueue.push_back(tParticle);
        }
    }

    genmsg_debug("file generator <" << GetName() << "> creates " << aParticleQueue.size() << " particles" << eom);

    return;
}

}  // namespace Kassiopeia
