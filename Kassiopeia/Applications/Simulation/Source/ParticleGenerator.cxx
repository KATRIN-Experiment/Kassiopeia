#include "KMessage.h"
#include "KSMainMessage.h"
#include "KSParticleFactory.h"
#include "KSRootElectricField.h"
#include "KSRootGenerator.h"
#include "KSRootMagneticField.h"
#include "KTextFile.h"
#include "KThreeVector.hh"
#include "KToolbox.h"
#include "KXMLInitializer.hh"
#include "KXMLTokenizer.hh"

#ifdef KEMFIELD_USE_MPI
#include "KMPIInterface.hh"
#endif


using namespace Kassiopeia;
using namespace katrin;
using namespace KGeoBag;
using namespace std;

int main(int argc, char** argv)
{
    if (argc < 5) {
        cout
            << "usage: ./ParticleGenerator <config_file.xml> <output_file.txt> <number_of_events> <generator_name1> [<generator_name2> <...>] "
            << endl;
        exit(-1);
    }

    KMessageTable::GetInstance().SetPrecision(16);

    mainmsg(eNormal) << "starting initialization..." << eom;

    auto& tXML = KXMLInitializer::GetInstance();
    tXML.AddDefaultIncludePath(CONFIG_DEFAULT_DIR);
    tXML.Configure(argc, argv, true);

    deque<string> tParameters = tXML.GetArguments().ParameterList();
    tParameters.pop_front();  // strip off config file name

    string outFileName(tParameters[0]);
    ofstream outFileStream;
    streambuf* outFileBuf;
    if (outFileName == "-") {
        outFileBuf = cout.rdbuf();
    }
    else {
        outFileStream.open(outFileName);
        outFileBuf = outFileStream.rdbuf();
    }
    ostream outFile(outFileBuf);

    unsigned long nEvents = stoi(tParameters[1]);

    mainmsg(eNormal) << "...initialization finished" << eom;

    // initialize generator
    KSRootGenerator tRootGenerator;

    KSRootMagneticField tMagneticField;
    KSRootElectricField tElectricField;

    KSParticleFactory::GetInstance().SetMagneticField(&tMagneticField);
    KSParticleFactory::GetInstance().SetElectricField(&tElectricField);

    /// TODO: cmdline option for B/E fields

    outFile << std::left << std::setfill(' ') << std::setw(20) << "# index_number\t"
            << "pid\t"
            << "mass\t"
            << "charge\t"
            << "total_spin\t"
            << "gyromagnetic_ratio\t"
            << "time\t"
            << "position_x\t"
            << "position_y\t"
            << "position_z\t"
            << "momentum_x\t"
            << "momentum_y\t"
            << "momentum_z\t"
            << "kinetic_energy_ev\t"
            << "magnetic_field\t"
            << "electric_field\t"
            << "electric_potential\t"
            << "label\t"
            << endl;

    for (size_t tIndex = 2; tIndex < tParameters.size(); tIndex++) {
        auto* tGeneratorObject = KToolbox::GetInstance().Get<KSGenerator>(tParameters[tIndex]);

        if (!tGeneratorObject) {
            mainmsg(eError) << "Generator <" << tParameters[tIndex] << "> does not exist in toolbox" << eom;
            continue;
        }

        tGeneratorObject->Initialize();
        tRootGenerator.SetGenerator(tGeneratorObject);

        for (unsigned i = 0; i < nEvents; i++) {

            auto* tEvent = new KSEvent();
            tRootGenerator.SetEvent(tEvent);

            try {
                tRootGenerator.ExecuteGeneration();
            }
            catch (...) {
                cout << endl;
                mainmsg(eWarning) << "error - cannot execute generator <" << tGeneratorObject->GetName() << ">" << eom;
                return 1;
            }

            auto tParticleQueue = tEvent->GetParticleQueue();

            mainmsg(eNormal) << "Generator <" << tGeneratorObject->GetName() << "> created <" << tParticleQueue.size()
                             << "> events for iteration " << i+1 << "/" << nEvents << eom;

            for (auto& tParticle : tParticleQueue) {
                tParticle->Print();

                if (!tParticle->IsValid())
                    continue;

                outFile << std::setw(20) << std::scientific << std::setprecision(9)
                        << tParticle->GetIndexNumber() << "\t"
                        << tParticle->GetPID() << "\t"
                        << tParticle->GetMass() << "\t"
                        << tParticle->GetCharge() << "\t"
                        << tParticle->GetSpinMagnitude() << "\t"
                        << tParticle->GetGyromagneticRatio() << "\t"
                        << tParticle->GetTime() << "\t"
                        << tParticle->GetX() << "\t" << tParticle->GetY() << "\t" << tParticle->GetZ() << "\t"
                        << tParticle->GetPX() << "\t" << tParticle->GetPY() << "\t" << tParticle->GetPZ() << "\t"
                        << tParticle->GetKineticEnergy_eV() << "\t"
                        << tParticle->GetMagneticField().Magnitude() << "\t"
                        << tParticle->GetElectricField().Magnitude() << "\t"
                        << tParticle->GetElectricPotential() << "\t"
                        << "\"" << tParticle->GetLabel() << "\"\t"
                        << endl;
            }

            mainmsg(eNormal) << eom;

            delete tEvent;
            tRootGenerator.SetEvent(nullptr);

        }

        tRootGenerator.ClearGenerator(tGeneratorObject);
        tGeneratorObject->Deinitialize();
    }

    if (outFileStream.is_open())
        outFileStream.close();

    return 0;
}
