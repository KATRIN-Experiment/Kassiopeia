/**
 * @file ks-particles.cxx
 *
 * @date 22.07.2020
 * @author Jan Behrens <jan.behrens@kit.edu>
 */

// Kasper Common includes
#include <KException.h>
#include <KLogger.h>
#include <KMessage.h>

// Kassiopeia includes
#include <KSMainMessage.h>
#include <KSParticle.h>
#include <KSParticleFactory.h>

KLOGGER("kassiopeia.examples");

using namespace katrin;
using namespace Kassiopeia;

int main(int argc, char** argv)
{
    // set message verbosity
    KMessageTable::GetInstance().SetTerminalVerbosity(eDebug);
    KMessageTable::GetInstance().SetLogVerbosity(eDebug);

    // create an instance of the particle factory
    KSParticleFactory& tFactory = KSParticleFactory::GetInstance();

    // generate particles and print information
    for (int i = 1; i < argc; i++) {
        KSParticle *tParticle;

        mainmsg(eNormal) << "Creating a particle with id: " << argv[i] << eom;

        try {
            try {
                // try using a numeric particle id
                long tPID = std::stol(argv[i]);
                tParticle = tFactory.Create(tPID);
            }
            catch (std::invalid_argument) {
                // if it fails, try to use a string id
                std::string tStringID(argv[i]);
                tParticle = tFactory.StringCreate(tStringID);
            }
        }
        catch (KException) {
            continue;
        }

        tParticle->Print();

        delete tParticle;
    }

    return 0;
}
