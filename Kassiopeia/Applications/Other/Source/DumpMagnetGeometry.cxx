#include "KMessage.h"
#include "KSFieldFinder.h"
#include "KSMainMessage.h"
#include "KSMagneticKEMField.h"
#include "KGStaticElectromagnetField.hh"
#include "KZonalHarmonicMagnetostaticFieldSolver.hh"
#include "KThreeVector.hh"
#include "KXMLInitializer.hh"
#include "KXMLTokenizer.hh"


using namespace Kassiopeia;
using namespace KEMField;
using namespace katrin;
using namespace std;

void dumpMagnetCoils(std::ostream& strm, std::shared_ptr<KElectromagnetContainer> container)
{
    for (size_t i = 0; i < container->size(); ++i) {
        const auto* tCoil = dynamic_cast<const KCoil*>(container->at(i));
        if (tCoil) {
            auto p0 = tCoil->GetCoordinateSystem().ToGlobal(KEMField::KPosition(0, 0, tCoil->GetZ0()));
            auto p1 = tCoil->GetCoordinateSystem().ToGlobal(KEMField::KPosition(0, 0, tCoil->GetZ1()));
            strm << ' ' << tCoil->GetCurrentDensity()
                 << '\t' << p0.X() << '\t' << p0.Y() << '\t' << p0.Z()
                 << '\t' << p1.X() << '\t' << p1.Y() << '\t' << p1.Z()
                 << '\t' << tCoil->GetR0() << '\t' << tCoil->GetR1()
                 << '\t' << tCoil->GetIntegrationScale()
                 << '\t' << tCoil->GetCurrent()
                 << '\t' << tCoil->GetNumberOfTurns()
                 << std::endl;
        }
    }
    strm << std::endl;
}

template<class BasisType>
void dumpCentralSourcePoints(std::ostream& strm, const KZonalHarmonicContainer<BasisType>* container)
{
    const int n_coeffs = container->GetParameters().GetNCentralCoefficients();

    for (auto& sp : container->GetCentralSourcePoints()) {
        auto z0 = sp->GetZ0() + container->GetCoordinateSystem().GetOrigin().Z();
        auto rho = sp->GetRho();
        strm << z0 << '\t' << rho << '\t' << sp->GetNCoeffs();
        for (int i = 0; i < n_coeffs; ++i)
            strm << '\t' << sp->GetCoeff(i);
        strm << std::endl;
    }
    strm << std::endl;

    for (auto& subcontainer : container->GetSubContainers()) {
        dumpCentralSourcePoints(strm, subcontainer);
    }
}

template<class BasisType>
void dumpRemoteSourcePoints(std::ostream& strm, const KZonalHarmonicContainer<BasisType>* container)
{
    const int n_coeffs = container->GetParameters().GetNRemoteCoefficients();

    for (auto& sp : container->GetRemoteSourcePoints()) {
        auto z0 = sp->GetZ0() + container->GetCoordinateSystem().GetOrigin().Z();
        auto rho = sp->GetRho();
        strm << z0 << '\t' << rho << '\t' << sp->GetNCoeffs();
        for (int i = 0; i < n_coeffs; ++i)
            strm << '\t' << sp->GetCoeff(i);
        strm << std::endl;
    }
    strm << std::endl;

    for (auto& subcontainer : container->GetSubContainers()) {
        dumpRemoteSourcePoints(strm, subcontainer);
    }
}

int main(int argc, char** argv)
{
    if (argc < 4) {
        std::cout
            << "usage: ./DumpMagnetGeometry <config_file.xml> <output_file.txt> <magnetic_field_name1> [<magnetic_field_name2> <...>] "
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
        outFileBuf = std::cout.rdbuf();
    }
    else {
        outFileStream.open(outFileName);
        outFileBuf = outFileStream.rdbuf();
    }
    ostream outFile(outFileBuf);

    outFile << std::setprecision(10);

    mainmsg(eNormal) << "...initialization finished" << eom;

    for (size_t tIndex = 1; tIndex < tParameters.size(); tIndex++) {
        auto & tMagFieldName = tParameters[tIndex];

        KSMagneticField* tMagneticFieldObject = getMagneticField(tMagFieldName);
        if (! tMagneticFieldObject) {
            mainmsg(eError) << "Magnetic field <" << tMagFieldName << "> does not exist" << eom;
            continue;
        }

        outFile << "# === MAGNETIC FIELD NAME: " << tMagFieldName << " ===" << std::endl;

        tMagneticFieldObject->Initialize();

        auto* tKEMFieldObject = dynamic_cast<KSMagneticKEMField*>(tMagneticFieldObject);
        if (! tKEMFieldObject) {
            mainmsg(eWarning) << "Magnetic field <" << tMagFieldName << "> is not a KEMField object" << eom;
            continue;
        }

        auto* tMagField = dynamic_cast<KGStaticElectromagnetField*>(tKEMFieldObject->GetMagneticField());
        if (! tMagField) {
            mainmsg(eWarning) << "Magnetic field <" << tMagFieldName << "> is not a static electromagnet field" << eom;
            continue;
        }

        tMagField->Initialize();

        outFile << std::endl;
        outFile << "# --- MAGNET GEOMETRY (SOLENOIDS) ---" << std::endl;
        outFile << "# " << "cur_dens"
             << '\t' << "p0.x" << '\t' << "p0.y" << '\t' << "p0.z"
             << '\t' << "p1.x" << '\t' << "p1.y" << '\t' << "p1.z"
             << '\t' << "r0" << '\t' << "r1"
             << '\t' << "num_disc" << '\t' << "current" << '\t' << "num_turns"
             << std::endl;

        dumpMagnetCoils(outFile, tMagField->GetContainer());

        auto* tMagZHSolver = dynamic_cast<KZonalHarmonicMagnetostaticFieldSolver*>(&(*tMagField->GetFieldSolver()));
        if (! tMagZHSolver) {
            mainmsg(eWarning) << "Magnetic field <" << tMagFieldName << "> does not have a zonal harmonic solver" << eom;
            continue;
        }

        outFile << std::endl;
        outFile << "# --- CENTRAL SOURCE POINTS ---" << std::endl;
        outFile << "# " << "z0" << '\t' << "r0" << '\t' << "num_coeff" << '\t' << "c(0) ... c(n-1)" << std::endl;

        dumpCentralSourcePoints(outFile, tMagZHSolver->GetContainer());

        outFile << std::endl;
        outFile << "# --- REMOTE SOURCE POINTS ---" << std::endl;
        outFile << "# " << "z0" << '\t' << "r0" << '\t' << "num_coeff" << '\t' << "c(0) ... c(n-1)" << std::endl;

        dumpRemoteSourcePoints(outFile, tMagZHSolver->GetContainer());

        outFile << std::endl;
    }

    for (size_t tIndex = 3; tIndex < tParameters.size(); tIndex++) {
        KSMagneticField* tMagneticFieldObject = getMagneticField(tParameters[tIndex]);
        tMagneticFieldObject->Deinitialize();
    }

    if (outFileStream.is_open())
        outFileStream.close();

    return 0;
}
