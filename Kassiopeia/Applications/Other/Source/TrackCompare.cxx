#include "KSMainMessage.h"
#include "KSReadFileROOT.h"
#include "TAxis.h"
#include "TCanvas.h"
#include "TFile.h"
#include "TGraph.h"
#include "TMath.h"
#include "TMultiGraph.h"
#include "TObjArray.h"

#include <boost/program_options.hpp>
#include <limits>
namespace po = boost::program_options;

using namespace Kassiopeia;
using namespace std;

static struct sMultigraph_s
{
    TMultiGraph* ZPosition;
    TMultiGraph* Radius;
    TMultiGraph* Steps;

    TMultiGraph* ElectricPotential;
    TMultiGraph* ElectricField;
    TMultiGraph* MagneticField;

    TMultiGraph* Time;
    TMultiGraph* Speed;
    TMultiGraph* Acceleration;

    TMultiGraph* KineticEnergy;
    TMultiGraph* PolarAngle;
    TMultiGraph* AziAngle;

    TMultiGraph* LongKineticEnergy;
    TMultiGraph* PolarAngleChange;
    TMultiGraph* StepLength;

    TMultiGraph* MagneticMomentChange;
    TMultiGraph* TotalEnergyChange;
    TMultiGraph* CumulTotalEnergyChange;
} sMultigraphs;

static struct sGraphs_s
{
    TGraph* ZPosition;
    TGraph* Radius;
    TGraph* Steps;

    TGraph* ElectricPotential;
    TGraph* ElectricField;
    TGraph* MagneticField;

    TGraph* Time;
    TGraph* Speed;
    TGraph* Acceleration;

    TGraph* KineticEnergy;
    TGraph* PolarAngle;
    TGraph* AziAngle;

    TGraph* LongKineticEnergy;
    TGraph* PolarAngleChange;
    TGraph* StepLength;

    TGraph* MagneticMomentChange;
    TGraph* TotalEnergyChange;
    TGraph* CumulTotalEnergyChange;
} sGraphs;

const int graphcolors[14] = {kBlack,
                             kRed + 2,
                             kGreen + 2,
                             kBlue + 2,
                             kMagenta + 2,
                             kYellow + 2,
                             kCyan + 2,
                             kPink - 1,
                             kSpring - 1,
                             kAzure - 1,
                             kOrange - 1,
                             kTeal - 1,
                             kViolet - 1,
                             kGray};

#define READ_VALUE(xREADER, xNAME, xDEFAULT, xTYPE, xGETTER)                                                           \
    ((xREADER).Exists<xTYPE>(xNAME) ? (xREADER).Get<xTYPE>(xNAME).xGETTER : (xDEFAULT))
#define READ_STRING(xREADER, xNAME, xDEFAULT)   READ_VALUE(xREADER, xNAME, xDEFAULT, KSString, Value())
#define READ_INT(xREADER, xNAME, xDEFAULT)      READ_VALUE(xREADER, xNAME, xDEFAULT, KSInt, Value())
#define READ_UINT(xREADER, xNAME, xDEFAULT)     READ_VALUE(xREADER, xNAME, xDEFAULT, KSUInt, Value())
#define READ_DOUBLE(xREADER, xNAME, xDEFAULT)   READ_VALUE(xREADER, xNAME, xDEFAULT, KSDouble, Value())
#define READ_VECTOR_Z(xREADER, xNAME, xDEFAULT) READ_VALUE(xREADER, xNAME, xDEFAULT, KSThreeVector, Value().Z())
#define READ_VECTOR_R(xREADER, xNAME, xDEFAULT) READ_VALUE(xREADER, xNAME, xDEFAULT, KSThreeVector, Value().Perp())
#define READ_VECTOR_MAG(xREADER, xNAME, xDEFAULT)                                                                      \
    READ_VALUE(xREADER, xNAME, xDEFAULT, KSThreeVector, Value().Magnitude())
#define READ_VECTOR_DOTP(xREADER, xNAMEA, xNAMEB, xDEFAULT)                                                            \
    ((xREADER).Exists<KSThreeVector>(xNAMEA) && (xREADER).Exists<KSThreeVector>(xNAMEB)                                \
         ? (xREADER).Get<KSThreeVector>(xNAMEA).Value().Dot((xREADER).Get<KSThreeVector>(xNAMEB).Value())              \
         : (xDEFAULT))

#define PREP_MULTIGRAPH(xNAME, xTITLE)                                                                                 \
    sMultigraphs.xNAME = new TMultiGraph(#xNAME, xTITLE);                                                              \
    tOutputArray->Add(sMultigraphs.xNAME);

#define DRAW_MULTIGRAPH(xNAME, xOPTION, xXLABEL, xYLABEL)                                                              \
    sMultigraphs.xNAME->Draw(xOPTION);                                                                                 \
    sMultigraphs.xNAME->GetXaxis()->SetTitle(xXLABEL);                                                                 \
    sMultigraphs.xNAME->GetXaxis()->SetTitleOffset(1.2);                                                               \
    sMultigraphs.xNAME->GetXaxis()->SetTitleSize(0.03);                                                                \
    sMultigraphs.xNAME->GetXaxis()->SetLabelSize(0.03);                                                                \
    sMultigraphs.xNAME->GetYaxis()->SetTitle(xYLABEL);                                                                 \
    sMultigraphs.xNAME->GetYaxis()->SetTitleOffset(1.8);                                                               \
    sMultigraphs.xNAME->GetYaxis()->SetTitleSize(0.03);                                                                \
    sMultigraphs.xNAME->GetYaxis()->SetLabelSize(0.03);

#define PREP_GRAPH(xNAME, xCOLOR)                                                                                      \
    sGraphs.xNAME = new TGraph();                                                                                      \
    sGraphs.xNAME->SetLineColor(xCOLOR);                                                                               \
    sMultigraphs.xNAME->Add(sGraphs.xNAME);

#define UPDATE_GRAPH(xNAME, xXVALUE, xYVALUE) sGraphs.xNAME->SetPoint(sGraphs.xNAME->GetN(), (xXVALUE), (xYVALUE));

#define PRINT_GRAPH(xNAME)                                                                                             \
    std::cout << "  " << #xNAME << ":"                                                                                 \
              << "\tmin=" << TMath::MinElement(sGraphs.xNAME->GetN(), sGraphs.xNAME->GetY())                           \
              << "\tmax=" << TMath::MaxElement(sGraphs.xNAME->GetN(), sGraphs.xNAME->GetY())                           \
              << "\tmean=" << sGraphs.xNAME->GetMean() << "\trms=" << sGraphs.xNAME->GetRMS(2)                         \
              << "\tarea=" << sGraphs.xNAME->Integral() << std::endl;


int AnalyzeFile(katrin::KRootFile* aRootFile, const po::variables_map& aOptions)
{
    static unsigned int tIndex = 0;

    int tColor = graphcolors[tIndex];

    KSReadFileROOT tReader;
    tReader.OpenFile(aRootFile);

    KSReadRunROOT& tRunReader = tReader.GetRun();
    KSReadEventROOT& tEventReader = tReader.GetEvent();
    KSReadTrackROOT& tTrackReader = tReader.GetTrack();
    KSReadStepROOT& tStepReader = tReader.GetStep();

    KSReadObjectROOT& tTrackWorld = tTrackReader.GetObject("output_track_world");
    //KSReadObjectROOT&   tStepWorld      = tStepReader.Get( "output_step_world" );

    for (tRunReader = 0; tRunReader <= tRunReader.GetLastRunIndex(); tRunReader++) {
        unsigned int tRunID = tRunReader.GetRunIndex();
        if (aOptions.count("run")) {
            vector<unsigned int> tRuns = aOptions["run"].as<vector<unsigned int>>();
            if (std::find(tRuns.begin(), tRuns.end(), tRunID) == tRuns.end())
                continue;
        }

        for (tEventReader = tRunReader.GetFirstEventIndex(); tEventReader <= tRunReader.GetLastEventIndex();
             tEventReader++) {
            unsigned int tEventID = tEventReader.GetEventIndex() - tRunReader.GetFirstEventIndex();
            if (aOptions.count("event")) {
                vector<unsigned int> tEvents = aOptions["event"].as<vector<unsigned int>>();
                if (std::find(tEvents.begin(), tEvents.end(), tEventID) == tEvents.end())
                    continue;
            }

            for (tTrackReader = tEventReader.GetFirstTrackIndex(); tTrackReader <= tEventReader.GetLastTrackIndex();
                 tTrackReader++) {
                unsigned int tTrackID = tTrackReader.GetTrackIndex() - tEventReader.GetFirstTrackIndex();
                if (aOptions.count("track")) {
                    vector<unsigned int> tTracks = aOptions["track"].as<vector<unsigned int>>();
                    if (std::find(tTracks.begin(), tTracks.end(), tTrackID) == tTracks.end())
                        continue;
                }

                if (!tTrackWorld.Valid())
                    continue;

                if (aOptions.count("terminator")) {
                    string tTerminator = READ_STRING(tTrackWorld, "terminator_name", "");
                    if (tTerminator != aOptions["terminator"].as<string>())
                        continue;
                }

                if (aOptions.count("generator")) {
                    string tGenerator = READ_STRING(tTrackWorld, "creator_name", "");
                    if (tGenerator != aOptions["generator"].as<string>())
                        continue;
                }

                std::cout << "Analyzing track #" << tTrackID << " of event #" << tEventID << std::endl;

                PREP_GRAPH(ZPosition, tColor);
                PREP_GRAPH(Radius, tColor);
                PREP_GRAPH(Steps, tColor);
                PREP_GRAPH(ElectricPotential, tColor);
                PREP_GRAPH(ElectricField, tColor);
                PREP_GRAPH(MagneticField, tColor);
                PREP_GRAPH(Time, tColor);
                PREP_GRAPH(Speed, tColor);
                PREP_GRAPH(Acceleration, tColor);
                PREP_GRAPH(KineticEnergy, tColor);
                PREP_GRAPH(PolarAngle, tColor);
                PREP_GRAPH(AziAngle, tColor);
                PREP_GRAPH(LongKineticEnergy, tColor);
                PREP_GRAPH(PolarAngleChange, tColor);
                PREP_GRAPH(StepLength, tColor);
                PREP_GRAPH(MagneticMomentChange, tColor);
                PREP_GRAPH(TotalEnergyChange, tColor);
                PREP_GRAPH(CumulTotalEnergyChange, tColor);

                if (tStepReader.HasObject("output_step_world")) {
                    KSReadObjectROOT& tStepWorld = tStepReader.GetObject("output_step_world");

                    bool tIsFirstStep = true;
                    double tCumulTotalEnergyChange = 0.;
                    double tLastPathLength = 0.;
                    double tLastPolarAngle = 0.;
                    double tLastTime = 0.;
                    double tLastSpeed = 0.;
                    double tLastMagneticMoment = 0.;
                    double tLastTotalEnergy = 0.;
                    double tLastDirection = 0.;

                    for (tStepReader = tTrackReader.GetFirstStepIndex(); tStepReader <= tTrackReader.GetLastStepIndex();
                         tStepReader++) {
                        unsigned int tStepID = tStepReader.GetStepIndex() - tTrackReader.GetFirstStepIndex();

                        if (!tStepWorld.Valid())
                            continue;

                        double tPositionZ = READ_VECTOR_Z(tStepWorld, "position", 0.);
                        double tPositionR = READ_VECTOR_R(tStepWorld, "position", 0.);
                        double tTime = READ_DOUBLE(tStepWorld, "total_time", 0.);
                        tTime = READ_DOUBLE(tStepWorld, "time", tTime);  // fallback
                        double tPathLength = READ_DOUBLE(tStepWorld, "total_length", 0.);
                        tPathLength = READ_DOUBLE(tStepWorld, "length", tPathLength);  // fallback
                        double tSpeed = READ_DOUBLE(tStepWorld, "speed", 0.);
                        //double tMomentumMag     = READ_VECTOR_MAG( tStepWorld,  "momentum",                 0. );
                        double tEFieldMag = READ_VECTOR_MAG(tStepWorld, "electric_field", 0.);
                        double tBFieldMag = READ_VECTOR_MAG(tStepWorld, "magnetic_field", 0.);
                        double tEPotential = READ_DOUBLE(tStepWorld, "electric_potential", 0.);
                        double tKinEnergy = READ_DOUBLE(tStepWorld, "kinetic_energy", 0.);
                        double tPolarAngle = READ_DOUBLE(tStepWorld, "polar_angle_to_b", 0.);
                        double tAziAngle = READ_DOUBLE(tStepWorld, "azimuthal_angle_to_x", 0.);
                        double tMagneticMoment = READ_DOUBLE(tStepWorld, "orbital_magnetic_moment", 0.);
                        double tTotalEnergy = READ_DOUBLE(tStepWorld, "total_energy", tKinEnergy - tEPotential);
                        double tLongKinEnergy =
                            READ_DOUBLE(tStepWorld,
                                        "longitudinal_kinetic_energy",
                                        tKinEnergy * cos(tPolarAngle * katrin::KConst::Pi() / 180.) *
                                            cos(tPolarAngle * katrin::KConst::Pi() / 180.));
                        double tDirection = READ_VECTOR_DOTP(tStepWorld, "magnetic_field", "momentum", 0.);

                        if (tPolarAngle > 90.)
                            tPolarAngle = 180. - tPolarAngle;

                        if (!tIsFirstStep) {
                            if (aOptions.count("nomirror") && (tDirection * tLastDirection < 0.))
                                continue;
                        }

                        UPDATE_GRAPH(ZPosition, tPathLength, tPositionZ);
                        UPDATE_GRAPH(Radius, tPathLength, tPositionR);
                        UPDATE_GRAPH(Steps, tPathLength, tStepID);
                        UPDATE_GRAPH(ElectricField, tPathLength, tEFieldMag);
                        UPDATE_GRAPH(MagneticField, tPathLength, tBFieldMag);
                        UPDATE_GRAPH(ElectricPotential, tPathLength, tEPotential);
                        UPDATE_GRAPH(Time, tPathLength, tTime);
                        UPDATE_GRAPH(Speed, tPathLength, tSpeed);
                        //UPDATE_GRAPH( Acceleration,             tPathLength, tAcceleration; );
                        UPDATE_GRAPH(KineticEnergy, tPathLength, tKinEnergy);
                        UPDATE_GRAPH(PolarAngle, tPathLength, tPolarAngle);
                        UPDATE_GRAPH(AziAngle, tPathLength, tAziAngle);
                        UPDATE_GRAPH(LongKineticEnergy, tPathLength, tLongKinEnergy);
                        //UPDATE_GRAPH( PolarAngleChange,         tPathLength, tPolarAngleChange );
                        //UPDATE_GRAPH( StepLength,               tPathLength, tStepLength );
                        //UPDATE_GRAPH( MagneticMomentChange,     tPathLength, tMagneticMomentChange );
                        //UPDATE_GRAPH( TotalEnergyChange,        tPathLength, tTotalEnergyChange );
                        //UPDATE_GRAPH( CumulTotalEnergyChange,   tPathLength, tCumulTotalEnergyChange );

                        if (!tIsFirstStep) {
                            double tAcceleration = (tSpeed - tLastSpeed) / (tTime - tLastTime);
                            double tStepLength = tPathLength - tLastPathLength;
                            double tPolarAngleChange = fmod(tPolarAngle - tLastPolarAngle, 180.);
                            double tMagneticMomentChange =
                                2. * (tMagneticMoment - tLastMagneticMoment) / (tMagneticMoment + tLastMagneticMoment);
                            double tTotalEnergyChange =
                                2. * (tTotalEnergy - tLastTotalEnergy) / (tTotalEnergy + tLastTotalEnergy);

                            tCumulTotalEnergyChange += fabs(tTotalEnergy - tLastTotalEnergy);

                            UPDATE_GRAPH(Acceleration, tPathLength, tAcceleration);
                            UPDATE_GRAPH(PolarAngleChange, tPathLength, tPolarAngleChange);
                            UPDATE_GRAPH(StepLength, tPathLength, tStepLength);
                            UPDATE_GRAPH(MagneticMomentChange, tPathLength, tMagneticMomentChange);
                            UPDATE_GRAPH(TotalEnergyChange, tPathLength, tTotalEnergyChange);
                            UPDATE_GRAPH(CumulTotalEnergyChange, tPathLength, tCumulTotalEnergyChange);
                        }

                        tIsFirstStep = false;
                        tLastPathLength = tPathLength;
                        tLastPolarAngle = tPolarAngle;
                        tLastTime = tTime;
                        tLastSpeed = tSpeed;
                        tLastTotalEnergy = tTotalEnergy;
                        tLastMagneticMoment = tMagneticMoment;
                        tLastDirection = tDirection;
                    }

                    PRINT_GRAPH(ZPosition);
                    //PRINT_GRAPH( LongKineticEnergy );
                    //PRINT_GRAPH( StepLength );
                    //PRINT_GRAPH( TotalEnergyChange );
                    //PRINT_GRAPH( MagneticMomentChange );
                }
            }
        }
    }

    tReader.CloseFile();
    tIndex++;

    return 0;
}

int main(int argc, char** argv)
{
    //katrin::KMessageTable::GetInstance().SetTerminalVerbosity( eDebug );
    //katrin::KMessageTable::GetInstance().SetLogVerbosity( eDebug );

    po::positional_options_description popt;
    popt.add("file", -1);

    po::options_description opt("Allowed options");
    opt.add_options()("help,h",
                      "produce help message")("run,R", po::value<vector<unsigned int>>(), "set run # to process")(
        "event,E",
        po::value<vector<unsigned int>>(),
        "set event # to process")("track,T", po::value<vector<unsigned int>>(), "set track # to process")(
        "terminator,t",
        po::value<string>(),
        "select tracks by terminator")("generator,g", po::value<string>(), "select tracks by generator")(
        "nomirror,M",
        "do not show reflected electrons")("file,f", po::value<vector<string>>(), "ROOT input file")(
        "output,o",
        po::value<string>()->default_value(argv[0]),
        "output filename (without extension)");

    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv).options(opt).positional(popt).run(), vm);

    if (vm.count("help") > 0 || vm.count("file") < 1) {
        cout << "Usage: " << argv[0] << " [options] <filename> [filename [...]]" << endl;
        cout << opt << endl;
        return 1;
    }

    auto* tOutputArray = new TObjArray();

    PREP_MULTIGRAPH(ZPosition, "Axial Position");
    PREP_MULTIGRAPH(Radius, "Radial Position");
    PREP_MULTIGRAPH(Steps, "Total Steps");
    PREP_MULTIGRAPH(ElectricPotential, "Electric Potential");
    PREP_MULTIGRAPH(ElectricField, "Electric Field");
    PREP_MULTIGRAPH(MagneticField, "Magnetic Field");
    PREP_MULTIGRAPH(Time, "Time of Flight");
    PREP_MULTIGRAPH(Speed, "Particle Speed");
    PREP_MULTIGRAPH(Acceleration, "Particle Acceleration");
    PREP_MULTIGRAPH(KineticEnergy, "Kinetic Energy");
    PREP_MULTIGRAPH(PolarAngle, "Pitch Angle");
    PREP_MULTIGRAPH(AziAngle, "Azimuthal Angle");
    PREP_MULTIGRAPH(LongKineticEnergy, "Long. Kinetic Energy");
    PREP_MULTIGRAPH(PolarAngleChange, "Pitch Angle Change");
    PREP_MULTIGRAPH(StepLength, "Step Length");
    PREP_MULTIGRAPH(MagneticMomentChange, "Magnetic Moment Change");
    PREP_MULTIGRAPH(TotalEnergyChange, "Total Energy Change");
    PREP_MULTIGRAPH(CumulTotalEnergyChange, "Cumul. Total Energy Change");

    katrin::KRootFile* tRootFile;
    vector<string> tInputFiles = vm["file"].as<vector<string>>();
    for (auto& tInputFile : tInputFiles) {
        const string& tFilename = tInputFile;
        mainmsg(eNormal) << "Reading from ROOT file <" << tFilename << "> ..." << eom;
        tRootFile = katrin::KRootFile::CreateOutputRootFile(tFilename);
        AnalyzeFile(tRootFile, vm);
        delete tRootFile;
    }

    auto* tCanvas = new TCanvas("canvas", "", 1500, 3000);
    tCanvas->Divide(3, 6);

    tCanvas->cd(1);
    DRAW_MULTIGRAPH(ZPosition, "AL", "s [m]", "z [m]");
    tCanvas->cd(2);
    DRAW_MULTIGRAPH(Radius, "AL", "s [m]", "r [m]");
    tCanvas->cd(3);
    DRAW_MULTIGRAPH(Steps, "AL", "s [m]", "n [1]");

    tCanvas->cd(4);
    DRAW_MULTIGRAPH(ElectricPotential, "AL", "s [m]", "U [V]");
    tCanvas->cd(5);
    DRAW_MULTIGRAPH(ElectricField, "AL", "s [m]", "|E| [V/m]");
    tCanvas->cd(6);
    DRAW_MULTIGRAPH(MagneticField, "AL", "s [m]", "|B| [T]");

    tCanvas->cd(7);
    DRAW_MULTIGRAPH(Time, "AL", "s [m]", "t [s]");
    tCanvas->cd(8);
    DRAW_MULTIGRAPH(Speed, "AL", "s [m]", "v [m/s]");
    tCanvas->cd(9);
    DRAW_MULTIGRAPH(Acceleration, "AL", "s [m]", "a [m/s^2]");

    tCanvas->cd(10);
    DRAW_MULTIGRAPH(KineticEnergy, "AL", "s [m]", "E_{kin} [eV]");
    tCanvas->cd(11);
    DRAW_MULTIGRAPH(PolarAngle, "AL", "s [m]", "#theta [deg]");
    tCanvas->cd(12);
    DRAW_MULTIGRAPH(AziAngle, "AL", "s [m]", "#phi [deg]");

    tCanvas->cd(13);
    DRAW_MULTIGRAPH(LongKineticEnergy, "AL", "s [m]", "E_{long} [eV]");
    tCanvas->cd(14);
    DRAW_MULTIGRAPH(PolarAngleChange, "AL", "s [m]", "#Delta #theta [deg]");
    tCanvas->cd(15);
    DRAW_MULTIGRAPH(StepLength, "AL", "s [m]", "#Delta s [m]");

    tCanvas->cd(16);
    DRAW_MULTIGRAPH(MagneticMomentChange, "AL", "s [m]", "#delta #mu [1]");
    tCanvas->cd(17);
    DRAW_MULTIGRAPH(TotalEnergyChange, "AL", "s [m]", "#delta E [1]");
    tCanvas->cd(18);
    DRAW_MULTIGRAPH(CumulTotalEnergyChange, "AL", "s [m]", "#Sigma(#Delta E) [eV]");

    tCanvas->Modified();
    tCanvas->Update();

    string tOutputFile = vm["output"].as<string>();

    tCanvas->SaveAs((tOutputFile + string(".png")).c_str());
    //tCanvas->SaveAs( (tOutputFile + string(".pdf")).c_str() );
    delete tCanvas;

    auto* tFile = new TFile((tOutputFile + string(".root")).c_str(), "RECREATE");
    tOutputArray->Write();
    tFile->Close();
    delete tFile;

    return 0;
}
