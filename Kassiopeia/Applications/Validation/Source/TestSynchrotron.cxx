#include "KConst.h"
#include "KSMainMessage.h"
#include "KSReadFileROOT.h"

#include <cmath>
#include <limits>

using namespace Kassiopeia;
using namespace katrin;
using namespace std;

int main()
{
    katrin::KMessageTable::GetInstance().SetTerminalVerbosity(eDebug);
    katrin::KMessageTable::GetInstance().SetLogVerbosity(eDebug);

    KRootFile* tRootFile = KRootFile::CreateOutputRootFile("TestSynchrotron.root");

    KSReadFileROOT tReader;
    tReader.OpenFile(tRootFile);

    KSReadRunROOT& tRunReader = tReader.GetRun();
    KSReadEventROOT& tEventReader = tReader.GetEvent();
    KSReadTrackROOT& tTrackReader = tReader.GetTrack();
    //    KSReadStepROOT& tStepReader= tReader.GetStep();

    KSReadObjectROOT& tTrackOutput = tTrackReader.GetObject("output_track_world");
    auto& tTrackID = tTrackOutput.Get<KSInt>("track_id");
    //    KSUInt& tTotalSteps = tTrackOutput.Get< KSUInt >( "total_steps" );
    //    KSDouble& tTrackEnergyLoss = tTrackOutput.Get< KSDouble >( "track_energy_loss" );
    auto& tInitialPosition = tTrackOutput.Get<KSThreeVector>("initial_position");
    auto& tFinalPosition = tTrackOutput.Get<KSThreeVector>("final_position");
    auto& tInitialMomentum = tTrackOutput.Get<KSThreeVector>("initial_momentum");
    //    KSThreeVector& tFinalMomentum = tTrackOutput.Get< KSThreeVector >( "final_momentum" );
    auto& tInitialMagfield = tTrackOutput.Get<KSThreeVector>("initial_magnetic_field");
    auto& tFinalMagfield = tTrackOutput.Get<KSThreeVector>("final_magnetic_field");
    auto& tInitialEKin = tTrackOutput.Get<KSDouble>("initial_kinetic_energy");
    auto& tFinalEKin = tTrackOutput.Get<KSDouble>("final_kinetic_energy");
    auto& tInitialTheta = tTrackOutput.Get<KSDouble>("initial_polar_angle_to_b");
    auto& tFinalTheta = tTrackOutput.Get<KSDouble>("final_polar_angle_to_b");


    for (tRunReader = 0; tRunReader <= tRunReader.GetLastRunIndex(); tRunReader++) {
        for (tEventReader = tRunReader.GetFirstEventIndex(); tEventReader <= tRunReader.GetLastEventIndex();
             tEventReader++) {
            for (tTrackReader = tEventReader.GetFirstTrackIndex(); tTrackReader <= tEventReader.GetLastTrackIndex();
                 tTrackReader++) {
                if (!tTrackReader.Valid()) {
                    continue;
                }

                mainmsg(eNormal) << "Analyzing track " << tTrackID.Value() + 1 << eom;

                //calculating analytic synchrotron loss
                // Delta E = -mu_0 * q^4 / (3 Pi c m^3) * B^2 * E_orth * gamma * t
                double tMagfieldMag = (tInitialMagfield.Value() + tFinalMagfield.Value()).Magnitude() / 2.0;
                mainmsg(eNormal) << "Assuming constant magnetic field of <" << tMagfieldMag << "> T" << eom;

                double tLength = (tFinalPosition.Value() - tInitialPosition.Value()).Magnitude();
                mainmsg(eNormal) << "Track length is <" << tLength << "> m" << eom;

                double tTheta_SI = tInitialTheta.Value() * katrin::KConst::Pi() / 180.;
                double tMomentum = tInitialMomentum.Value().Magnitude();
                double tEnergy_SI = tInitialEKin.Value() * katrin::KConst::Q();

                //assuming particle is electron
                double tFactor = -1.0 * katrin::KConst::MuNull() * pow(katrin::KConst::Q(), 4) /
                                 (3.0 * katrin::KConst::Pi() * katrin::KConst::C() * pow(katrin::KConst::M_el_kg(), 3));

                double tGamma = sqrt(1.0 + tMomentum * tMomentum /
                                               (katrin::KConst::M_el_kg() * katrin::KConst::M_el_kg() *
                                                katrin::KConst::C() * katrin::KConst::C()));
                double tVelocity = (1. / (katrin::KConst::M_el_kg() * tGamma)) * tMomentum;
                double tTime = tLength / (cos(tTheta_SI) * tVelocity);

                double tDeltaE_ana = tFactor * tMagfieldMag * tMagfieldMag * tEnergy_SI * sin(tTheta_SI) *
                                     sin(tTheta_SI) * tGamma * tTime;
                tDeltaE_ana /= katrin::KConst::Q();

                mainmsg(eNormal) << "Analytic synchrotron energie loss is <" << tDeltaE_ana * 1000 << "> meV " << eom;

                double tDeltaE_sim = tFinalEKin.Value() - tInitialEKin.Value();

                mainmsg(eNormal) << "Simulation synchrotron energie loss is <" << tDeltaE_sim * 1000 << "> meV " << eom;
                mainmsg(eNormal) << "Difference is <" << tDeltaE_ana * 1000 - tDeltaE_sim * 1000 << "> meV" << eom;
                //			    mainmsg( eNormal ) <<"Track energy loss is <"<<tTrackEnergyLoss.Value()*1000<<"> eV "<<eom;


                double tEKinFinal_orth_ana = tInitialEKin.Value() * sin(tTheta_SI) * sin(tTheta_SI) + tDeltaE_ana;
                double tEKinFinal_ana = tInitialEKin.Value() + tDeltaE_ana;
                double tThetaFinal_ana =
                    asin(sqrt(tEKinFinal_orth_ana / tEKinFinal_ana)) * 180.0 / katrin::KConst::Pi();
                double tDeltaTheta_ana = tThetaFinal_ana - tInitialTheta.Value();

                mainmsg(eNormal) << "Analytic polar angle changed from <" << tInitialTheta.Value() << "> degree to <"
                                 << tThetaFinal_ana << "> degree, delta is <" << tDeltaTheta_ana << "> degree " << eom;

                double tDeltaTheta = tFinalTheta.Value() - tInitialTheta.Value();
                mainmsg(eNormal) << "Simulation polar angle changed from <" << tInitialTheta.Value() << "> degree to <"
                                 << tFinalTheta.Value() << "> degree, delta is <" << tDeltaTheta << "> degree " << eom;

                double tEKin_orth_initial = tInitialEKin.Value() *
                                            sin(tInitialTheta.Value() * katrin::KConst::Pi() / 180.) *
                                            sin(tInitialTheta.Value() * katrin::KConst::Pi() / 180.);
                double tEKin_orth_final = tFinalEKin.Value() * sin(tFinalTheta.Value() * katrin::KConst::Pi() / 180.) *
                                          sin(tFinalTheta.Value() * katrin::KConst::Pi() / 180.);
                mainmsg(eNormal) << "Simulation energy loss in orthogonal direction <"
                                 << (tEKin_orth_final - tEKin_orth_initial) * 1000 << "> meV " << eom;

                double tEKin_long_initial = tInitialEKin.Value() *
                                            cos(tInitialTheta.Value() * katrin::KConst::Pi() / 180.) *
                                            cos(tInitialTheta.Value() * katrin::KConst::Pi() / 180.);
                double tEKin_long_final = tFinalEKin.Value() * cos(tFinalTheta.Value() * katrin::KConst::Pi() / 180.) *
                                          cos(tFinalTheta.Value() * katrin::KConst::Pi() / 180.);
                mainmsg(eNormal) << "Simulation energy loss in longitudinal direction <"
                                 << (tEKin_long_final - tEKin_long_initial) * 1000 << "> meV " << eom;
            }
        }
    }


    tReader.CloseFile();

    delete tRootFile;

    return 0;
}
