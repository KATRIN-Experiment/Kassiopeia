#include "KEMRootFieldCanvas.hh"

#include "TBox.h"
#include "TCanvas.h"
#include "TColor.h"
#include "TEllipse.h"
#include "TH1F.h"
#include "TH2F.h"
#include "TH3F.h"
#include "TLine.h"
#include "TMarker.h"
#include "TStyle.h"

#include <cmath>
#include <stdexcept>
#include <vector>

namespace KEMField
{
KEMRootFieldCanvas::KEMRootFieldCanvas(double x_1, double x_2, double y_1, double y_2, double zmir, bool isfull) :
    KEMFieldCanvas(x_1, x_2, y_1, y_2, zmir, isfull)
{
    InitializeCanvas();
}

void KEMRootFieldCanvas::InitializeCanvas()
{
    int SCALE1 = 800;
    int SCALE2;
    if (full)
        SCALE2 = 800;
    else
        SCALE2 = 400;

    canvas = new TCanvas("canvas", "Geometry Canvas", 5, 5, SCALE1, SCALE2);
    canvas->SetBorderMode(0);
    canvas->SetFillColor(kWhite);

    gPad->Range(x1, y1, x2, y2);

    // Draw x,y axis

    hist = new TH2F("empty_hist", "", 10, x1, x2, 10, y1, y2);

    hist->SetStats(false);
    hist->Draw();
}

//______________________________________________________________________________

KEMRootFieldCanvas::~KEMRootFieldCanvas()
{
    delete canvas;
    delete hist;
}

//______________________________________________________________________________

void KEMRootFieldCanvas::DrawGeomRZ(const std::string& conicSectfile, const std::string& wirefile,
                                    const std::string& coilfile)
{
    // Read in the conicSects

    if (conicSectfile != "NULL") {
        FILE* inputfull = fopen(conicSectfile.c_str(), "r");

        int NconicSect;

        int _ret = fscanf(inputfull, "%i", &NconicSect);

        if (_ret != 1) {
            throw std::runtime_error("Could not read NconicSect");
        }

        if (NconicSect < 0) {
            throw std::runtime_error("NconicSect must be positive");
        }

        size_t nConicSect = static_cast<size_t>(NconicSect);
        std::vector<TLine*> eline(nConicSect);

        std::vector<double> ex_0(nConicSect);
        std::vector<double> ey_0(nConicSect);
        std::vector<double> ex_1(nConicSect);
        std::vector<double> ey_1(nConicSect);
        double temp1;
        int temp2;

        for (int s = 0; s < NconicSect; s++) {
            int _ret = fscanf(inputfull, "%le %le %le %le %le %i", &ex_0[s], &ey_0[s], &ex_1[s], &ey_1[s], &temp1, &temp2);

            if (_ret < 4 || _ret > 6) {
                throw std::runtime_error("Could not read conic section");
            }

            eline[s] = new TLine(ex_0[s], ey_0[s], ex_1[s], ey_1[s]);
        }

        fclose(inputfull);

        // Draw conicSects

        for (int i = 0; i < NconicSect; i++) {
            eline[i]->SetLineWidth(1);
            //     eline[i]->SetLineColor(i+1);
            eline[i]->Draw();
        }

        if (full) {
            for (int i = 0; i < NconicSect; i++) {
                eline[i] = new TLine(ex_0[i], -ey_0[i], ex_1[i], -ey_1[i]);
                eline[i]->SetLineWidth(1);
                eline[i]->Draw();
            }
        }

        if (zmirror != 1.e10) {
            for (int i = 0; i < NconicSect; i++) {
                eline[i] = new TLine(2 * zmirror - ex_0[i], ey_0[i], 2 * zmirror - ex_1[i], ey_1[i]);
                eline[i]->SetLineWidth(1);
                eline[i]->Draw();
            }

            if (full) {
                for (int i = 0; i < NconicSect; i++) {
                    eline[i] = new TLine(2 * zmirror - ex_0[i], -ey_0[i], 2 * zmirror - ex_1[i], -ey_1[i]);
                    eline[i]->SetLineWidth(1);
                    eline[i]->Draw();
                }
            }
        }
    }

    // Read in the wires

    if (wirefile != "NULL") {
        FILE* inputwire = fopen(wirefile.c_str(), "r");

        int Nwire;

        int _ret = fscanf(inputwire, "%i", &Nwire);

        if (_ret != 1) {
            throw std::runtime_error("Could not read Nwire");
        }

        if (Nwire < 0) {
            throw std::runtime_error("Nwire must be positive");
        }

        size_t nWire = static_cast<size_t>(Nwire);
        std::vector<TLine*> wline(nWire);

        std::vector<double> wx_0(nWire);
        std::vector<double> wy_0(nWire);
        std::vector<double> wx_1(nWire);
        std::vector<double> wy_1(nWire);
        std::vector<double> wd(nWire);
        double temp3;
        int temp4;
        double temp5;
        int temp6;

        for (int s = 0; s < Nwire; s++) {
            int _ret = fscanf(inputwire,
                          "%le %le %le %le %le %le %i %le %i",
                          &wx_0[s],
                          &wy_0[s],
                          &wx_1[s],
                          &wy_1[s],
                          &wd[s],
                          &temp3,
                          &temp4,
                          &temp5,
                          &temp6);
            
            if (_ret < 5 || _ret > 9) {
                throw std::runtime_error("Could not read wire");
            }

            wline[s] = new TLine(wx_0[s], wy_0[s], wx_1[s], wy_1[s]);
        }

        fclose(inputwire);

        // Draw wires

        double thickScale = 10;

        for (int i = 0; i < Nwire; i++) {
            wline[i]->SetLineWidth(wd[i] * thickScale);
            //     wline[i]->SetLineColor(i+1);
            wline[i]->Draw();
        }

        if (full) {
            for (int i = 0; i < Nwire; i++) {
                wline[i] = new TLine(wx_0[i], -wy_0[i], wx_1[i], -wy_1[i]);
                wline[i]->SetLineWidth(wd[i] * thickScale);
                wline[i]->Draw();
            }
        }

        if (zmirror != 1.e10) {
            for (int i = 0; i < Nwire; i++) {
                wline[i] = new TLine(2 * zmirror - wx_0[i], wy_0[i], 2 * zmirror - wx_1[i], wy_1[i]);
                wline[i]->SetLineWidth(wd[i] * thickScale);
                wline[i]->Draw();
            }

            if (full) {
                for (int i = 0; i < Nwire; i++) {
                    wline[i] = new TLine(2 * zmirror - wx_0[i], -wy_0[i], 2 * zmirror - wx_1[i], -wy_1[i]);
                    wline[i]->SetLineWidth(wd[i] * thickScale);
                    wline[i]->Draw();
                }
            }
        }
    }

    // Read in magnet coils

    if (coilfile != "NULL") {
        FILE* inputcoil = fopen(coilfile.c_str(), "r");

        int Ncoil;

        int _ret = fscanf(inputcoil, "%i", &Ncoil);

        if (_ret != 1) {
            throw std::runtime_error("Could not read Ncoil");
        }

        if (Ncoil < 0) {
            throw std::runtime_error("Ncoil must be positive");
        }

        size_t nCoil = static_cast<size_t>(Ncoil);
        std::vector<TBox*> box(nCoil);

        std::vector<double> z_mid(nCoil);
        std::vector<double> r_min(nCoil);
        std::vector<double> r_thk(nCoil);
        std::vector<double> z_len(nCoil);
        double temp;

        for (int j = 0; j < Ncoil; j++) {
            int _ret = fscanf(inputcoil, "%le %le %le %le %le ", &z_mid[j], &r_min[j], &r_thk[j], &z_len[j], &temp);
            
            if (_ret < 4 || _ret > 5) {
                throw std::runtime_error("Could not read coil");
            }
            
            box[j] = new TBox((z_mid[j] - z_len[j] / 2), r_min[j], (z_mid[j] + z_len[j] / 2), (r_min[j] + r_thk[j]));
        }

        // Draw magnet coils

        for (int k = 0; k < Ncoil; k++) {
            box[k]->SetLineWidth(0);
            box[k]->SetFillStyle(3004);
            box[k]->SetFillColor(kBlue);
            box[k]->Draw();
        }

        if (full) {
            for (int k = 0; k < Ncoil; k++) {
                box[k] =
                    new TBox((z_mid[k] - z_len[k] / 2), -r_min[k], (z_mid[k] + z_len[k] / 2), -(r_min[k] + r_thk[k]));
                box[k]->SetLineWidth(0);
                box[k]->SetFillStyle(3004);
                box[k]->SetFillColor(kBlue);
                box[k]->Draw();
            }
        }
    }
}

//______________________________________________________________________________

void KEMRootFieldCanvas::DrawGeomXY(double z, const std::string& conicSectfile, const std::string& wirefile,
                                    const std::string& coilfile)
{
    // Read in the conicSects

    if (conicSectfile != "NULL") {
        FILE* inputfull = fopen(conicSectfile.c_str(), "r");

        int NconicSect;

        int _ret = fscanf(inputfull, "%i", &NconicSect);

        if (_ret != 1) {
            throw std::runtime_error("Could not read NconicSect");
        }

        if (NconicSect < 0) {
            throw std::runtime_error("NconicSect must be positive");
        }

        size_t nConicSect = static_cast<size_t>(NconicSect);
        std::vector<TEllipse*> e(nConicSect);

        std::vector<double> ez_0(nConicSect);
        std::vector<double> er_0(nConicSect);
        std::vector<double> ez_1(nConicSect);
        std::vector<double> er_1(nConicSect);
        double temp1;
        int temp2;

        for (int s = 0; s < NconicSect; s++) {
            int _ret = fscanf(inputfull, "%le %le %le %le %le %i", &ez_0[s], &er_0[s], &ez_1[s], &er_1[s], &temp1, &temp2);

            if (_ret < 4 || _ret > 6) {
                throw std::runtime_error("Could not read conic section");
            }

            if (ez_0[s] < z && ez_1[s] > z) {
                double z_tot = fabs(ez_1[s] - ez_0[s]);

                double r = er_0[s] * fabs(z - ez_0[s]) / z_tot + er_1[s] * fabs(z - ez_1[s]) / z_tot;

                e[s] = new TEllipse(0, 0, r);
                e[s]->SetLineWidth(1);
                //     e[s]->SetLineColor(i+1);
                e[s]->Draw();
            }
        }

        fclose(inputfull);
    }

    // Read in the wires

    if (wirefile != "NULL") {
        FILE* inputwire = fopen(wirefile.c_str(), "r");

        int Nwire;

        int _ret = fscanf(inputwire, "%i", &Nwire);

        if (_ret != 1) {
            throw std::runtime_error("Could not read Nwire");
        }

        if (Nwire < 0) {
            throw std::runtime_error("Nwire must be positive");
        }

        size_t nWire = static_cast<size_t>(Nwire);
        std::vector<TEllipse*> w(nWire);

        std::vector<double> wz_0(nWire);
        std::vector<double> wr_0(nWire);
        std::vector<double> wz_1(nWire);
        std::vector<double> wr_1(nWire);
        std::vector<double> wd(nWire);
        std::vector<double> phi(nWire);
        std::vector<int> numwire(nWire);
        double temp5;
        int temp6;

        for (int s = 0; s < Nwire; s++) {
            int _ret = fscanf(inputwire,
                          "%le %le %le %le %le %le %i %le %i",
                          &wz_0[s],
                          &wr_0[s],
                          &wz_1[s],
                          &wr_1[s],
                          &wd[s],
                          &phi[s],
                          &numwire[s],
                          &temp5,
                          &temp6);
            
            if (_ret < 7 || _ret > 9) {
                throw std::runtime_error("Could not read wire");
            }

            double z_tot = fabs(wz_1[s] - wz_0[s]);

            double r = wr_0[s] * fabs(z - wz_0[s]) / z_tot + wr_1[s] * fabs(z - wz_1[s]) / z_tot;

            double x = r * cos(phi[s] / 180 * 3.1415);
            double y = r * sin(phi[s] / 180 * 3.1415);

            w[s] = new TEllipse(x, y, wd[s] / 2);
            w[s]->SetLineWidth(1);
            //     w[i]->SetLineColor(i+1);
            w[s]->Draw();

            double phin = 2. * 3.1415 / numwire[s];
            double cosphin = cos(phin);
            double sinphin = sin(phin);
            double tmp;

            for (int i = 0; i < numwire[s] - 1; i++) {
                tmp = x;
                x = x * cosphin - y * sinphin;
                y = tmp * sinphin + y * cosphin;

                w[s] = new TEllipse(x, y, wd[s] / 2);

                w[s]->SetLineWidth(1);
                //     w[i]->SetLineColor(i+1);
                w[s]->Draw();
            }
        }

        fclose(inputwire);
    }

    // Read in magnet coils

    if (coilfile != "NULL") {
        FILE* inputcoil = fopen(coilfile.c_str(), "r");

        int Ncoil;

        int _ret = fscanf(inputcoil, "%i", &Ncoil);

        if (_ret != 1) {
            throw std::runtime_error("Could not read Ncoil");
        }

        if (Ncoil < 0) {
            throw std::runtime_error("Ncoil must be positive");
        }

        size_t nCoil = static_cast<size_t>(Ncoil);
        std::vector<TEllipse*> coil(nCoil);

        std::vector<double> z_mid(nCoil);
        std::vector<double> r_min(nCoil);
        std::vector<double> r_thk(nCoil);
        std::vector<double> z_len(nCoil);
        double temp;

        for (int j = 0; j < Ncoil; j++) {
            int _ret = fscanf(inputcoil, "%le %le %le %le %le ", &z_mid[j], &r_min[j], &r_thk[j], &z_len[j], &temp);
            
            if (_ret < 4 || _ret > 5) {
                throw std::runtime_error("Could not read coil");
            }
            
            if (z_mid[j] + z_len[j] / 2 > z && z_mid[j] - z_len[j] / 2 < z) {
                coil[j] = new TEllipse(0, 0, r_min[j] + r_thk[j], r_min[j]);
                coil[j]->SetLineWidth(0);
                coil[j]->SetFillStyle(3004);
                coil[j]->SetFillColor(kBlue);
                coil[j]->Draw();
            }
        }
    }
}

//______________________________________________________________________________

void KEMRootFieldCanvas::DrawFieldMapCube(double x_1, double x_2, double y_1, double y_2)
{
    if (x_1 >= x1 && x_2 <= x2 && y_1 >= y1 && y_2 <= y2) {
        auto* box = new TBox(x_1, y_1, x_2, y_2);
        box->SetFillStyle(0);
        box->SetLineColor(1);
        // box->SetLineWidth(.5);
        box->Draw();
    }
}

//______________________________________________________________________________

void KEMRootFieldCanvas::DrawFieldMap(const std::vector<double>& x, const std::vector<double>& y,
                                      const std::vector<double>& V, bool xy, double z)
{
    (void) z;

    delete hist;
    int ysize = y.size();
    if (full && !xy)
        ysize *= 2;
    hist = new TH2F("contour_hist", "", x.size(), x1, x2, ysize, y1, y2);

    int V_val = 0;

    for (double i : x) {
        for (double j : y) {
            hist->Fill(i, j, V[V_val]);
            if (full && !xy)
                hist->Fill(i, -j, V[V_val]);
            V_val++;
        }
    }

    gStyle->SetPalette(1);
    gStyle->SetOptTitle(1);
    gStyle->SetOptStat(0);

    //   TGaxis::SetMaxDigits(1);

    hist->SetContour(100);
    hist->Draw("COLZ");
}

//______________________________________________________________________________

void KEMRootFieldCanvas::DrawComparisonMap(int nPoints, const std::vector<double>& x, const std::vector<double>& y,
                                           const std::vector<double>& V1, const std::vector<double>& V2)
{

    delete hist;
    int ysize = y.size();
    if (full)
        ysize *= 2;
    hist = new TH2F("contour_hist", "", x.size(), x1, x2, ysize, y1, y2);

    for (int i = 0; i < nPoints; i++) {
        hist->Fill(x[i], y[i], (V2[i] - V1[i]));
        if (full)
            hist->Fill(x[i], -y[i], (V2[i] - V1[i]));
    }

    gStyle->SetPalette(1, nullptr);
    gStyle->SetOptTitle(1);
    gStyle->SetOptStat(0);

    hist->SetMaximum(10);
    hist->SetMinimum(-10);

    //   TGaxis::SetMaxDigits(2);

    hist->SetContour(100);
    hist->Draw("COLZ");
}

//______________________________________________________________________________

void KEMRootFieldCanvas::DrawFieldLines(const std::vector<double>& x, const std::vector<double>& y)
{
    int nLineSegs = x.size() - 1;

    size_t nLineSegsSize = nLineSegs > 0 ? static_cast<size_t>(nLineSegs) : 0;
    std::vector<TLine*> fl(nLineSegsSize);

    for (int n = 0; n < nLineSegs; n++) {
        fl[n] = new TLine(x[n], y[n], x[n + 1], y[n + 1]);
        // fl[n]->SetLineWidth(.2);
        fl[n]->SetLineColor(kGreen);
        fl[n]->Draw();
    }

    if (full) {
        for (int n = 0; n < nLineSegs; n++) {
            fl[n] = new TLine(x[n], -y[n], x[n + 1], -y[n + 1]);
            // fl[n]->SetLineWidth(.2);
            fl[n]->SetLineColor(kGreen);
            fl[n]->Draw();
        }
    }
    //   DrawGeomRZ();
}

//______________________________________________________________________________

void KEMRootFieldCanvas::LabelAxes(const std::string& xname, const std::string& yname, const std::string& zname)
{
    canvas->SetRightMargin(.15);
    hist->GetXaxis()->SetTitle(xname.c_str());
    hist->GetYaxis()->SetTitle(yname.c_str());
    hist->GetZaxis()->SetTitle(zname.c_str());
}

//______________________________________________________________________________

void KEMRootFieldCanvas::LabelCanvas(const std::string& name)
{
    hist->SetTitle(name.c_str());
}

//______________________________________________________________________________

void KEMRootFieldCanvas::SaveAs(const std::string& savename)
{
    canvas->SaveAs(savename.c_str());
}
}  // namespace KEMField
