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

    int __attribute__((unused)) _ret;

    if (conicSectfile != "NULL") {
        FILE* inputfull = fopen(conicSectfile.c_str(), "r");

        int NconicSect;

        _ret = fscanf(inputfull, "%i", &NconicSect);

        TLine* eline[NconicSect];

        double ex_0[NconicSect];
        double ey_0[NconicSect];
        double ex_1[NconicSect];
        double ey_1[NconicSect];
        double temp1;
        int temp2;

        for (int s = 0; s < NconicSect; s++) {
            _ret = fscanf(inputfull, "%le %le %le %le %le %i", &ex_0[s], &ey_0[s], &ex_1[s], &ey_1[s], &temp1, &temp2);
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

        _ret = fscanf(inputwire, "%i", &Nwire);

        TLine* wline[Nwire];

        double wx_0[Nwire];
        double wy_0[Nwire];
        double wx_1[Nwire];
        double wy_1[Nwire];
        double wd[Nwire];
        double temp3;
        int temp4;
        double temp5;
        int temp6;

        for (int s = 0; s < Nwire; s++) {
            _ret = fscanf(inputwire,
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

        _ret = fscanf(inputcoil, "%i", &Ncoil);

        TBox* box[Ncoil];

        double z_mid[Ncoil];
        double r_min[Ncoil];
        double r_thk[Ncoil];
        double z_len[Ncoil];
        double temp;

        for (int j = 0; j < Ncoil; j++) {
            _ret = fscanf(inputcoil, "%le %le %le %le %le ", &z_mid[j], &r_min[j], &r_thk[j], &z_len[j], &temp);
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

    int __attribute__((unused)) _ret;

    if (conicSectfile != "NULL") {
        FILE* inputfull = fopen(conicSectfile.c_str(), "r");

        int NconicSect;

        _ret = fscanf(inputfull, "%i", &NconicSect);

        TEllipse* e[NconicSect];

        double ez_0[NconicSect];
        double er_0[NconicSect];
        double ez_1[NconicSect];
        double er_1[NconicSect];
        double temp1;
        int temp2;

        for (int s = 0; s < NconicSect; s++) {
            _ret = fscanf(inputfull, "%le %le %le %le %le %i", &ez_0[s], &er_0[s], &ez_1[s], &er_1[s], &temp1, &temp2);

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

        _ret = fscanf(inputwire, "%i", &Nwire);

        TEllipse* w[Nwire];

        double wz_0[Nwire];
        double wr_0[Nwire];
        double wz_1[Nwire];
        double wr_1[Nwire];
        double wd[Nwire];
        double phi[Nwire];
        int numwire[Nwire];
        double temp5;
        int temp6;

        for (int s = 0; s < Nwire; s++) {
            _ret = fscanf(inputwire,
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

        _ret = fscanf(inputcoil, "%i", &Ncoil);

        TEllipse* coil[Ncoil];

        double z_mid[Ncoil];
        double r_min[Ncoil];
        double r_thk[Ncoil];
        double z_len[Ncoil];
        double temp;

        for (int j = 0; j < Ncoil; j++) {
            _ret = fscanf(inputcoil, "%le %le %le %le %le ", &z_mid[j], &r_min[j], &r_thk[j], &z_len[j], &temp);
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

    TLine* fl[nLineSegs];

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
