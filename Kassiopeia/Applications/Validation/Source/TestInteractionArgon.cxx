/*
 * TestInteractionArgon.cxx
 *
 *  Created on: 11.12.2013
 *      Author: oertlin
 */

#include "KIntCalculatorArgon.h"

#include <TAxis.h>
#include <TCanvas.h>
#include <TFile.h>
#include <TGraph.h>
#include <TGraph2D.h>
#include <TMultiGraph.h>
#include <cmath>
#include <complex>
#include <fstream>
#include <vector>
//#include <KArgumentList.h>

using namespace Kassiopeia;
using namespace std;

//map<double, vector<double> >* ReadPaseShifts();
//double GetDiffXSecAt(const double &anEnergy, const double &anAngle, const map<double, vector<double> >* phaseShifts);
//
//int main(int argc, char** argv) {
////	KArgumentList args(argc, argv);
//
//	map<double, vector<double> > *phaseShifts = ReadPaseShifts();
//
//	const unsigned int POINTS = 2000;
//	const double ENERGY_LOW = 1;
//	const double ENERGY_HIGH = 1e6;
//	const double ENERGY_DELTA = (log10(ENERGY_HIGH) - log10(ENERGY_LOW)) / POINTS;
//
//	double energy = 10;
//
//	if(argc > 1) {
//		switch(argv[1][0]) {
//		case '0':
//			energy = 10;
//			break;
//
//		case '1':
//			energy = 15;
//			break;
//
//		case '2':
//			energy = 20;
//			break;
//
//		case '3':
//			energy = 50;
//			break;
//
//		case '4':
//			energy = 100;
//			break;
//		}
//	}
//
//	stringstream file;
//	file << "TestInteractionArgon_data_" << energy << "eV.dat";
//
//	ifstream in(file.str().c_str());
//	KIntCalculatorArgonTotalCrossSectionReader dataReader(&in, 0);
//	dataReader.Read();
//
//	map<double, double> *data = dataReader.GetData();
//
//	double *x_data = new double[data->size()];
//	double *y_data = new double[data->size()];
//	double *x_inter = new double[POINTS];
//	double *y_inter = new double[POINTS];
//
//	double sigma, angle;
//	for(unsigned int i = 0; i < POINTS; ++i) {
////			energy = ENERGY_LOW + pow(10, i * ENERGY_DELTA);
//			angle = 180.0 / static_cast<double>(POINTS) * static_cast<double>(i);
//			sigma = GetDiffXSecAt(energy, angle / 180.0 * katrin::KConst::Pi(), phaseShifts);
//
//			x_inter[i] = angle;
//			y_inter[i] = sigma * 1e20 * 1e31 * (1 + (energy - 10) * 10.0 / 90.0);
//
//			//cout << angle << "\t" << sigma << endl;
//	}
//
//	unsigned int i = 0;
//	for(map<double, double>::iterator p = data->begin(); p != data->end(); ++p) {
//			x_data[i] = p->first;
//			y_data[i] = p->second * 1e20;
//
//			//cout << p->first << "\t" << p->second << endl;
//
//			++i;
//	}
//
//	TCanvas canvas("myCan", "", 800, 600);
//	canvas.cd(1);
//	//canvas.cd(1)->SetLogx();
//	canvas.cd(1)->SetLogy();
//
//	TMultiGraph mg;
//
//	TGraph* dgraph = new TGraph(data->size(), x_data, y_data);
//	dgraph->SetMarkerColor(2);
//	dgraph->SetLineColor(2);
//	dgraph->SetMarkerStyle(20);
//	dgraph->SetLineWidth(1);
//
//	TGraph* graph = new TGraph(POINTS, x_inter, y_inter);
//	graph->SetMarkerColor(46);
//	graph->SetLineColor(46);
//	graph->SetMarkerStyle(1);
//	graph->SetLineWidth(1);
//
//	mg.Add(graph);
//	mg.Add(dgraph);
//
//	mg.Draw("AP");
//
//	mg.GetXaxis()->SetTitle("Angle in degrees");
//	mg.GetYaxis()->SetTitle("#sigma_{ion} in 10^{-20} m^{2}");
//
//	file << "diffx.pdf";
//
//	canvas.SaveAs(file.str().c_str());
//
//	return 0;
//}
//
//map<double, vector<double> >* ReadPaseShifts() {
//	ifstream source("TestInteractionArgon_phase_shifts.dat");
//
//	map<double, vector<double> > *data = new map<double, vector<double> >();
//
//	vector<double> *tmpVector;
//
//	double tmp, energy;
//	unsigned int i = 0;
//	while(source >> tmp) {
//		if(0 == i) {
//			tmpVector = new vector<double>();
//			energy = tmp;
//
//			++i;
//		} else {
//			tmpVector->push_back(tmp);
//
//			++i;
//
//			if(6 == i) {
//				data->insert(pair<double, vector<double> >(energy, *tmpVector));
//
//				i = 0;
//			}
//		}
//	}
//
//	source.close();
//
//	return data;
//}
//
//double P(const unsigned int &l, const double &x) {
//	switch(l) {
//	case 0:
//		return 1;
//
//	case 1:
//		return x;
//
//	case 2:
//		return 0.5 * (3 * x * x - 1);
//
//	case 3:
//		return 0.5 * (5 * x * x * x - 3 * x);
//
//	case 4:
//		return 1.0 / 8.0 * (35 * x * x * x * x - 30 * x * x + 3);
//
//	case 5:
//		return 1.0 / 8.0 * (63 * x * x * x * x * x - 70 * x * x * x + 15 * x);
//
//	case 6:
//		return 1.0 / 16.0 * (231 * x * x * x * x * x * x - 315 * x * x * x * x + 105 * x * x - 5);
//
//	default:
//		cout << "P(" << l << ", " << x << ") is not implemented!" << endl;
//		return 0;
//	}
//}
//
//double GetDiffXSecAt(const double &anEnergy, const double &anAngle, const map<double, vector<double> >* phaseShifts) {
//	vector<double> delta = phaseShifts->at(anEnergy);
//
//	complex<double> sigma(0, 0);
//	double k = anEnergy / katrin::KConst::C() / (katrin::KConst::Hbar() * 2 * katrin::KConst::Pi());
//
//	for(unsigned int l = 0; l < 5; ++l) {
//		sigma += complex<double>(2 * l + 1, 0) * (exp(complex<double>(0, 2 * delta[l])) - 1.0) * P(l, cos(anAngle));
//	}
//
//	sigma *= 1.0 / (2 * k);
//
//	double realSigma = abs(sigma);
//
//	return realSigma * realSigma;
//}

void AddArgonCalculatorGraph(KIntCalculatorArgon* calc, TMultiGraph& mg, int dataColor = 2, int interpolColor = 46);
void AddArgonDiffXGraph(KIntCalculatorArgon* calc, TMultiGraph& mg, int dataColor = 2, int interpolColor = 46);

int main()
{
    TCanvas canvas("myCan", "", 800, 600);
    canvas.cd(1);
    //	canvas.cd(1)->SetLogx();
    //	canvas.cd(1)->SetLogy();


    TMultiGraph mg;

    //	KIntCalculatorArgonSingleIonisation singleIonization;
    //	singleIonization.Initialize();
    //	AddArgonCalculatorGraph(&singleIonization, mg);
    //
    //	KIntCalculatorArgonDoubleIonisation doubleIonization;
    //	doubleIonization.Initialize();
    //	AddArgonCalculatorGraph(&doubleIonization, mg);
    //
    KIntCalculatorArgonElastic elastic;
    elastic.Initialize();
    //	AddArgonCalculatorGraph(&elastic, mg);
    //
    //	for(unsigned int i = 0; i < 25; ++i) {
    //		KIntCalculatorArgonExcitation excitation;
    //		excitation.SetExcitationState(i + 1);
    //		excitation.Initialize();
    //		AddArgonCalculatorGraph(&excitation, mg, int(i + 1), int(i + 1));
    //	}

    //	const double SIZE = 6;
    //	double energy[] = {400, 500, 750, 1000, 2000, 3000};
    //	double angle[] = {40, 40, 40, 40, 40, 40};
    //
    //	for(unsigned int i = 0; i < SIZE; ++i) {
    //		cout << "(" << energy[i] << ", " << angle[i] << ") = " << elastic.GetDifferentialCrossSectionAt(energy[i], angle[i]) << endl;;
    //	}

    AddArgonDiffXGraph(&elastic, mg);

    //	double energy = 0.2;
    //	double angle = 2;
    //
    //	cout << "(" << energy << ", " << angle << ") = " << elastic.GetDifferentialCrossSectionAt(energy, angle) << endl;

    //	mg.Draw("AP");
    //
    //	mg.GetXaxis()->SetTitle("Angle in degree");
    //    mg.GetYaxis()->SetTitle("#sigma_{ion} in 10^{-20} m^{2}");
    //
    //    canvas.SaveAs("TestInteractionArgon.pdf");

    return 0;
}

double GetDiffXSectionH(double theta, double energy)
{
    double t = energy / 27.21;
    double c = cos(theta * katrin::KConst::Pi() / 180);
    double k = 2 * sqrt(t * (1 - c));
    return 4 * (8 + k * k) * (8 + k * k) / pow(4 + k * k, 4);
}

void AddArgonDiffXGraph(KIntCalculatorArgon* calc, TMultiGraph& mg, int dataColor, int interpolColor)
{
    map<double*, double>* data = calc->DEBUG_GetSupportingPointsDiffX();

    const double SHIFT = 1e20;

    const unsigned int POINTS = 45;
    //	const double ENERGY_LOW = 1;
    //	const double ENERGY_HIGH = 1e6;
    //	const double ENERGY_DELTA = (log10(ENERGY_HIGH) - log10(ENERGY_LOW)) / POINTS;

    const unsigned int SIZE = 37;

    const double ENERGY_MAX = 600;
    const double ENERGY_MIN = 0;

    double* x_data = new double[SIZE];
    double* y_data = new double[SIZE];
    double* x_inter = new double[POINTS * POINTS];
    double* y_inter = new double[POINTS * POINTS];
    double* z_inter = new double[POINTS * POINTS];

    double* x_inter_fa = new double[POINTS];
    double* y_inter_fa = new double[POINTS];

    ofstream out("TestInteractionArgonDiffX.log");

    double t = 300 / 27.23;
    double anAngle = 0;
    double c = cos(anAngle * katrin::KConst::Pi() / 180);
    double k = 2 * sqrt(t * (1 - c));
    double s = 4.0 * (8.0 + k * k) * (8.0 + k * k) / pow(4.0 + k * k, 4.0);

    cout << "c = " << c << "; k = " << k << "; s = " << s << endl;

    double sigma, energy = 15, angle;
    for (unsigned int i = 0, o = 0; i < POINTS; ++i) {
        angle = 180.0 / static_cast<double>(POINTS) * static_cast<double>(i);

        for (unsigned int n = 0; n < POINTS; ++n) {
            energy = ENERGY_MIN + (ENERGY_MAX - ENERGY_MIN) / static_cast<double>(POINTS) * static_cast<double>(n);

            sigma = calc->GetDifferentialCrossSectionAt(energy, angle);

            x_inter[o] = energy;
            y_inter[o] = angle;
            z_inter[o] = sigma * SHIFT;

            out << "[" << o << "]: (" << x_inter[o] << ", " << y_inter[o] << ") = " << z_inter[o] << endl;

            if (10 == i) {
                x_inter_fa[n] = energy;
                y_inter_fa[n] = sigma * SHIFT;
                //				cout << "[" << n << "] " << x_inter_fa[n] << " = " << y_inter_fa[n] << endl;
            }

            ++o;
        }
    }

    energy = 2000;
    for (unsigned int i = 0; i < POINTS; ++i) {
        angle = 180.0 / static_cast<double>(POINTS) * static_cast<double>(i);

        sigma = calc->GetDifferentialCrossSectionAt(energy, angle);

        x_inter_fa[i] = angle;
        y_inter_fa[i] = sigma * SHIFT;
    }

    //	unsigned int i = 0;
    //	for(map<double*, double>::iterator p = data->begin(); p != data->end(); ++p) {
    //		if(p->first[0] == energy) {
    //			x_data[i] = p->first[1];
    //			y_data[i] = p->second * 1e20;
    //			++i;
    //		}
    //	}

    //	TGraph* dgraph = new TGraph(SIZE, x_data, y_data);
    //	dgraph->SetMarkerColor(dataColor);
    //	dgraph->SetLineColor(dataColor);
    //	dgraph->SetMarkerStyle(20);
    //	dgraph->SetLineWidth(1);

    TCanvas canvas("myCan2", "", 800, 600);
    //	canvas.Divide(1, 2);

    //	canvas.cd(1);
    canvas.SetLogz();

    TGraph2D* graph = new TGraph2D(POINTS * POINTS, x_inter, y_inter, z_inter);
    graph->SetMarkerColor(interpolColor);
    graph->SetLineColor(interpolColor);
    graph->SetMarkerStyle(1);
    graph->SetLineWidth(1);

    canvas.SetPhi(210);
    graph->Draw("TRI2");

    TGraph2D* p = new TGraph2D();
    p->SetPoint(0, 400, 5, 34.3 * katrin::KConst::BohrRadiusSquared() * SHIFT);
    p->SetPoint(1, 400, 20, 3.4 * katrin::KConst::BohrRadiusSquared() * SHIFT);
    p->SetPoint(2, 400, 30, 1.01 * katrin::KConst::BohrRadiusSquared() * SHIFT);
    p->SetPoint(3, 400, 50, 0.305 * katrin::KConst::BohrRadiusSquared() * SHIFT);

    //	p->Draw("same p0");

    graph->GetXaxis()->SetTitle("Energy in eV");
    graph->GetYaxis()->SetTitle("Angle in degree");
    graph->GetZaxis()->SetTitle("#sigma_{ion} in 10^{-20} m^{2}");

    //	canvas.cd(2);

    TGraph* g = new TGraph(POINTS, x_inter_fa, y_inter_fa);
    g->SetMarkerColor(interpolColor);
    g->SetLineColor(interpolColor);
    g->SetMarkerStyle(1);
    g->SetLineWidth(1);
    g->GetXaxis()->SetTitle("Angle in degrees");
    g->GetYaxis()->SetTitle("#sigma_{ion} in 10^{-20} m^{2}");

    //	g->Draw("AL");

    TGraph* g2 = new TGraph();
    g2->SetMarkerStyle(8);
    // 400 eV
    //	g2->SetPoint(0, 5, 34.3 * katrin::KConst::BohrRadiusSquared() * SHIFT);
    //	g2->SetPoint(1, 20, 3.4 * katrin::KConst::BohrRadiusSquared() * SHIFT);
    //	g2->SetPoint(2, 30, 1.01 * katrin::KConst::BohrRadiusSquared() * SHIFT);
    //	g2->SetPoint(3, 50, 0.305 * katrin::KConst::BohrRadiusSquared() * SHIFT);
    //	g2->SetPoint(4, 10, 16.5 * katrin::KConst::BohrRadiusSquared() * SHIFT);
    //	g2->SetPoint(5, 15, 7.42 * katrin::KConst::BohrRadiusSquared() * SHIFT);
    //	g2->SetPoint(6, 25, 1.72 * katrin::KConst::BohrRadiusSquared() * SHIFT);
    //	g2->SetPoint(7, 35, 0.688 * katrin::KConst::BohrRadiusSquared() * SHIFT);
    //	g2->SetPoint(8, 40, 0.515 * katrin::KConst::BohrRadiusSquared() * SHIFT);
    //	g2->SetPoint(9, 45, 0.394 * katrin::KConst::BohrRadiusSquared() * SHIFT);

    // 500 eV
    //	g2->SetPoint(0, 5, 32.8 * katrin::KConst::BohrRadiusSquared() * SHIFT);
    //	g2->SetPoint(1, 10, 15.3 * katrin::KConst::BohrRadiusSquared() * SHIFT);
    //	g2->SetPoint(2, 15, 6.59 * katrin::KConst::BohrRadiusSquared() * SHIFT);
    //	g2->SetPoint(3, 20, 2.94 * katrin::KConst::BohrRadiusSquared() * SHIFT);
    //	g2->SetPoint(4, 25, 1.48 * katrin::KConst::BohrRadiusSquared() * SHIFT);
    //	g2->SetPoint(5, 30, 0.881 * katrin::KConst::BohrRadiusSquared() * SHIFT);
    //	g2->SetPoint(6, 35, 0.605 * katrin::KConst::BohrRadiusSquared() * SHIFT);
    //	g2->SetPoint(7, 40, 0.446 * katrin::KConst::BohrRadiusSquared() * SHIFT);
    //	g2->SetPoint(8, 45, 0.327 * katrin::KConst::BohrRadiusSquared() * SHIFT);
    //	g2->SetPoint(9, 50, 0.241 * katrin::KConst::BohrRadiusSquared() * SHIFT);

    // 2000 eV
    g2->SetPoint(0, 5, 24.6 * katrin::KConst::BohrRadiusSquared() * SHIFT);
    g2->SetPoint(1, 10, 6.31 * katrin::KConst::BohrRadiusSquared() * SHIFT);
    g2->SetPoint(2, 15, 1.86 * katrin::KConst::BohrRadiusSquared() * SHIFT);
    g2->SetPoint(3, 20, 0.781 * katrin::KConst::BohrRadiusSquared() * SHIFT);
    g2->SetPoint(4, 25, 0.407 * katrin::KConst::BohrRadiusSquared() * SHIFT);
    g2->SetPoint(5, 30, 0.226 * katrin::KConst::BohrRadiusSquared() * SHIFT);
    g2->SetPoint(6, 35, 0.136 * katrin::KConst::BohrRadiusSquared() * SHIFT);
    g2->SetPoint(7, 40, 0.0930 * katrin::KConst::BohrRadiusSquared() * SHIFT);
    g2->SetPoint(8, 45, 0.0630 * katrin::KConst::BohrRadiusSquared() * SHIFT);
    g2->SetPoint(9, 50, 0.0416 * katrin::KConst::BohrRadiusSquared() * SHIFT);

    //	g2->Draw("same p0");


    canvas.SaveAs("TestInteractionArgonDiffX.pdf");
    canvas.SaveAs("TestInteractionArgonDiffX.root");

    // mg.Add(graph);
    // mg.Add(dgraph);
}

void AddArgonCalculatorGraph(KIntCalculatorArgon* calc, TMultiGraph& mg, int dataColor, int interpolColor)
{
    map<double, double>* data = calc->DEBUG_GetSupportingPoints();

    const unsigned int POINTS = 5000;
    const double ENERGY_LOW = 1;
    const double ENERGY_HIGH = 1e6;
    const double ENERGY_DELTA = (log10(ENERGY_HIGH) - log10(ENERGY_LOW)) / POINTS;

    double* x_data = new double[data->size()];
    double* y_data = new double[data->size()];
    double* x_inter = new double[POINTS];
    double* y_inter = new double[POINTS];

    double sigma, energy;
    for (unsigned int i = 0; i < POINTS; ++i) {
        energy = ENERGY_LOW + pow(10, i * ENERGY_DELTA);
        calc->CalculateCrossSection(energy, sigma);

        x_inter[i] = energy;
        y_inter[i] = sigma * 1e20;
    }

    unsigned int i = 0;
    for (map<double, double>::iterator p = data->begin(); p != data->end(); ++p) {
        x_data[i] = p->first;
        y_data[i] = p->second * 1e20;

        ++i;
    }

    TGraph* dgraph = new TGraph(data->size(), x_data, y_data);
    dgraph->SetMarkerColor(dataColor);
    dgraph->SetLineColor(dataColor);
    dgraph->SetMarkerStyle(20);
    dgraph->SetLineWidth(1);

    TGraph* graph = new TGraph(POINTS, x_inter, y_inter);
    graph->SetMarkerColor(interpolColor);
    graph->SetLineColor(interpolColor);
    graph->SetMarkerStyle(1);
    graph->SetLineWidth(1);

    mg.Add(graph);
    mg.Add(dgraph);
}
