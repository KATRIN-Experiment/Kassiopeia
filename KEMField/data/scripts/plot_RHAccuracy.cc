#include "TAxis.h"
#include "TCanvas.h"
#include "TColor.h"
#include "TGraph.h"
#include "TLine.h"
#include "TMarker.h"
#include "TMultiGraph.h"
#include "TROOT.h"

#include <ctime>
#include <fstream>
#include <iostream>
#include <sstream>

std::vector<std::string> Tokenize(std::string separators, std::string input)
{
    UInt_t startToken = 0, endToken;  // Pointers to the token pos
    std::vector<std::string> tokens;  // Vector to keep the tokens
    UInt_t commentPos = input.size() + 1;

    if (separators.size() > 0 && input.size() > 0) {
        // Check for comment
        for (UInt_t i = 0; i < input.size(); i++) {
            if (input[i] == '#' || (i < input.size() - 1 && (input[i] == '/' && input[i + 1] == '/'))) {
                commentPos = i;
                break;
            }
        }

        while (startToken < input.size()) {
            // Find the start of token
            startToken = input.find_first_not_of(separators, startToken);

            // Stop parsing when comment symbol is reached
            if (startToken == commentPos) {
                if (tokens.size() == 0)
                    tokens.push_back("#");
                return tokens;
            }

            // If found...
            if (startToken != (UInt_t) std::string::npos) {
                // Find end of token
                endToken = input.find_first_of(separators, startToken);

                if (endToken == (UInt_t) std::string::npos)
                    // If there was no end of token, assign it to the end of string
                    endToken = input.size();

                // Extract token
                tokens.push_back(input.substr(startToken, endToken - startToken));

                // Update startToken
                startToken = endToken;
            }
        }
    }

    return tokens;
}

std::string ParseTimeDiff(Int_t start, Int_t end)
{
    std::stringstream ss;
    Int_t duration_s = end - start;
    Int_t duration_d = duration_s / 86400;
    duration_s -= duration_d * 86400;
    Int_t duration_h = duration_s / 3600;
    duration_s -= duration_h * 3600;
    Int_t duration_m = duration_s / 60;
    duration_s -= duration_m * 60;

    if (duration_d != 0)
        ss << duration_d << " d ";
    if (duration_d != 0 || duration_h != 0)
        ss << duration_h << " h ";
    if (duration_d != 0 || duration_h != 0 || duration_m != 0)
        ss << duration_m << " m ";
    ss << duration_s << " s\n";

    return ss.str();
}

void plot_RHAccuracy(std::string ws_name)
{

    // Clear global scope
    gROOT->Reset();

    std::stringstream fname_ss;
    fname_ss << "accuracy_" << ws_name << ".txt";

    std::ifstream in(fname_ss.str().c_str());

    if (!in) {
        std::cout << "No file!" << std::endl;
        return 1;
    }

    Int_t nGroups = 0;
    Int_t nIterations = 0;

    std::string dummy;
    std::vector<std::string> line;
    while (getline(in, dummy)) {
        line = Tokenize(" \t", dummy);
        nIterations++;
    }
    nGroups = line.size() - 2;

    in.close();
    in.open(fname_ss.str().c_str());

    std::vector<std::vector<Double_t>> accuracyMatrix(nGroups, std::vector<Double_t>(nIterations, 0));

    std::vector<Double_t> iterationArray(nIterations, 0);
    std::vector<Int_t> timeArray(nIterations, 0);

    for (Int_t i = 0; i < nIterations; i++) {
        in >> timeArray[i];
        in >> iterationArray[i];
        for (Int_t j = 0; j < nGroups; j++)
            in >> accuracyMatrix[j][i];
    }

    in.close();

    std::ofstream infoFile;
    std::stringstream infoFile_ss;
    infoFile_ss << "jobInfo_" << ws_name << ".txt";
    infoFile.open(infoFile_ss.str().c_str());
    const time_t* t = new time_t(timeArray[0]);
    infoFile << ctime(t);
    delete t;
    t = new time_t(timeArray[nIterations - 1]);
    infoFile << ctime(t);
    delete t;

    infoFile << ParseTimeDiff(timeArray[0], timeArray[nIterations - 1]);

    Float_t iterationsPerSec = 0;
    if (nIterations > 1)
        iterationsPerSec = (iterationArray[nIterations - 1] - iterationArray[nIterations - 2]) /
                           (timeArray[nIterations - 1] - timeArray[nIterations - 2]);

    infoFile << iterationsPerSec << "\n";

    infoFile << accuracyMatrix[0][nIterations - 1] << "\n";

    // infoFile.close();

    TCanvas* C = new TCanvas("C", "Canvas", 5, 5, 450, 450);
    C->SetBorderMode(0);
    C->SetFillColor(kWhite);
    gStyle->SetOptStat(0000000);
    gStyle->SetOptFit(0000);
    // gPad->SetLogx();
    gPad->SetLogy();

    TMultiGraph* mg = new TMultiGraph();

    for (Int_t j = nGroups - 1; j >= 0; j--) {
        TGraph* g = new TGraph(nIterations, &(iterationArray[0]), &(accuracyMatrix[j][0]));
        if (j == 0) {
            if (nIterations >= 100) {
                g->Fit("expo", "", "", iterationArray[nIterations - 100], iterationArray[nIterations - 1]);
                TF1* f = g->GetFunction("expo");
                TF1* f_inv = new TF1("f_inv", "1./[1]*(log(x)-[0])", 1.e-10, 10.);
                f_inv->SetParameter(0, f->GetParameter(0));
                f_inv->SetParameter(1, f->GetParameter(1));
                Int_t final_it = f_inv->Eval(1.e-4);
                Int_t it_ToGo = final_it - iterationArray[nIterations - 1];
                Int_t time_ToGo = it_ToGo / iterationsPerSec;
                infoFile << ParseTimeDiff(0, time_ToGo);
                t = new time_t(timeArray[nIterations - 1] + time_ToGo);
                infoFile << ctime(t);
                delete t;
            }
        }

        mg->Add(g);
    }

    mg->Draw("AL");
    std::stringstream title_ss;
    title_ss << "Convergence accuracy on " << ws_name;
    mg->SetTitle(title_ss.str().c_str());
    mg->GetXaxis()->SetTitle("Iteration number");
    mg->GetXaxis()->CenterTitle();
    // mg->GetXaxis()->SetLimits(4.e-9,1.1e-5);
    mg->GetXaxis()->SetTitleOffset(1.25);
    mg->GetYaxis()->SetTitle("Relative accuracy");
    mg->GetYaxis()->CenterTitle();
    mg->GetYaxis()->SetTitleOffset(1.25);
    mg->Draw("AL");

    std::stringstream saveName_ss;
    saveName_ss << "accuracy_" << ws_name << ".gif";
    C->SaveAs(saveName_ss.str().c_str());

    infoFile.close();

    gApplication->Terminate();
}
