{

    // Clear global scope
    gROOT->Reset();

    std::string fname = "results/sphere_cap.txt";
    Int_t N = 20;

    std::ifstream in(fname.c_str());

    if (!in) {
        std::cout << "No file!" << std::endl;
        return 1;
    }

    Double_t nTri;
    Double_t cap;
    Double_t time;

    Double_t nTris[N];
    Double_t caps[N];
    Double_t times[N];

    Int_t counter = 0;

    while (!in.eof()) {
        if (counter == N)
            break;
        in >> nTri;
        in >> cap;
        in >> time;
        nTris[counter] = nTri;
        caps[counter] = 1. - cap;
        times[counter] = time;
        counter++;
    }

    in.close();

    TCanvas* C = new TCanvas("C", "Canvas", 5, 5, 450, 450);
    C->SetBorderMode(0);
    C->SetFillColor(kWhite);
    gStyle->SetOptStat(0000000);
    gStyle->SetOptFit(0111);
    gPad->SetLogx();
    gPad->SetLogy();

    TGraph* cap_G = new TGraph(N, nTris, caps);
    cap_G->SetTitle("Capacitance of a Unit Sphere");
    cap_G->GetXaxis()->SetTitle("Number of triangles");
    cap_G->GetYaxis()->SetTitle("1 - (C_{computed}/C_{analytic})");
    cap_G->GetXaxis()->CenterTitle();
    cap_G->GetYaxis()->CenterTitle();
    // cap_G->GetYaxis()->SetTitleOffset(1.3);
    cap_G->SetMarkerStyle(20);
    TGraph* time_G = new TGraph(N, nTris, times);
    time_G->SetTitle("Unit Sphere Computation Time");
    time_G->GetXaxis()->SetTitle("Number of triangles");
    time_G->GetYaxis()->SetTitle("Time (s)");
    time_G->GetXaxis()->CenterTitle();
    time_G->GetYaxis()->CenterTitle();
    time_G->SetMarkerStyle(20);
    //   time_G->GetYaxis()->SetTitleOffset(1.3);

    // time_G->Fit("pol2");
    // time_G->GetFunction("pol2")->SetLineColor(kRed);
    // time_G->GetFunction("pol2")->SetLineWidth(.5);
    // time_G->Draw("AP");
    //   C->SaveAs("time_plot_tri.pdf");

    //   cap_G->GetYaxis()->SetLimits(1.e-6,1.e-1);
    cap_G->Draw("AP");

    //   cap_G->Draw("AC");

    //   C->SaveAs("capacitance_plot_2_time.pdf");
}
