{
  // gSystem.Load("libgsl.so");
  gSystem.Load("libgsl");
  gSystem.Load("libfftw3");
  //  gSystem.Load("libCLHEP-2.0.4.2.so");
  gSystem.Load("libRIO.so");
  gSystem.Load("libNet.so");
  gSystem.Load("libHist.so");
  gSystem.Load("libGraf.so");
  gSystem.Load("libGraf3d.so");
  gSystem.Load("libTree.so");
  gSystem.Load("libRint.so");
  gSystem.Load("libPostscript.so");
  gSystem.Load("libMatrix.so");
  gSystem.Load("libPhysics.so");
  gSystem.Load("libMathCore.so");
  gSystem.Load("libThread");

  gSystem.Load("libKGeometry.so");
  gSystem.Load("libKBEM.so");
  gSystem.Load("libKRobinHood.so");
  gSystem.Load("libKIO.so");
  gSystem.Load("libKField.so");
  gSystem.Load("libKDirect.so");
  gSystem.Load("libKZHExpansion.so");
  gSystem.Load("libKSphericalMultipole.so");
  gSystem.Load("libKFFTM.so");

  THtml html;
//   html.SetSourceDir(".");

  html.SetOutputDir("./doc/html");
  html.SetAuthorTag("T.J. Corona");
  html.SetProductName("KEMField");
  html.MakeIndex();
  html.MakeAll();
}
