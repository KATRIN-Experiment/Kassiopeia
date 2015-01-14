
#include <string>
#include <iostream>
#include <iomanip>
#include <sstream>

#include <TMath.h>

#include "KGGateValve.hh"
#include "KThreeVector.h"

using namespace KGeoBag;

//Notes: 
/*
 * I have no idea what the metrics are here.
 * The functions defining test output need to be done properly.
 */

typedef enum {eHeader, eNormal, ePass, eWarning, eFail, eReturn} kjwSeverity;
static const std::string colorList[6]={"\033[95m",
                                       "\033[94m",
                                       "\033[92m",
                                       "\033[93m",
                                       "\033[91m",
                                       "\033[0m"};

void PrintPrettyColors(std::string message, Int_t color)
{ 
  for(Int_t i=0; i<color;i++) std::cout<<"    ";
  std::cout << std::setprecision(15) << colorList[color] << message 
    << colorList[eReturn] << std::endl; 
}


int main( )
{
  //Construct a shape
  //consider this a little more.
  PrintPrettyColors("Constructing Shape",eHeader);

  Double_t xyz_len[3]     = {398.e-3,700.9e-3,78.e-3};
  Double_t distFromBottomToOpening = 66.e-3;
  Double_t opening_rad    = 125.e-3;
  Double_t openingYoffset = -(xyz_len[1]*.5 -
            opening_rad - 
            distFromBottomToOpening);
  Double_t us_len         = 45.783e-3;
  Double_t ds_len         = 45.783e-3;
  Double_t center[3]      = {0.,-openingYoffset,195.e-3};

  KGGateValve* gV = new KGGateValve(center,
                                    xyz_len,
                                    openingYoffset,
                                    opening_rad,
                                    us_len,
                                    ds_len);

  if(gV)                                        PrintPrettyColors("Done", ePass);
  else                                          PrintPrettyColors("Failure", eWarning);

  // Testing a few things for the basics
  //----------------------TESTING CONTAINS POINT---------------------------------
  PrintPrettyColors("Testing Contains Point",eHeader);
  Double_t testPoint[3]={center[0],center[1],center[2]};//test point starts in the center

  PrintPrettyColors("Testing On Top Face: ", eNormal);
  testPoint[1] = center[1] + (xyz_len[1]*.25);//move test point onto Top Face
  if(gV->ContainsPoint(testPoint))              PrintPrettyColors("Pass", ePass);
  else                                          PrintPrettyColors("Fail", eFail);
  
  PrintPrettyColors("Testing Off Top Face: ", eNormal);
  testPoint[1]=center[1]+(xyz_len[1]*.5); //move test point above face
  if(!gV->ContainsPoint(testPoint))             PrintPrettyColors("Pass", ePass);
  else                                          PrintPrettyColors("Fail", eFail);
 
  PrintPrettyColors("Testing On Bottom Face: ", eNormal);
  testPoint[1]=center[1]-(xyz_len[1]*.25);//move test point onto bottom face
  if(gV->ContainsPoint(testPoint))              PrintPrettyColors("Pass", ePass);
  else                                          PrintPrettyColors("Fail", eFail); 

  PrintPrettyColors("Testing Off Bottom Face: ", eNormal);
  testPoint[1]=center[1]-(xyz_len[1]*.5);//move test point below face
  if(!gV->ContainsPoint(testPoint))             PrintPrettyColors("Pass", ePass);
  else                                          PrintPrettyColors("Fail", eFail); 
  
  PrintPrettyColors("Testing On Right Face: ", eNormal);
  testPoint[1] = 0.;
  testPoint[0] = center[0]+xyz_len[0]/4;
  if(gV->ContainsPoint(testPoint))              PrintPrettyColors("Pass", ePass);
  else                                          PrintPrettyColors("Fail", eFail);

  PrintPrettyColors("Testing Off Right Face: ", eNormal);
  testPoint[0] = testPoint[0]+(xyz_len[0]*.25);
  if(!gV->ContainsPoint(testPoint))             PrintPrettyColors("Pass", ePass);
  else                                          PrintPrettyColors("Fail", eFail);

  PrintPrettyColors("Testing On Left Face: ", eNormal);
  testPoint[0] = center[0]-(xyz_len[0]*.25);
  if(gV->ContainsPoint(testPoint))              PrintPrettyColors("Pass", ePass);
  else                                          PrintPrettyColors("Fail", eFail);

  PrintPrettyColors("Testing Off Left Face: ", eNormal);
  testPoint[0] = testPoint[0]-(xyz_len[0]*.25);
  if(!gV->ContainsPoint(testPoint))             PrintPrettyColors("Pass", ePass);
  else                                          PrintPrettyColors("Fail", eFail);

  PrintPrettyColors("Testing Downstream On Face: ", eNormal);
  testPoint[0] = center[0];
  testPoint[1] = center[1]+openingYoffset+opening_rad;
  testPoint[2] = center[2]-(xyz_len[2]*.25);
  if(gV->ContainsPoint(testPoint))              PrintPrettyColors("Pass", ePass);
  else                                          PrintPrettyColors("Fail", eFail);

  PrintPrettyColors("Testing Downstream Off Face: ", eNormal);
  testPoint[2] = center[2]-(xyz_len[2]*.5);
  if(!gV->ContainsPoint(testPoint))             PrintPrettyColors("Pass", ePass);
  else                                          PrintPrettyColors("Fail", eFail);


  PrintPrettyColors("Testing Upstream On Face: ", eNormal);
  testPoint[2] = center[2]+(xyz_len[2]*.25);
  if(gV->ContainsPoint(testPoint))              PrintPrettyColors("Pass", ePass);
  else                                          PrintPrettyColors("Fail", eFail);

  PrintPrettyColors("Testing Off Upstream Off Face: ", eNormal);
  testPoint[2] = center[2]+(xyz_len[2]*.5);
  if(!gV->ContainsPoint(testPoint))             PrintPrettyColors("Pass", ePass);
  else                                          PrintPrettyColors("Fail", eFail);

  PrintPrettyColors("Testing Upstream On Radius: ", eNormal);
  testPoint[0]=center[0];//move x coordinate into the middle.
  testPoint[1]=center[0];//move y-coordinate onto radius edge
  testPoint[2] = center[2]+(xyz_len[2]*.25)+us_len;
  if(gV->ContainsPoint(testPoint))              PrintPrettyColors("Pass", ePass);
  else                                          PrintPrettyColors("Fail", eFail);

  PrintPrettyColors("Testing Upstream Off Radius: ", eNormal);
  testPoint[2] = center[2]+(xyz_len[2]*.5)+us_len;
  if(!gV->ContainsPoint(testPoint))             PrintPrettyColors("Pass", ePass);
  else                                          PrintPrettyColors("Fail", eFail);

  PrintPrettyColors("Testing Downstream On Radius: ", eNormal);
  testPoint[2] = center[2]-(xyz_len[2]*.25)- ds_len;
  if(gV->ContainsPoint(testPoint))              PrintPrettyColors("Pass", ePass);
  else                                          PrintPrettyColors("Fail", eFail);

  PrintPrettyColors("Testing Downstream Off Radius: ", eNormal);
  testPoint[2] = center[2]-(xyz_len[2]*.5)-ds_len;
  if(!gV->ContainsPoint(testPoint))             PrintPrettyColors("Pass", ePass);
  else                                          PrintPrettyColors("Fail", eFail);
  //----------------------TESTING CONTAINS POINT---------------------------------

  //Test Distance To
    //Test a point off each side
  //Double_t DistanceTo(const Double_t* P,Double_t* P_in=NULL) const;
  //----------------------TESTING Distance TO___---------------------------------
  PrintPrettyColors("Testing Distance To",eHeader);
  Double_t result_point[3];
  Double_t  predicted_distance = .0 ;
  Double_t  result_distance = .0 ;
  std::stringstream output;

  for(Int_t i=0; i<3 ;i++)    testPoint[i]=center[i];

  PrintPrettyColors("Testing Distance Above: ", eNormal);
  predicted_distance = 1.e-4;
  testPoint[1] = center[1] + xyz_len[1]*.5 + predicted_distance;
  result_distance = (gV->DistanceTo(testPoint, result_point)); 
  if(predicted_distance < result_distance+0.001 && predicted_distance > result_distance-0.001)     PrintPrettyColors("Pass", ePass);  
  else                                                                                             PrintPrettyColors("Fail", eFail);
  output.str("");
  output <<"Distance: "<<(predicted_distance)<<", "<<(result_distance);
  PrintPrettyColors(output.str(), eWarning);

  PrintPrettyColors("Testing Distance Below: ", eNormal);
  predicted_distance = 1.e-4;
  testPoint[1] = center[1]-(xyz_len[1]*.5)-predicted_distance;
  result_distance = gV->DistanceTo(testPoint, result_point); 
  if(predicted_distance < result_distance+0.001 && predicted_distance > result_distance-0.001)     PrintPrettyColors("Pass", ePass);  
  else                                                                                             PrintPrettyColors("Fail", eFail);
  output.str("");
  output <<"Distance: "<<(predicted_distance)<<", "<<(result_distance);
  PrintPrettyColors(output.str(), eWarning);


  testPoint[1]=center[1];
  PrintPrettyColors("Testing Distance Right: ", eNormal);
  predicted_distance = 1.e-4;
  testPoint[0] = center[0]+(xyz_len[0]*.5)+predicted_distance;
  result_distance = gV->DistanceTo(testPoint, result_point); 
  if(predicted_distance < result_distance+0.001 && predicted_distance > result_distance-0.001)     PrintPrettyColors("Pass", ePass);  
  else                                                                                             PrintPrettyColors("Fail", eFail);
  output.str("");
  output <<"Distance: "<<(predicted_distance)<<", "<<(result_distance);
  PrintPrettyColors(output.str(), eWarning);


  PrintPrettyColors("Testing Distance Left: ", eNormal);
  predicted_distance = 1.e-3;
  testPoint[0] = center[0]-(xyz_len[0]*.5)-predicted_distance;
  result_distance = gV->DistanceTo(testPoint); 
  if(predicted_distance < result_distance+0.001 && predicted_distance > result_distance-0.001)     PrintPrettyColors("Pass", ePass);  
  else                                                                                             PrintPrettyColors("Fail", eFail);
  output.str("");
  output <<"Distance: "<<(predicted_distance)<<", "<<(result_distance);
  PrintPrettyColors(output.str(), eWarning);

  //test just off the radius  
  PrintPrettyColors("Testing Off Radius: ", eNormal);
  predicted_distance = 1.e-4;
  testPoint[0] = center[0];
  testPoint[1] = center[1] - xyz_len[1]*.5 + distFromBottomToOpening; //center[1] + openingYoffset + opening_rad + predicted_distance ;
  testPoint[2] = center[2] + (xyz_len[2]*.5) + (us_len) +predicted_distance;//put at the end of the opening c'est ne pas un pipe
  result_distance = gV->DistanceTo( testPoint, result_point ); 
  if(predicted_distance < result_distance+0.001 && predicted_distance > result_distance-0.001)     PrintPrettyColors("Pass", ePass);  
  else                                                                                             PrintPrettyColors("Fail", eFail);
  output.str("");
  output <<"Distance: "<<(predicted_distance)<<", "<<(result_distance);
  PrintPrettyColors(output.str(), eWarning);
  output.str("");
  for(Int_t i=0; i<3 ;i++) output<<" "<< std::setprecision(15)<<result_point[i];
    PrintPrettyColors(output.str(), eWarning);
  //----------------------TESTING Distance TO___---------------------------------

  //virtual void NearestNormal( const katrin::KThreeVector& aPoint, katrin::KThreeVector& aNormal ) const;
  //----------------------TESTING NearestNormal---------------------------------
  PrintPrettyColors("Testing NearestNormal", eHeader);
  katrin::KThreeVector testVector, testNormal;//test vector

  PrintPrettyColors("Testing Normal on top Surface",eNormal);
  testPoint[0] = center[0];
  testPoint[2] = center[2]; 
  testPoint[1] = center[1] +xyz_len[1]*.5+0.01;
  testVector.SetComponents(testPoint);
  gV->NearestNormal(testVector, testNormal);
  if(testNormal == katrin::KThreeVector(-1,0,0)) PrintPrettyColors("Pass", ePass);
  else                                           PrintPrettyColors("Fail", eFail);
  output.str("");
  for(Int_t i=0;i<3;i++) output<<" "<<testNormal.Components()[i];
  PrintPrettyColors(output.str(),eWarning);  


  PrintPrettyColors("Testing Normal on side Surface",eNormal);
  testPoint[1] = center[1];
  testPoint[2] = center[2]; 
  testPoint[0] = center[0] +xyz_len[0]*.5+0.01;
  testVector.SetComponents(testPoint);
  gV->NearestNormal(testVector, testNormal);
  if(testNormal == katrin::KThreeVector(0,-1,0)) PrintPrettyColors("Pass", ePass);
  else                                           PrintPrettyColors("Fail", eFail);
  output.str("");
  for(Int_t i=0;i<3;i++) output<<" "<<testNormal.Components()[i];
  PrintPrettyColors(output.str(),eWarning);

  PrintPrettyColors("Testing Normal on radius Surface",eNormal);
  testPoint[1] = center[1];
  testPoint[2] = center[2]; 
  testPoint[0] = center[0] +xyz_len[0]*.5+0.01;
  testVector.SetComponents(testPoint);
  gV->NearestNormal(testVector, testNormal);
  if(testNormal == katrin::KThreeVector(0,-1,0)) PrintPrettyColors("Pass", ePass);
  else                                           PrintPrettyColors("Fail", eFail);
  output.str("");
  for(Int_t i=0;i<3;i++) output<<" "<<testNormal.Components()[i];
  PrintPrettyColors(output.str(),eWarning);

  //----------------------TESTING NearestNormal---------------------------------


  delete gV;
}
