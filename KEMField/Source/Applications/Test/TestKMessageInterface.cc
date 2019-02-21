#include <string>
#include <iostream>
#include <fstream>
#include <cstdio>

#include "KMessageInterface.hh"
#include "KEMCout.hh"

using namespace KEMField;

int main()
{
  std::cout<<"Testing KMessageInterface"<<std::endl;

  // kemstmsg<<"Hello world!"<<eom;

  KEMField::cout<<"Hello from KEMField!"<<std::endl;
  std::cout<<"Trying with my own endl"<<std::endl;
  KEMField::cout<<"Hello from KEMField again!"<<KEMField::endl;

  KDataDisplay<KMessage_KEMField> new_cout;

  new_cout<<"Hello from KMessage-enabled KEMField!"<<KEMField::endl;
}
