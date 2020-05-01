#include "KEMCout.hh"
#include "KMessageInterface.hh"

#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>

using namespace KEMField;

int main()
{
    std::cout << "Testing KMessageInterface" << std::endl;

    // kemstmsg<<"Hello world!"<<eom;

    KEMField::cout << "Hello from KEMField!" << std::endl;
    std::cout << "Trying with my own endl" << std::endl;
    KEMField::cout << "Hello from KEMField again!" << KEMField::endl;

    KDataDisplay<KMessage_KEMField> new_cout;

    new_cout << "Hello from KMessage-enabled KEMField!" << KEMField::endl;
}
