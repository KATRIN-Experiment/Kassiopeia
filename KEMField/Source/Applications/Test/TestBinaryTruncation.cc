#include "KMD5HashGenerator.hh"

#include <cmath>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>

template<class T> std::string type_to_hex(const T arg)
{
    std::ostringstream hexstr;
    const char* addr = reinterpret_cast<const char*>(&arg);

    hexstr << "0x";
    hexstr << std::setw(2) << std::setprecision(2) << std::setfill('0') << std::hex;

    // if( isLittleEndian() )
    // {
    for (int b = sizeof(arg) - 1; b >= 0; b--) {
        hexstr << static_cast<unsigned>(*(addr + b) & 0xff);
    }
    // }
    // else
    // {
    //   for( int b = 0; b < sizeof(arg); b++ )
    //   {
    // 	hexstr << static_cast<unsigned>(*(addr+b) & 0xff) ;
    //   }
    // }
    return hexstr.str();
}

template<class T> std::string type_to_bin(const T arg)
{
    static std::string quads[16] = {"0000",
                                    "0001",
                                    "0010",
                                    "0011",
                                    "0100",
                                    "0101",
                                    "0110",
                                    "0111",
                                    "1000",
                                    "1001",
                                    "1010",
                                    "1011",
                                    "1100",
                                    "1101",
                                    "1110",
                                    "1111"};

    std::ostringstream hexstr;
    const char* addr = reinterpret_cast<const char*>(&arg);

    // if( isLittleEndian() )
    // {
    for (int b = sizeof(arg) - 1; b >= 0; b--) {
        std::stringstream s;
        s << std::setw(2) << std::setprecision(2) << std::setfill('0') << std::hex;
        s << static_cast<unsigned>(*(addr + b) & 0xff);

        char c = s.str()[0];

        if (c >= '0' && c <= '9')
            hexstr << quads[c - '0'];
        if (c >= 'A' && c <= 'F')
            hexstr << quads[10 + c - 'A'];
        if (c >= 'a' && c <= 'f')
            hexstr << quads[10 + c - 'a'];

        c = s.str()[1];

        if (c >= '0' && c <= '9')
            hexstr << quads[c - '0'];
        if (c >= 'A' && c <= 'F')
            hexstr << quads[10 + c - 'A'];
        if (c >= 'a' && c <= 'f')
            hexstr << quads[10 + c - 'a'];
    }
    // }
    // else
    // {
    //   for( int b = 0; b < sizeof(arg); b++ )
    //   {
    // 	hexstr << static_cast<unsigned>(*(addr+b) & 0xff) ;
    //   }
    // }
    return hexstr.str();
}

double Round(double x, unsigned int masked)
{
    static int one = 1;
    static int endian_min = (*(char*) &one == 1 ? 0 : sizeof(double) - 1);
    static int endian_max = (*(char*) &one == 1 ? sizeof(double) - 1 : 0);
    static int endian_dir = (*(char*) &one == 1 ? 1 : -1);

    double y = x;

    int maskedBits = masked % 8;
    int maskedBytes = masked / 8;

    // first we round
    if ((maskedBits != 0) || (maskedBytes != 0)) {
        unsigned short index, significantBit;
        if (maskedBits != 0) {
            index = endian_min + endian_dir * maskedBytes;
            significantBit = (unsigned short) (~(0xff << 1)) << (maskedBits - 1);
        }
        else {
            index = endian_min + endian_dir * (maskedBytes - 1);
            significantBit = 0x80;
        }

        if ((reinterpret_cast<unsigned char*>(&y)[index] & significantBit) != 0) {
            double roundOff = y;
            reinterpret_cast<unsigned char*>(&roundOff)[index] ^= significantBit;
            y += (y - roundOff);
        }
    }

    // then we mask the volatile bytes
    for (int i = endian_min; i != endian_max; i += endian_dir) {
        if (i == maskedBytes) {
            reinterpret_cast<unsigned char*>(&y)[i] &= (0xff << maskedBits);
            break;
        }
        reinterpret_cast<unsigned char*>(&y)[i] &= 0x00;
    }
    return y;
}

using namespace KEMField;

int main(int /*argc*/, char** /*argv*/)
{
    // double value = -1.;
    // for (unsigned int i=0;i<7;i++)
    // reinterpret_cast<unsigned char*>(&value)[i] = 0xff;
    // reinterpret_cast<unsigned char*>(&value)[2] = 0x0;

    double value = -1. / 3.;

    std::cout << "Value:" << std::endl;
    std::cout << value << std::endl;
    std::cout << type_to_hex(value) << std::endl;
    std::cout << type_to_bin(value) << std::endl;

    std::cout << "" << std::endl;

    for (unsigned int i = 0; i < 53; i++) {
        double value2 = Round(value, i);

        std::cout << "Rounding " << i << " bits" << std::endl;
        std::cout << value2 << std::endl;
        std::cout << type_to_hex(value2) << std::endl;
        std::cout << type_to_bin(value2) << std::endl;
        for (unsigned int j = 0; j < 64; j++) {
            if (j == 64 - i)
                std::cout << "^";
            else
                std::cout << " ";
        }

        std::cout << "" << std::endl;
    }

    return 0;
}
