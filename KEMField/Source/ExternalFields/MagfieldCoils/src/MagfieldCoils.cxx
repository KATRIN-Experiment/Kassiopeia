#include "MagfieldCoils.h"


MagfieldCoils::MagfieldCoils(string inputdirname, string inputobjectname, string inputcoilfilename, int inputNelliptic,
                             int inputnmax, double inputepstol)
{
    // In this constructor:
    //    a, coil parameters are read from file coilfilename; coil parameters are tested.
    //    b, symmetry groups are computed; coil and group parameters are written into file dirname+objectname+coil.txt
    //    c, source constants are computed, and written into data files defined by dirname+objectname
    fdirname = inputdirname;  // name of directory where the source coefficient files are stored
    fobjectname =
        inputobjectname;  // this string identifies the Magfield object (used as starting part of the source files)
    fcoilfilename = inputcoilfilename;  // name of coil input file
    fNelliptic =
        inputNelliptic;    // radial numerical integration parameter for magnetic field calc. with elliptic integrals
    fnmax = inputnmax;      // number of source coefficients for fixed source point: nmax+1
    fepstol = inputepstol;  // small tolerance parameter needed for the symmetry group definitions
                           // Recommended values: inputNelliptic=32, inputnmax=500, inputepstol=1.e-8.
                           //
    CoilRead();
    DynamicMemoryAllocations();
    frclimit = 0.99;
    CoilGroupWrite();
    //
    // The following functions compute all the source constants that are needed
    // for the remote, magnetic charge and central Legendre polynomial calculations.
    // The source constants are written to the
    //  data files dirname+objectname+magsource_remote.txt
    // and dirname+objectname+magsource_central.txt.
    // Remote and magnetic charge source constant calculations for all coils and groups:
    MagsourceRemoteCoils();
    MagsourceMagchargeCoils();
    MagsourceRemoteGroups();
    // Central source constant calculations for all coils and groups:
    MagsourceCentralCoils();
    MagsourceCentralGroups();
    //
}

///////////////////////////////////////////////////////////////////////

MagfieldCoils::MagfieldCoils(string inputdirname, string inputobjectname)
{
    // In this constructor:
    //    a, coil and group parameters are read from file dirname+objectname+coil.txt
    //    b, source constants are read from data files defined by dirname+objectname
    fdirname = inputdirname;  // name of directory where the source coefficient files are stored
    fobjectname =
        inputobjectname;  //  this string identifies the Magfield object (used as starting part of the source files)
                          //
    CoilGroupRead();
    MagsourceRead();
    DynamicMemoryAllocations();
    frclimit = 0.99;
}


/////////////////////////////////////

MagfieldCoils::~MagfieldCoils()
{
    // One-dim arrays:

    delete[] fC0;
    delete[] fC1;
    delete[] fc1;
    delete[] fc2;
    delete[] fc3;
    delete[] fc4;
    delete[] fc5;
    delete[] fc6;
    delete[] fc7;
    delete[] fc8;
    delete[] fc9;
    delete[] fc10;
    delete[] fc11;
    delete[] fc12;
    delete[] fP;
    delete[] fPp;
    delete[] fBzplus;
    delete[] fBrplus;
    delete[] fjlast;
    delete[] fjlastG;
    delete[] fG;
    delete[] fBrem1;
    delete[] fBcen1;
    delete[] frorem;
    delete[] froremG;
    delete[] fz0remG;


    // Two-dim arrays:
    for (int i = 0; i < fNcoil; i++) {
        delete[] fcoil[i];
        delete[] fBrem[i];
        delete[] fVrem[i];
        delete[] fz0cen[i];
        delete[] frocen[i];
    }
    delete[] fcoil;
    delete[] fz0cen;
    delete[] frocen;
    delete[] fBrem;
    delete[] fVrem;

    for (int g = 0; g < fNg; g++) {
        delete[] fCin[g];
        delete[] fLine[g];
        delete[] fZ[g];
        delete[] fBremG[g];
        delete[] fz0cenG[g];
        delete[] frocenG[g];
    }
    delete[] fCin;
    delete[] fLine;
    delete[] fZ;
    delete[] fBremG;
    delete[] fz0cenG;
    delete[] frocenG;

    // Three-dim arrays:
    for (int i = 0; i < fNcoil; i++) {
        for (int j = 0; j < fNsp[i]; j++)
            delete[] fBcen[i][j];
        delete[] fBcen[i];
    }
    delete[] fBcen;
    for (int g = 0; g < fNg; g++) {
        for (int j = 0; j < fNspG[g]; j++)
            delete[] fBcenG[g][j];
        delete[] fBcenG[g];
    }
    delete[] fBcenG;
    delete[] fNsp;
    delete[] fNspG;
    delete[] fNc;
}


///////////////////////////////////////////////////////////////////////////////////////

void MagfieldCoils::CoilRead()
{
    // This function reads the coil parameters (current and geometry)
    //  of a general 3-dimensional coil system,
    // from the data file defined by the string coilfilename.
    // The coils have rectangular shape cross section with various
    //  symmetry axes.
    // The data in the file defined by string coilfilename are:
    //     First line: number of coils  (fNcoil).
    //     Then there are fNcoil number of lines; each line contains:
    //       cu  Cx Cy Cz  alpha  beta  tu  L Rmin Rmax
    //    cu: current of coil (A)
    //    Cx:  x component of coil center (m)
    //    Cy:  y component of coil center (m)
    //    Cz:  z component of coil center (m)
    //    alpha, beta:  coil direction Euler angles in degrees.
    //    Coil direction unit vector (ux, uy, uz) is defined by
    //           ux=sin(beta)*sin(alpha),   uy=-sin(beta)*cos(alpha),   uz=cos(beta),
    //  (e.g. beta=0: u in+z direction;  beta=180:  u in -z direction)
    //    tu: number of coil turns (double)
    //    L: coil length (m)
    //    Rmin:  inner radius of coil   (m)
    //    Rmax:  outer radius of coil  (m).
    //  Positive current: current flow has right-handed screw with the (ux,uy,uz) coil direction vector
    //   (e.g. beta=0, cu positive: magnetic field in +z direction).
    // The fcoil parameters are written into the array fcoil[fNcoil][14].
    //   i: fcoil index (i=0,...,fNcoil-1)
    //   fcoil[i][0]=cu, fcoil[i][1]=Cx, fcoil[i][2]=Cy, fcoil[i][3]=Cz, fcoil[i][4]=alpha, fcoil[i][5]=beta,
    //   fcoil[i][6]=tu,  fcoil[i][7]=L, fcoil[i][8]=Rmin, fcoil[i][9]=Rmax.
    //  SI units are used here (A for cu, and m for Cx, Cy, Cz, L, Rmin, Rmax)!
    ifstream input;
    input.open(fcoilfilename.c_str());
    if (!input.is_open()) {
        puts("Message from  function CoilRead1 of class Magfield:");
        puts("Cannot open the  input source file!");
        puts("Program running is stopped !!! ");
        exit(1);
    }
    input >> fNcoil;
    if (fNcoil < 1) {
        puts("Message from function MagfieldCoils::CoilRead:");
        puts("fNcoil is not positive !");
        puts("Program running is stopped !!! ");
        exit(1);
    }
    // Dynamic memory allocation for the fcoil parameters:
    fcoil = new double*[fNcoil];
    for (int i = 0; i < fNcoil; i++) {
        fcoil[i] = new double[14];
    }
    // Reading the coil parameters:
    double cu, Cx, Cy, Cz, alpha, beta, tu, L, Rmin, Rmax;
    for (int i = 0; i < fNcoil; i++) {
        input >> cu >> Cx >> Cy >> Cz >> alpha >> beta >> tu >> L >> Rmin >> Rmax;
        // Coil parameter determination:
        fcoil[i][0] = cu;
        fcoil[i][1] = Cx;
        fcoil[i][2] = Cy;
        fcoil[i][3] = Cz;
        fcoil[i][4] = alpha;
        fcoil[i][5] = beta;
        fcoil[i][6] = tu;
        fcoil[i][7] = L;
        fcoil[i][8] = Rmin;
        fcoil[i][9] = Rmax;
        fcoil[i][10] = cu * tu / (L * (Rmax - Rmin));                        // current density
        fcoil[i][11] = sin(beta / 180. * M_PI) * sin(alpha / 180. * M_PI);   // coil direction unit vector comp. ux
        fcoil[i][12] = -sin(beta / 180. * M_PI) * cos(alpha / 180. * M_PI);  // coil direction unit vector comp. uy
        fcoil[i][13] = cos(beta / 180. * M_PI);                              // coil direction unit vector comp. uz
    }
    input.close();
    //
    // Test of coil parameters:
    for (int i = 0; i < fNcoil; i++) {
        if (fcoil[i][6] <= 1.e-9 || fcoil[i][7] <= 1.e-9 || fcoil[i][8] <= 1.e-9 || fcoil[i][9] < 1.e-9) {
            printf("Message from function MagfieldCoils::CoilRead1:\
             non-positive coil parameters: tu, Rmin, Rmax or L !!!\
             Program running is stopped !!! i= %9i \n",
                   i);
            exit(1);
        }
        if (fcoil[i][8] >= fcoil[i][9]) {
            printf("Message from function MagfieldCoils::CoilRead1:\
             Rmin>=Rmax!!!\
             Program running is stopped !!! i= %9i \n",
                   i);
            exit(1);
        }
    }

    //------------------------------------------------------
    //
    // Coil symmetry group calculations:
    //
    fG = new int[fNcoil];  // coil with index i is in symmetry group g=G[i]
    for (int i = 0; i < fNcoil; i++)
        fG[i] = -1;
    int g = -1;
    // Temporary arrays:
    int* Nctemp = new int[fNcoil];  // max. number of groups=fNcoil
    int** Cintemp = new int*[fNcoil];
    for (int i = 0; i < fNcoil; i++)
        Cintemp[i] = new int[fNcoil];  // max. number of coils in a group=fNcoil
    double** Linetemp = new double*[fNcoil];
    for (int i = 0; i < fNcoil; i++)
        Linetemp[i] = new double[6];
    double** Ztemp = new double*[fNcoil];
    for (int i = 0; i < fNcoil; i++)
        Ztemp[i] = new double[fNcoil];  // max. number of coils in a group=fNcoil
                                       // Starting the symmetry group calculation:
    for (int i = 0; i < fNcoil; i++) {
        int c;
        if (fG[i] == -1)  // starting a new symmetry group
        {
            g += 1;
            fNg = g + 1;
            Nctemp[g] = 1;
            // Coil i defines the symmetry axis of the new symmetry group.
            // The symmetry axis line is defined by the coil i center C[3] and the coil i direction unit vector u[3];
            // a point of the line: C[k]+z*u[k], k=0,1,2 (z: line point parameter).
            double C[3], u[3], alpha, beta;
            for (int k = 0; k < 3; k++)
                C[k] = fcoil[i][1 + k];  // coil center
            alpha = fcoil[i][4];
            beta = fcoil[i][5];
            u[0] = sin(beta / 180. * M_PI) * sin(alpha / 180. * M_PI);
            u[1] = -sin(beta / 180. * M_PI) * cos(alpha / 180. * M_PI);
            u[2] = cos(beta / 180. * M_PI);
            // Linetemp:
            for (int k = 0; k < 3; k++) {
                Linetemp[g][k] = C[k];
                Linetemp[g][3 + k] = u[k];
            }
            // First coil in the symmetry group g (with local coil index c=0): coil i.
            Cintemp[g][0] = i;
            // z coordinate of first coil (local coil index c=0) in symmetry group g: Z[g][0]=0.:
            Ztemp[g][0] = 0.;
            // If fNcoil=1: end of the i-loop
            if (fNcoil == 1)
                break;
            // Now we start to collect the coils which belong to this symmetry group:
            for (int ic = i + 1; ic < fNcoil; ic++)  // ic is here global coil index
            {
                double vecA[3], vecB[3], dA, dB, ZA, ZB, A[3], B[3], L, vec[3], uc[3], alphac, betac;
                // Coil ic end points: A[3], B[3]
                alphac = fcoil[ic][4];
                betac = fcoil[ic][5];
                uc[0] = sin(betac / 180. * M_PI) * sin(alphac / 180. * M_PI);
                uc[1] = -sin(betac / 180. * M_PI) * cos(alphac / 180. * M_PI);
                uc[2] = cos(betac / 180. * M_PI);
                L = fcoil[ic][7];
                ZA = ZB = 0.;
                for (int k = 0; k < 3; k++) {
                    A[k] = fcoil[ic][1 + k] - uc[k] * L / 2.;
                    B[k] = fcoil[ic][1 + k] + uc[k] * L / 2.;  // coil ic endpoints
                    ZA += (A[k] - C[k]) * u[k];
                    ZB += (B[k] - C[k]) * u[k];
                }
                for (int k = 0; k < 3; k++) {
                    vecA[k] = A[k] - C[k] - ZA * u[k];
                    vecB[k] = B[k] - C[k] - ZB * u[k];
                }
                dA = Mag(vecA);
                dB = Mag(vecB);
                if (dA < fepstol && dB < fepstol)  // coil ic is put to this symmetry group
                {
                    fG[ic] = g;
                    Nctemp[g] += 1;
                    c = Nctemp[g] - 1;
                    Cintemp[g][c] = ic;
                    Ztemp[g][c] = (ZA + ZB) / 2.;
                    // Coil shift:
                    for (int k = 0; k < 3; k++) {
                        A[k] = C[k] + ZA * u[k];
                        B[k] = C[k] + ZB * u[k];
                        vec[k] = B[k] - A[k];
                        fcoil[ic][1 + k] = (A[k] + B[k]) / 2.;
                    }
                    // alphac-> alpha, betac-> beta;
                    // If uc is opposite to u: change of current sign
                    for (int k = 0; k < 3; k++)
                        vec[k] = u[k] - uc[k];
                    if (Mag(vec) > 0.5)
                        fcoil[ic][0] *= -1.;
                    fcoil[ic][4] = alpha;
                    fcoil[ic][5] = beta;
                    fcoil[ic][11] =
                        sin(beta / 180. * M_PI) * sin(alpha / 180. * M_PI);  // coil direction unit vector comp. ux
                    fcoil[ic][12] =
                        -sin(beta / 180. * M_PI) * cos(alpha / 180. * M_PI);  // coil direction unit vector comp. uy
                    fcoil[ic][13] = cos(beta / 180. * M_PI);                   // coil direction unit vector comp. uz
                    // After that, all coils within a symmetry group have the same coil directions (alpha, beta).
                }  // end of epstol-if
            }      // end of symmetry group loop ic
        }          // end of symmetry group if
    }              // end of coil index loop i
                   // Nc, Cin, Line and Z array calculation:
    fNc = new int[fNg];
    for (int g = 0; g < fNg; g++)
        fNc[g] = Nctemp[g];  // number of coils in symmetry group g
    fCin = new int*[fNg];
    for (int g = 0; g < fNg; g++) {
        fCin[g] = new int[fNc[g]];
        for (int c = 0; c < fNc[g]; c++)
            fCin[g][c] = Cintemp[g][c];  // coil indices i in symmetry group g
    }
    fLine = new double*[fNg];
    for (int g = 0; g < fNg; g++) {
        fLine[g] = new double[6];
        for (int k = 0; k < 6; k++)
            fLine[g][k] = Linetemp[g][k];  // line point and line direction unit vector
    }
    fZ = new double*[fNg];
    for (int g = 0; g < fNg; g++) {
        fZ[g] = new double[fNc[g]];
        for (int c = 0; c < fNc[g]; c++)
            fZ[g][c] = Ztemp[g][c];  // local z coordinate of coil center in symmetry group g;  Z[g][0]=0.
    }
    // Delete Nctemp, Cintemp, Linetemp, Ztemp:
    delete[] Nctemp;
    for (int i = 0; i < fNcoil; i++)
        delete[] Cintemp[i];
    delete[] Cintemp;
    for (int i = 0; i < fNcoil; i++)
        delete[] Linetemp[i];
    delete[] Linetemp;
    for (int i = 0; i < fNcoil; i++)
        delete[] Ztemp[i];
    delete[] Ztemp;

    return;
}

///////////////////////////////////////////////////////////////////////

void MagfieldCoils::CoilGroupWrite()
{
    // This function writes the coil and symmetry group parameters into the data file dirname+objectname+coilgroup.txt.
    string filename = fdirname + fobjectname + "coilgroup.txt";
    ofstream output;
    output.precision(16);
    output.open(filename.c_str());
    output << setw(15) << fNcoil << setw(15) << fNg << endl;  // number of coils and groups
    output << setw(15) << fNelliptic << setw(15) << fnmax << endl;
    for (int i = 0; i < fNcoil; i++) {
        output << setw(10) << i << endl;
        output << scientific << setw(26) << fcoil[i][0] << setw(26) << fcoil[i][1] << setw(26) << fcoil[i][2] << setw(26)
               << fcoil[i][3] << endl;
        output << scientific << setw(26) << fcoil[i][4] << setw(26) << fcoil[i][5] << setw(26) << fcoil[i][6] << endl;
        output << scientific << setw(26) << fcoil[i][7] << setw(26) << fcoil[i][8] << setw(26) << fcoil[i][9] << setw(26)
               << fcoil[i][10] << endl;
        output << scientific << setw(26) << fcoil[i][11] << setw(26) << fcoil[i][12] << setw(26) << fcoil[i][13] << endl;
    }
    for (int g = 0; g < fNg; g++) {
        output << setw(12) << g << setw(12) << fNc[g] << endl;  // number of coils in symmetry group g
        //  Cin[g][c]: // global coil index corresponding to local coil index c in group g
        // Z[g][c]: local z coordinate of coil center in symmetry group g; c=0,..., Nc[g]-1;   Z[g][0]=0
        // Symmetry axis line point:  Line[g][0], Line[g][1], Line[g][2];  line direction unit vector:  Line[g][3], Line[g][4], Line[g][5];
        output << scientific << setw(26) << fLine[g][0] << setw(26) << fLine[g][1] << setw(26) << fLine[g][2] << endl;
        output << scientific << setw(26) << fLine[g][3] << setw(26) << fLine[g][4] << setw(26) << fLine[g][5] << endl;
        for (int c = 0; c < fNc[g]; c++)
            output << setw(12) << fCin[g][c] << scientific << setw(26) << fZ[g][c] << endl;
    }
    output.close();
    return;
}


/////////////////////////////////////////////////////////////////////////////////////


void MagfieldCoils::CoilGroupRead()
{
    // This function reads the coil parameters from the data file dirname+objectname+coilgroup.txt.
    string filename = fdirname + fobjectname + "coilgroup.txt";
    ifstream input;
    input.open(filename.c_str());
    if (!input.is_open()) {
        puts("Message from  function CoilRead2 of class Magfield:");
        puts("Cannot open the  coil.txt file!");
        puts("Program running is stopped !!! ");
        exit(1);
    }
    // We read the coil parameters:
    input >> fNcoil >> fNg;
    if (fNcoil < 1 || fNg < 1) {
        printf("Message from function MagfieldCoils::CoilRead2:\
            fNcoil<1 or Ng<1 !!!\
            Computation is  stopped !!! fNcoil, Ng= %9i %9i  \n",
               fNcoil,
               fNg);
        exit(1);
    }
    input >> fNelliptic >> fnmax;
    // Dynamic memory allocation for the coil and symmetry group parameters:
    fcoil = new double*[fNcoil];
    for (int i = 0; i < fNcoil; i++) {
        fcoil[i] = new double[14];
    }
    // We read the coil parameters:
    int ix;
    for (int i = 0; i < fNcoil; i++) {
        input >> ix;
        input >> fcoil[i][0] >> fcoil[i][1] >> fcoil[i][2] >> fcoil[i][3];
        input >> fcoil[i][4] >> fcoil[i][5] >> fcoil[i][6];
        input >> fcoil[i][7] >> fcoil[i][8] >> fcoil[i][9] >> fcoil[i][10];
        input >> fcoil[i][11] >> fcoil[i][12] >> fcoil[i][13];
    }
    // Dynamic memory allocation for the symmetry group arrays:
    fNc = new int[fNg];
    fLine = new double*[fNg];
    fCin = new int*[fNg];
    fZ = new double*[fNg];
    // We read the symmetry group parameters:
    int gx;
    for (int g = 0; g < fNg; g++) {
        input >> gx >> fNc[g];
        //  Cin[g][c]: // global coil index corresponding to local coil index c in group g
        // Z[g][c]: local z coordinate of coil center in symmetry group g; c=0,..., Nc[g]-1;   Z[g][0]=0
        // Symmetry axis line point:  Line[g][0], Line[g][1], Line[g][2];  line direction unit vector:  Line[g][3], Line[g][4], Line[g][5];
        fLine[g] = new double[6];
        input >> fLine[g][0] >> fLine[g][1] >> fLine[g][2];
        input >> fLine[g][3] >> fLine[g][4] >> fLine[g][5];
        fCin[g] = new int[fNc[g]];
        fZ[g] = new double[fNc[g]];
        for (int c = 0; c < fNc[g]; c++)
            input >> fCin[g][c] >> fZ[g][c];
    }


    input.close();

    fG = new int[fNcoil];
    fBrem1 = new double[fnmax + 1];
    fBcen1 = new double[fnmax + 1];

    return;
}


/////////////////////////////////////////////////////////////////////////////////////

void MagfieldCoils::MagsourceRead()
{
    // This function reads the source constants  from the data files
    //    dirname+objectname+magsource_central.txt and dirname+objectname+magsource_remote.txt.
    // Part 1: reading the remote source constants:
    string filename = fdirname + fobjectname + "magsource_remote.txt";
    ifstream input;
    input.open(filename.c_str());
    if (!input.is_open()) {
        puts("Message from  function  MagfieldCoils::MagsourceRead:");
        puts("Cannot open the  input source file magsource_remote.txt!");
        puts("Program running is stopped !!! ");
        exit(1);
    }
    // Dynamic memory allocations (rorem, Brem):
    frorem = new double[fNcoil];
    fBrem = new double*[fNcoil];
    for (int i = 0; i < fNcoil; i++)
        fBrem[i] = new double[fnmax + 1];
    // Coil index loop:
    int ix, nx;
    for (int i = 0; i < fNcoil; i++) {
        input >> ix >> frorem[i];
        for (int n = 2; n <= fnmax; n++)
            input >> nx >> fBrem[i][n];
    }
    // Dynamic memory allocations (Vrem):
    fVrem = new double*[fNcoil];
    for (int i = 0; i < fNcoil; i++)
        fVrem[i] = new double[fnmax + 1];
    for (int i = 0; i < fNcoil; i++) {
        input >> ix;
        for (int n = 0; n <= fnmax; n++)
            input >> nx >> fVrem[i][n];
    }
    // Dynamic memory allocations (z0remG, roremG, BremG):
    fz0remG = new double[fNg];
    froremG = new double[fNg];
    fBremG = new double*[fNg];
    for (int g = 0; g < fNg; g++)
        fBremG[g] = new double[fnmax + 1];
    // Group index loop:
    int gx, Ncg;
    for (int g = 0; g < fNg; g++) {
        if (fNc[g] == 1)
            continue;
        input >> gx >> Ncg >> fz0remG[g] >> froremG[g];
        for (int n = 2; n <= fnmax; n++)
            input >> nx >> fBremG[g][n];
    }
    input.close();
    //
    // Part 2: reading the central source coefficients:
    filename = fdirname + fobjectname + "magsource_central.txt";
    input.open(filename.c_str());
    if (!input.is_open()) {
        puts("Message from  function  MagfieldCoils::MagsourceRead:");
        puts("Cannot open the  input source file magsource_central.txt!");
        puts("Program running is stopped !!! ");
        exit(1);
    }
    int jx;
    // Dynamic memory allocations (Nsp, z0cen, rocen, Bcen):
    fNsp = new int[fNcoil];
    fz0cen = new double*[fNcoil];
    frocen = new double*[fNcoil];
    fBcen = new double**[fNcoil];
    // Coil index loop:
    string text;
    //  input >> text;
    for (int i = 0; i < fNcoil; i++) {
        input >> ix >> fNsp[i];
        // Source point loop:
        fz0cen[i] = new double[fNsp[i]];
        frocen[i] = new double[fNsp[i]];
        fBcen[i] = new double*[fNsp[i]];
        for (int j = 0; j < fNsp[i]; j++) {
            input >> jx >> fz0cen[i][j] >> frocen[i][j];
            fBcen[i][j] = new double[fnmax + 1];
            for (int n = 0; n <= fnmax; n++)
                input >> nx >> fBcen[i][j][n];
        }
    }
    // Dynamic memory allocations (NspG, z0cenG, rocenG, BcenG):
    fNspG = new int[fNg];
    fz0cenG = new double*[fNg];
    frocenG = new double*[fNg];
    fBcenG = new double**[fNg];
    // Group index loop:
    //  input >> text;
    for (int g = 0; g < fNg; g++) {
        if (fNc[g] == 1)  // no group calculation if the group has only 1 coil
        {
            fz0cenG[g] = new double[1];
            frocenG[g] = new double[1];
            fNspG[g] = 1;
            fBcenG[g] = new double*[1];
            fBcenG[g][0] = new double[fnmax + 1];
            continue;
        }
        input >> gx >> fNspG[g];
        // Source point loop:
        fz0cenG[g] = new double[fNspG[g]];
        frocenG[g] = new double[fNspG[g]];
        fBcenG[g] = new double*[fNspG[g]];
        for (int j = 0; j < fNspG[g]; j++) {
            input >> jx >> fz0cenG[g][j] >> frocenG[g][j];
            fBcenG[g][j] = new double[fnmax + 1];
            for (int n = 0; n <= fnmax; n++)
                input >> nx >> fBcenG[g][j][n];
        }
    }
    input.close();
}


/////////////////////////////////////////////////////////////////////////////////////

void MagfieldCoils::DynamicMemoryAllocations()
{
    // This function makes dynamic memory allocations for the variables
    //  C0, C1, c0 to c12, P, Pp, Bzplus, Brplus.
    // It also computes the values C0[n], C1[n], c0[n] -- c12[n] (n=2,...,nmax).
    fC0 = new double[fnmax + 1];
    fC1 = new double[fnmax + 1];
    fc1 = new double[fnmax + 1];
    fc2 = new double[fnmax + 1];
    fc3 = new double[fnmax + 1];
    fc4 = new double[fnmax + 1];
    fc5 = new double[fnmax + 1];
    fc6 = new double[fnmax + 1];
    fc7 = new double[fnmax + 1];
    fc8 = new double[fnmax + 1];
    fc9 = new double[fnmax + 1];
    fc10 = new double[fnmax + 1];
    fc11 = new double[fnmax + 1];
    fc12 = new double[fnmax + 1];

    fC0[1] = 1.;
    for (int n = 2; n <= fnmax; n++) {
        fC0[n] = 1. / (1. * n);
        fC1[n] = 1. / (1. * (n - 1.));
        fc1[n] = (2. * n - 1.) / (1. * n);
        fc2[n] = (n - 1.) / (1. * n);
        fc3[n] = (2. * n - 1.) / (1. * (n - 1.));
        fc4[n] = (1. * n) / (1. * (n - 1.));
        fc5[n] = (1.) / (n * 1.);
        fc6[n] = (1.) / (n + 1.);
        if (n >= 4) {
            int m = n - 2;
            double M = (m + 1.) * (m + 2.) * (2. * m - 1.);
            double Mp = (m) * (m + 1.) * (2. * m - 1.);
            double A = (2. * m - 1.) * (2. * m + 1.) * (2. * m + 3.);
            double Ap = A;
            double B = 2. * m * m * (2. * m + 3.) - 1.;
            double Bp = 2. * m * (m + 2.) * (2. * m - 1.) - 3.;
            double C = m * (m - 1.) * (2. * m + 3.);
            double Cp = m * (m + 1.) * (2. * m + 3.);
            fc7[n] = A / M;
            fc8[n] = B / M;
            fc9[n] = C / M;
            fc10[n] = Ap / Mp;
            fc11[n] = Bp / Mp;
            fc12[n] = Cp / Mp;
        }
    }
    // Dynamic memory for P, Pp, Bzplus, Brplus, jlast, jlastG:
    fP = new double[fnmax + 2];
    fPp = new double[fnmax + 2];
    fBzplus = new double[fnmax + 1];
    fBrplus = new double[fnmax + 1];
    //
    fjlast = new int[fNcoil];
    for (int i = 0; i < fNcoil; i++)
        fjlast[i] = -2;
    fjlastG = new int[fNg];
    for (int g = 0; g < fNg; g++)
        fjlastG[g] = -2;
}

/////////////////////////////////////////////////////


//  #include "MagfieldElliptic.cc"

////////////////////////////////////////////////////////////////////////////////////////////

//  Magnetic field calculations with elliptic integrals

//////////////////////////////////////////////////////////////////////////////////////////


void MagfieldCoils::Magfield2EllipticCoil(int i, double z, double r, double& Bz, double& Br)
{
    // This function computes the magnetic field components Bz and Br
    // of an axially symmetric coil index i, with z axis as symmetry axis,
    // at a fieldpoint with (z,r) cylindrical coordinates relative to coil center and direction, using
    // the first, second and third complete elliptic integrals.
    double L, Zmin, Zmax, Rmin, Rmax, sigma;
    double R, Z, delr2, sumr2, delz2, eta, d, K, EK, PIK, S;
    double sign, delRr, Rlow[2], Rhigh[2];
    const double mu0 = 4. * M_PI * 1.e-7;
    double x[2][1001], w[2][1001];  // nodes and weights
                                    // Coil parameters:
    L = fcoil[i][7];                 // coil length
    Zmin = -L / 2.;
    Zmax = L / 2.;  // coil endpoints relative to coil center
    Rmin = fcoil[i][8];
    Rmax = fcoil[i][9];    // coil inner and outer radii
    sigma = fcoil[i][10];  // current density
                          // Field point  test:
    if (fabs(z - Zmin) < 1.e-8 && r >= Rmin - (Rmax - Rmin) * 1.e-8 && r <= Rmax + (Rmax - Rmin) * 1.e-8)
        z = Zmin - 1.e-8;
    if (fabs(z - Zmax) < 1.e-8 && r >= Rmin - (Rmax - Rmin) * 1.e-8 && r <= Rmax + (Rmax - Rmin) * 1.e-8)
        z = Zmax + 1.e-8;
    // Radial numerical integration node definition:
    int N[2];
    N[0] = N[1] = fNelliptic;
    //
    // R-integration limits:
    int M;
    if (z > Zmin && z < Zmax && r > Rmin && r < Rmax)  // field point inside the coil winding: integral in 2 steps
    {
        M = 2;
        Rlow[0] = Rmin;
        Rhigh[0] = r - (r - Rmin) * 1.e-12;
        Rlow[1] = r + (Rmax - r) * 1.e-12;
        Rhigh[1] = Rmax;
        GaussLegendreIntegration(N[0], Rlow[0], Rhigh[0], x[0], w[0]);
        GaussLegendreIntegration(N[1], Rlow[1], Rhigh[1], x[1], w[1]);
    }
    else {
        M = 1;
        Rlow[0] = Rmin;
        Rhigh[0] = Rmax;
        GaussLegendreIntegration(N[0], Rlow[0], Rhigh[0], x[0], w[0]);
    }
    // Integration:
    Bz = Br = 0.;
    for (int m = 0; m < M; m++)  // integral step m-loop
    {
        for (int iR = 0; iR < N[m]; iR++)  // radius R-loop
        {
            R = x[m][iR];
            double Bzhat = 0.;
            double Brhat = 0.;
            for (int iZ = 0; iZ < 2; iZ++)  // Z-loop
            {
                if (iZ == 0) {
                    Z = Zmax;
                    sign = 1.;
                }
                else {
                    Z = Zmin;
                    sign = -1.;
                }
                delr2 = (r - R) * (r - R);
                delz2 = (z - Z) * (z - Z);
                sumr2 = (r + R) * (r + R);
                d = delr2 / sumr2;
                eta = (delr2 + delz2) / (sumr2 + delz2);
                S = sqrt(sumr2 + delz2);
                K = RF_Carlson(0., eta, 1.);
                EK = -1. / 3. * RD_Carlson(0., eta, 1.);
                delRr = R - r;
                if (d < 1.e-18) {
                    d = 1.e-18;
                    if (R > r)
                        delRr = (r + R) * 1.e-9;
                    else if (R < r)
                        delRr = -(r + R) * 1.e-9;
                }
                PIK = 1. / 3. * RJ_Carlson(0., eta, 1., d);
                Bzhat += sign * R * (z - Z) / (S * (R + r)) * (K + delRr / (2. * R) * PIK * (1. - d));
                Brhat += sign * R / S * (2. * EK + K);
            }  // end of Z-loop
            Bz += w[m][iR] * Bzhat;
            Br += w[m][iR] * Brhat;
            //	cout.precision(4);
            //	cout << scientific << m << "   " << iR << "      " << x[m][iR] << "     " << w[m][iR] << endl;
        }  // end of radius R-loop
    }      // end of integral step m-loop
    double C = -mu0 / M_PI * sigma;
    Bz *= C;
    Br *= C;
    return;
}

///////////////////////////////////////////////////

void MagfieldCoils::MagfieldEllipticCoil(int i, const double* P, double* B)
{
    // This function computes the magnetic field components B[0],B[1],B[2]
    // at a field point P[0],P[1],P[2], due to the 3-dimensional coil
    // with index i, using the first, second and third complete elliptic
    // integrals.  SI units are used (P[k] in m, B[k] in T, k=0,1,2)!
    // The coil is defined by the coil parameters coil[i][j], j=0,...,10.
    double z, r, Bz, Br, u[3], Ploc[3], Pr[3], w[3], C[3];
    // We define a local coordinate system:
    //    origo at the coil center, local z axis parallel to coil axis, in u vector direction.
    // Coil direction u and center C:
    u[0] = fcoil[i][11];
    u[1] = fcoil[i][12];
    u[2] = fcoil[i][13];
    for (int k = 0; k <= 2; k++)
        C[k] = fcoil[i][1 + k];  // coil center
                                // Local z and r coordinates of the field point P:
    for (int k = 0; k <= 2; k++)
        Ploc[k] = P[k] - C[k];
    z = Ploc[1] * u[1] + Ploc[2] * u[2] + Ploc[0] * u[0];
    for (int k = 0; k <= 2; k++)
        Pr[k] = Ploc[k] - z * u[k];
    r = sqrt(Pr[1] * Pr[1] + Pr[2] * Pr[2] + Pr[0] * Pr[0]);
    // Bz and Br calculation:
    Magfield2EllipticCoil(i, z, r, Bz, Br);
    // B[k] calculation from Bz, Br:
    if (r < 1.e-12)
        for (int k = 0; k <= 2; k++)
            B[k] = Bz * u[k];
    else {
        for (int k = 0; k <= 2; k++)
            w[k] = Pr[k] / r;
        for (int k = 0; k <= 2; k++)
            B[k] = Bz * u[k] + Br * w[k];
    }
    return;
}

///////////////////////////////////////////////////////////////////////////////

void MagfieldCoils::MagfieldElliptic(const double* P, double* B)
{
    // This function computes the magnetic field components B[0],B[1],B[2]
    // at a field point P[0],P[1],P[2], due to all coils, using elliptic integrals.
    // SI units are used (P[k] in m, B[k] in T, k=0,1,2 --> x, y, z components)!
    double Bi[3];
    for (int k = 0; k <= 2; k++)
        B[k] = 0.;
    for (int i = 0; i < fNcoil; i++) {
        MagfieldEllipticCoil(i, P, Bi);
        for (int k = 0; k <= 2; k++)
            B[k] += Bi[k];
    }
    return;
}


//  #include "Magfield.cc"

///////////////////////////////////////////////////////////////////////////////////////////

bool MagfieldCoils::Magfield(const double* P, double* B)
{
    // This function computes the magnetic field components B[0],B[1],B[2]
    // at a field point P[0],P[1],P[2], due to all symmetry groups (coils),
    // using central or remote zonal harmonic (Legendre polynomial) expansion, or elliptic integrals.
    // SI units are used (P[k] in m, B[k] in T, k=0,1,2 --> x,y,z components)!
    // The return value is true, if only zonal harmonic expansions are used for the magnetic field calc.;
    //   if elliptic integral calc. is used (even for 1 coil): the return value is false.
    double Bgroup[3];
    bool magfield = true;
    for (int k = 0; k < 3; k++)
        B[k] = 0.;
    for (int g = 0; g < fNg; g++) {
        bool magfieldgroup = MagfieldGroup(g, P, Bgroup);
        if (magfieldgroup == false)
            magfield = false;
        for (int k = 0; k < 3; k++)
            B[k] += Bgroup[k];
    }
    return magfield;
}

///////////////////////////////////////////////////////////////////////////////////////


//  #include "MagsourceRemote.cc"

// Remote convergence radius calculation:

double MagfieldCoils::Funrorem(int i, double z0)
// This function computes the remote convergence radius
// Funrorem=rorem for coil i, at the coil axis source point z0
// (defined relative to the coil center; positive z0 in coil direction u, which is determined by alpha and beta).
// rorem = maximum distance of the axis point z0 from the coil winding (outer corner points).
{
    double L, Rmax, Zmin, Zmax;
    L = fcoil[i][7];
    Rmax = fcoil[i][9];
    Zmin = -L / 2.;
    Zmax = L / 2.;  // coil endpoint Z values defined relative to the coil center
    if (z0 > 0.)
        return sqrt((z0 - Zmin) * (z0 - Zmin) + Rmax * Rmax);
    else
        return sqrt((z0 - Zmax) * (z0 - Zmax) + Rmax * Rmax);
}

/////////////////////////////////////////////////////

// Remote source coefficient calculation for 1 coil:
//  (local axisymmetric case)

void MagfieldCoils::Magsource2RemoteCoil(int i, double z0, double rorem)
// This subroutine computes the magnetic field remote source constants Brem1[n]
//  (n=2,...,nmax) of coil i, at the local coil axis source point z0
// (defined relative to the coil center; positive z0 in coil direction u),
//   with remote convergence radius rorem as input parameter.
// Radial integration number: N; this depends on the radial thickness:
//  N is small if the(Rmax-Rmin)/Rmin ratio is small, and it is large
//   if this ratio is large.
{
    const double mu0 = 4. * M_PI * 1.e-7;
    double x[1001], w[1001];  // nodes and weights
    double L, sigma, Zmin, Zmax, Rmin, Rmax;
    // Coil parameters:
    L = fcoil[i][7];  // coil length
    Zmin = -L / 2.;
    Zmax = L / 2.;  // coil endpoints relative to coil center
    Rmin = fcoil[i][8];
    Rmax = fcoil[i][9];    // coil inner and outer radii
    sigma = fcoil[i][10];  // current density
                          // Radial integration number N:
    int N;
    double ratio = (Rmax - Rmin) / Rmin;
    if (ratio < 0.2)
        N = 32;
    else
        N = 32 * ratio / 0.2;
    if (N > 1000)
        N = 1000;
    // Initialization of Brem1[n] and Pp[n]:
    for (int n = 2; n <= fnmax; n++)
        fBrem1[n] = 0.;
    fPp[0] = 0.;
    fPp[1] = 1.;
    // C:
    double C = mu0 * sigma / (2. * rorem * rorem);
    // Radial integration nodes and weights:
    GaussLegendreIntegration(N, Rmin, Rmax, x, w);
    // Zminmax, sign:
    double Zminmax[2] = {Zmin, Zmax};
    int sign[2] = {-1, 1};
    // Gauss-Legendre integration loop:
    for (int integ = 0; integ < N; integ++) {
        double R = x[integ];
        double R2 = R * R;
        for (int iz = 0; iz <= 1; iz++) {
            double Z = Zminmax[iz];
            double ros = sqrt(R2 + (Z - z0) * (Z - z0));
            double us = (Z - z0) / ros;
            double a = ros / rorem;
            double b = a;
            double d = C * R2 * sign[iz] * w[integ];
            for (int n = 2; n <= fnmax; n++) {
                double usPp = us * fPp[n - 1];
                fPp[n] = 2. * usPp - fPp[n - 2] + (usPp - fPp[n - 2]) * fC1[n];
                fBrem1[n] += d * fc6[n] * b * fPp[n];
                b *= a;
            }
        }
    }
    return;
}

//////////////////////////////////////////////////////////////////////////////////////

// Remote source constant calculation for all coils:
//  (3-dim. case)

void MagfieldCoils::MagsourceRemoteCoils()
// This function computes the Brem[i][n] (i=0,...,fNcoil-1; n=2,...,nmax)
//   remote source constants for all coils.
// Number of remote source points for a fixed coil is 1 (coil center).
// The results are written into the data file  dirname+objectname+magsource_remote.txt:
//     fNcoil: number of coils,
//     nmax: number of source coefficients for a fixed coil is nmax+1 (n=0,...,nmax),
//     i: coil index,
//     rorem[i]: remote convergence radius for coil i.
{
    // Output to file dirname+objectname+magsource_remote.txt :
    string filename = fdirname + fobjectname + "magsource_remote.txt";
    ofstream output;
    output.precision(16);
    output.open(filename.c_str());
    // Dynamic memory allocations (Brem1, rorem, Brem):
    fBrem1 = new double[fnmax + 1];
    frorem = new double[fNcoil];
    fBrem = new double*[fNcoil];
    for (int i = 0; i < fNcoil; i++)
        fBrem[i] = new double[fnmax + 1];
    // Coil index loop:
    for (int i = 0; i < fNcoil; i++) {
        // Brem1[n] calc.:
        frorem[i] = Funrorem(i, 0.);
        Magsource2RemoteCoil(i, 0., frorem[i]);
        // Output to file dirname+objectname+magsource_remote.txt  (rorem, Brem1[n])
        output << scientific << setw(7) << i << setw(28) << frorem[i] << endl;
        for (int n = 2; n <= fnmax; n++) {
            if (fabs(fBrem1[n]) < 1.e-30)
                fBrem1[n] = 0.;
            if (n % 2 == 1)
                output << scientific << setprecision(1) << setw(15) << n << setw(28) << fBrem1[n] << endl;
            else
                output << scientific << setprecision(16) << setw(15) << n << setw(28) << fBrem1[n] << endl;
            fBrem[i][n] = fBrem1[n];
        }
    }
    output.close();
    return;
}

//////////////////////////////////////////////////////////////////////////////////////

// Remote source constant calculation for all symmetry groups:

void MagfieldCoils::MagsourceRemoteGroups()
// This function computes the BremG[g][n] (g=0,...,Nc[g]-1; n=2,...,nmax)
//   remote source constants for those symmetry groups which have more than 1 coil.
// Number of remote source points for a fixed group is 1 (group center).
// The results are written into the data file  dirname+objectname+magsource_remote.txt:
//     Ng: number of symmetry groups,
//     nmax: number of source constants for a fixed group is nmax+1 (n=0,...,nmax),
//     g: group index, c: local coil index, i: global coil index.
//     z0remG[g]: local z value of the source point for group g in the Z coord. system (group center),
//     roremG[g]: remote convergence radius for group g at the remote source point z0remG[g].
{
    // Output to file dirname+objectname+magsource_remote.txt :
    string filename = fdirname + fobjectname + "magsource_remote.txt";
    ofstream output;
    output.precision(16);
    output.open(filename.c_str(), ios::app);
    // Dynamic memory allocations (z0remG, roremG, BremG):
    fz0remG = new double[fNg];
    froremG = new double[fNg];
    fBremG = new double*[fNg];
    for (int g = 0; g < fNg; g++)
        fBremG[g] = new double[fnmax + 1];
    // Group index loop:
    for (int g = 0; g < fNg; g++) {
        if (fNc[g] == 1)  // no group calculation if the group has only 1 coil
            continue;
        // z0remG[g], roremG[g] calculation:
        RemoteSourcepointGroup(g);
        // Output to file dirname+objectname+magsource_remote.txt  (g, Nc[g], )
        output << scientific << setw(10) << g << setw(10) << fNc[g] << setw(26) << fz0remG[g] << setw(26) << froremG[g]
               << endl;
        // BremG[g][n] calculation by local coil index loop:
        for (int n = 2; n <= fnmax; n++)
            fBremG[g][n] = 0.;
        // Local coil index loop:
        for (int c = 0; c < fNc[g]; c++)  // c: local coil index in group
        {
            int i = fCin[g][c];
            Magsource2RemoteCoil(i, fz0remG[g] - fZ[g][c], froremG[g]);
            for (int n = 2; n <= fnmax; n++)
                fBremG[g][n] += fBrem1[n];
        }
        for (int n = 2; n <= fnmax; n++) {
            if (fabs(fBremG[g][n]) < 1.e-30)
                fBremG[g][n] = 0.;
            output << scientific << setw(9) << n << setw(26) << fBremG[g][n] << endl;
        }
    }
    output.close();
    return;
}

/////////////////////////////////////////////////////////////////////

void MagfieldCoils::RemoteSourcepointGroup(int g)
// This function computes the remote source point z0remG[g] and remote
// convergence radius roremG[g] for group g.
//     z0remG[g]: local z value of the source point for group g in the Z coord. system (group center),
//     roremG[g]: remote convergence radius for group g at the remote source point z0remG[g].
{
    // We define a local Z-coordinate system:
    //    origo at Z[g][0], local z axis parallel to group symmetry axis, in u vector direction.
    // Remote source point of group --> z0remG[g] (in group Z-coordinate system):
    double z0rem, zmin = 1.e9, zmax = -1.e9;
    for (int c = 0; c < fNc[g]; c++)  // c: local coil index in group
    {
        int i = fCin[g][c];
        double L = fcoil[i][7];
        double zA = fZ[g][c] - L / 2.;
        double zB = fZ[g][c] + L / 2.;  // coil edges
        if (zA < zmin)
            zmin = zA;
        if (zB > zmax)
            zmax = zB;
    }
    z0rem = fz0remG[g] = (zmin + zmax) / 2.;  // group center in group Z-coordinate system
                                             // Remote convergence radius of group --> roremG[g] :
    double rorem = 0.;
    for (int c = 0; c < fNc[g]; c++)  // c: local coil index in group
    {
        int i = fCin[g][c];
        double roremc = Funrorem(i, z0rem - fZ[g][c]);  // remote source point relative to coil center
        if (roremc > rorem)
            rorem = roremc;
    }
    froremG[g] = rorem;
}


//  #include "MagsourceMagcharge.cc"

/////////////////////////////////////////////////////

// Magnetic charge remote source constant calculation for all coils

void MagfieldCoils::MagsourceMagchargeCoils()
// This function computes the Vrem[i][n] (i=0,...,fNcoil-1; n=0,...,nmax)
//   magnetic charge remote source constants for all coils.
// The results are written into the data file  dirname+objectname+magsource_remote.txt:
//     fNcoil: number of coils,
//     nmax: number of source coefficients for a fixed coil is nmax+1 (n=0,...,nmax),
//     i: coil index, n source constant index.
{
    // Output to file dirname+objectname+magsource_remote.txt :
    double Rmin, Rmax, sigma;
    string filename = fdirname + fobjectname + "magsource_remote.txt";
    ofstream output;
    output.precision(16);
    output.open(filename.c_str(), ios::app);
    // Dynamic memory allocations (Vrem):
    fVrem = new double*[fNcoil];
    for (int i = 0; i < fNcoil; i++)
        fVrem[i] = new double[fnmax + 1];
    // Coil index loop:
    for (int i = 0; i < fNcoil; i++) {
        // Coil i parameters:
        Rmin = fcoil[i][8];
        Rmax = fcoil[i][9];                             // inner and outer radius
        sigma = fcoil[i][10];                           // current density
        output << scientific << setw(7) << i << endl;  // coil index
        fP[0] = 1.;
        double ratio = Rmin / Rmax;
        double ration3 = ratio * ratio * ratio;
        for (int n = 0; n <= fnmax; n++) {
            if (n > 0 && n % 2 == 0)  // even positive n
                fP[n] = -(n - 1.) / double(n) * fP[n - 2];
            if (n % 2 == 1)
                fVrem[i][n] = 0.;
            else
                fVrem[i][n] = sigma * fP[n] * Rmax * Rmax / (2. * (n + 2.) * (n + 3.)) * (1. - ration3);
            if (n % 2 == 1)
                output << scientific << setprecision(1) << setw(15) << n << setw(28) << fVrem[i][n] << endl;
            else
                output << scientific << setprecision(16) << setw(15) << n << setw(28) << fVrem[i][n] << endl;
            if (ration3 < 1.e-20)
                ration3 = 0.;
            else
                ration3 *= ratio;
        }
    }
    output.close();
    return;
}


//  #include "MagsourceCentral.cc"

// Central convergence radius calculation:

double MagfieldCoils::Funrocen(int i, double z0)
// This function computes the effective central convergence radius
// Funrocen=rocen for coil i, at the axis source point z0
// (defined relative to the coil center; positive z0 in coil direction u).
// rocen = minimum distance of the axis point z0 from the coil inner corner points
//  (Zmin,Rmin) and (Zmax,Rmin).
{
    double L, Zmin, Zmax, Rmin;
    L = fcoil[i][7];
    Rmin = fcoil[i][8];
    Zmin = -L / 2.;
    Zmax = L / 2.;  // coil endpoint Z values defined relative to the coil center
    if (z0 < 0.)
        return sqrt((z0 - Zmin) * (z0 - Zmin) + Rmin * Rmin);
    else
        return sqrt((z0 - Zmax) * (z0 - Zmax) + Rmin * Rmin);
}

/////////////////////////////////////////////////////

double MagfieldCoils::FunrocenG(int g, double z0)
// This function computes the central convergence radius
// FunrocenG=rocenmin for the symmetry group g, at the axis source point z0
// (defined in the group Z coordinate system).
// rocenmin = minimum distance of the axis point z0 from  the windings of the group coils.
{
    double rocenmin = 1.e9;
    for (int c = 0; c < fNc[g]; c++)  // c: local coil index in group
    {
        int i = fCin[g][c];  // i: global coil index
        double L, Zmin, Zmax, Rmin, Z0;
        L = fcoil[i][7];
        Rmin = fcoil[i][8];
        Zmin = -L / 2.;
        Zmax = L / 2.;      // coil endpoint Z values defined relative to the coil center
        Z0 = z0 - fZ[g][c];  // source point relative to coil center
        double rocen;
        if (Z0 <= Zmin)
            rocen = sqrt((Z0 - Zmin) * (Z0 - Zmin) + Rmin * Rmin);
        else if (Z0 >= Zmax)
            rocen = sqrt((Z0 - Zmax) * (Z0 - Zmax) + Rmin * Rmin);
        else
            rocen = Rmin;
        if (rocen < rocenmin)
            rocenmin = rocen;
    }
    return rocenmin;
}

/////////////////////////////////////////////////////

// Central source constant calculation for 1 coil:
//  (local axisymmetric case)

void MagfieldCoils::Magsource2CentralCoil(int i, double z0, double Rocen)
// This function computes the magnetic field central source constants Bcen1[n]
//  (n=0,...,nmax) of coil i, at the axis source point z0
// (defined relative to the coil center; positive z0 in coil direction u),
//   with central convergence radius Rocen as input parameter.
// Indices: integ: numerical integration;  n: source constants,
//   iz=0: Z=Zmin; iz=1: Z=Zmax.
// Radial integration number: N; this depends on the radial thickness:
//  N is small if the(Rmax-Rmin)/Rmin ratio is small, and it is large
//   if this ratio is large.
{
    const double mu0 = 4. * M_PI * 1.e-7;
    double x[1001], w[1001];  // nodes and weights
    double L, sigma, Zmin, Zmax, Rmin, Rmax;
    // Coil parameters:
    L = fcoil[i][7];
    Zmin = -L / 2.;
    Zmax = L / 2.;  // coil endpoints relative to coil center
    Rmin = fcoil[i][8];
    Rmax = fcoil[i][9];
    sigma = fcoil[i][10];  // current density
                          // Radial integration number N:
    int N;
    double ratio = (Rmax - Rmin) / Rmin;
    int Nmin = 32;
    if (ratio < 0.2)
        N = Nmin;
    else
        N = Nmin * ratio / 0.2;
    if (N > 1000)
        N = 1000;
    // Initialization of Bcen1[n] and Pp[n]:
    for (int n = 0; n <= fnmax; n++)
        fBcen1[n] = 0.;
    fPp[0] = 0.;
    fPp[1] = 1.;
    // C:
    double C = -0.5 * mu0 * sigma * Rocen;
    // Radial integration nodes and weights:
    GaussLegendreIntegration(N, Rmin, Rmax, x, w);
    // Zminmax, sign:
    double Zminmax[2] = {Zmin, Zmax};
    const int sign[2] = {-1, 1};
    // Gauss-Legendre integration loop:
    for (int integ = 0; integ < N; integ++) {
        double R = x[integ];
        double R2 = R * R;
        for (int iz = 0; iz <= 1; iz++) {
            double Z = Zminmax[iz];
            double ros = sqrt(R2 + (Z - z0) * (Z - z0));
            double us = (Z - z0) / ros;
            double a = Rocen / ros;
            double b = a;
            double d = 1. / (ros * ros * ros);
            double D = C * R2 * sign[iz] * d * w[integ];
            fBcen1[0] += 0.5 * mu0 * sigma * sign[iz] * us * w[integ];
            fBcen1[1] += D;
            for (int n = 2; n <= fnmax; n++) {
                double usPp = us * fPp[n - 1];
                fPp[n] = 2. * usPp - fPp[n - 2] + (usPp - fPp[n - 2]) * fC1[n];
                fBcen1[n] += D * fC0[n] * b * fPp[n];
                b *= a;
            }
        }
    }
    return;
}

//////////////////////////////////////////////////////////////////////////////////////

// Central source coonstant calculation for all coils:
//  (3-dim. case)

void MagfieldCoils::MagsourceCentralCoils()
// This function computes the central source constants for all coils.
// The results are written into the data file dirname+objectname+magsource_central.txt:
//     fNcoil: number of coils,
//     nmax: number of source coefficients for a fixed coil and
//              source point is nmax+1 (n=0,...,nmax),
//     i: coil index,
//     Nsp[i]: number of source points for coil i,
//     j: source point index for a fixed coil with index i,
//     z0: local z value of the source point, for a fixed coil with index i,
//   (defined relative to the coil center; positive z0 in coil direction u),
//     rocen: central convergence radius for a fixed source point and coil.
{
    // Output to file dirname+objectname+magsource_central.txt :
    string filename = fdirname + fobjectname + "magsource_central.txt";
    ofstream output;
    output.precision(16);
    output.open(filename.c_str());
    // Dynamic memory allocations (Bcen1, Nsp, z0cen, rocen, Bcen):
    fBcen1 = new double[fnmax + 1];
    fNsp = new int[fNcoil];
    fz0cen = new double*[fNcoil];
    frocen = new double*[fNcoil];
    fBcen = new double**[fNcoil];
    // Coil index loop:
    //    output << "coils" << endl;
    for (int i = 0; i < fNcoil; i++) {
        CentralSourcepointsCoil(i);  // central source point calculation
        output << setw(9) << i << setw(9) << fNsp[i] << endl;
        // Central source constant calculation (source point loop):
        fBcen[i] = new double*[fNsp[i]];
        // Source point loop:
        //     cout << "i, Nsp[i]=     " << i << "      " << Nsp[i] << endl;
        for (int j = 0; j < fNsp[i]; j++) {
            Magsource2CentralCoil(i, fz0cen[i][j], frocen[i][j]);
            // Output to file dirname+objectname+magsource_central.txt  (j, z0, rocen, Bcen1[n])
            output << scientific << setw(9) << j << setw(26) << fz0cen[i][j] << setw(26) << frocen[i][j] << endl;
            fBcen[i][j] = new double[fnmax + 1];
            for (int n = 0; n <= fnmax; n++) {
                if (fabs(fBcen1[n]) < 1.e-30)
                    fBcen1[n] = 0.;
                output << scientific << setw(9) << n << setw(28) << fBcen1[n] << endl;
                fBcen[i][j][n] = fBcen1[n];
            }
        }  // end of source point loop
    }      // end of coil index loop
    output.close();
    return;
}

/////////////////////////////////////////////////////////////////////

void MagfieldCoils::CentralSourcepointsCoil(int i)
// This function computes the central source points z0cen[i][j] and remote
// convergence radii rocen[i][j] for coil i.
{
    vector<double> Z0cen;
    Z0cen.push_back(0.);
    double z0;
    double roremC = frorem[i];
    double z0max = 2.5 * roremC;
    int nsp = 0;  // number of central source points on positive z side
    int k = 0;
    do {
        k += 1;
        double rocen1 = Funrocen(i, Z0cen[k - 1]);
        double del1 = rocen1 / 4.;
        z0 = Z0cen[k - 1] + del1;
        rocen1 = Funrocen(i, z0);
        double del2 = rocen1 / 4.;
        double del = FMIN(del1, del2);
        z0 = Z0cen[k - 1] + del;
        Z0cen.push_back(z0);
        if (z0 < z0max)
            nsp += 1;
    } while (z0 < z0max);
    // Number of central source points for coil i:
    fNsp[i] = 2 * nsp + 1;
    // Central source points and conv. radii for coil i (source point loop):
    fz0cen[i] = new double[fNsp[i]];
    frocen[i] = new double[fNsp[i]];
    //
    int j = nsp;
    z0 = fz0cen[i][j] = 0.;
    frocen[i][j] = Funrocen(i, z0);
    for (int k = 1; k <= nsp; k++) {
        j = nsp + k;
        z0 = fz0cen[i][j] = Z0cen[k];
        frocen[i][j] = Funrocen(i, z0);
        j = nsp - k;
        z0 = fz0cen[i][j] = -Z0cen[k];
        frocen[i][j] = Funrocen(i, z0);
    }
    Z0cen.clear();
}

///////////////////////////////////////////////////////////////////////

// Central source constant calculation for all groups:

void MagfieldCoils::MagsourceCentralGroups()
// This function computes the central source constants for all symmetry groups that have more than 1 coil.
// The results are written into the data file dirname+objectname+magsource_central.txt:
//     Ng: number of symmetry groups,
//     nmax: number of source constants for a fixed group and
//              source point is nmax+1 (n=0,...,nmax),
//     g: group index, c: local coil index, i: global coil index.
//     NspG[g]: number of source points for group g,
//     j: central source point index for a fixed group,
{
    // Output to file dirname+objectname+magsource_central.txt :
    string filename = fdirname + fobjectname + "magsource_central.txt";
    ofstream output;
    output.precision(16);
    output.open(filename.c_str(), ios::app);
    // Dynamic memory allocations (NspG, z0cenG, rocenG, BcenG):
    fNspG = new int[fNg];
    fz0cenG = new double*[fNg];
    frocenG = new double*[fNg];
    fBcenG = new double**[fNg];
    // Group index loop:
    //  output << "groups" << endl;
    for (int g = 0; g < fNg; g++) {
        if (fNc[g] == 1)  // no group calculation if the group has only 1 coil
        {
            fz0cenG[g] = new double[1];
            frocenG[g] = new double[1];
            fNspG[g] = 1;
            fBcenG[g] = new double*[1];
            fBcenG[g][0] = new double[fnmax + 1];
            continue;
        }
        CentralSourcepointsGroup(g);  // central source point calculation for group g
                                      // Number of central source points for group g:  NspG[g]
        output << setw(9) << g << setw(9) << fNspG[g] << endl;
        // Central source constant calculation (source point loop):
        fBcenG[g] = new double*[fNspG[g]];
        for (int j = 0; j < fNspG[g]; j++) {
            // Output to file dirname+objectname+magsource_central.txt  (j, z0, rocen)
            output << scientific << setw(9) << j << setw(28) << fz0cenG[g][j] << setw(28) << frocenG[g][j] << endl;
            // BcenG[g][j][n] initialization:
            fBcenG[g][j] = new double[fnmax + 1];
            for (int n = 0; n <= fnmax; n++)
                fBcenG[g][j][n] = 0.;
            // Local coil index loop:
            for (int c = 0; c < fNc[g]; c++)  // c: local coil index in group g
            {
                int i = fCin[g][c];                                               // global coil index
                Magsource2CentralCoil(i, fz0cenG[g][j] - fZ[g][c], frocenG[g][j]);  // source point relative to coil center
                for (int n = 0; n <= fnmax; n++)
                    fBcenG[g][j][n] += fBcen1[n];
            }
            // Output to file dirname+objectname+magsource_central.txt  of BcenG[g][j][n]:
            for (int n = 0; n <= fnmax; n++) {
                if (fabs(fBcenG[g][j][n]) < 1.e-30)
                    fBcenG[g][j][n] = 0.;
                output << scientific << setw(9) << n << setw(28) << fBcenG[g][j][n] << endl;
            }
        }  // end of source point j loop
    }      // end of group index g loop
    output.close();
    return;
}

/////////////////////////////////////////////////////////////////////

void MagfieldCoils::CentralSourcepointsGroup(int g)
// This function computes the central source points z0cenG[g][j] (in group Z-system) and central
// convergence radii rocenG[g][j] for group g.
// Central source points:
{
    vector<double> Z0cen;
    double z0;
    double roremC = froremG[g];
    double z0min = fz0remG[g] - 2. * roremC;
    double z0max = fz0remG[g] + 2. * roremC;
    Z0cen.push_back(z0min);  // j=0
    int nsp = 1;             // number of central source points in group g
    int j = 0;               // --> z0min
    do {
        j += 1;
        double rocen1 = FunrocenG(g, Z0cen[j - 1]);
        double del1 = rocen1 / 4.;
        z0 = Z0cen[j - 1] + del1;
        rocen1 = FunrocenG(g, z0);
        double del2 = rocen1 / 4.;
        double del = FMIN(del1, del2);
        z0 = Z0cen[j - 1] + del;
        Z0cen.push_back(z0);
        if (z0 < z0max)
            nsp += 1;
    } while (z0 < z0max);
    // Number of central source points for group g:
    fNspG[g] = nsp;
    // Central source points and conv. radii for group g (source point loop):
    fz0cenG[g] = new double[nsp];
    frocenG[g] = new double[nsp];
    //
    for (int j = 0; j < nsp; j++) {
        z0 = fz0cenG[g][j] = Z0cen[j];
        frocenG[g][j] = FunrocenG(g, z0);
    }
}


//  #include "MagfieldRemote.cc"

/////////////////////////////////////////////////////

// Magnetic field calculation with remote zonal harmonic expansion
//  (local axisymmetric case)

bool MagfieldCoils::Magfield2Remote(bool bcoil, int ig, double z, double r, double& Bz, double& Br, double& rc)
// This function computes the magnetic field components Bz and Br at fieldpoint z and r by remote expansion,
//  either for a coil or for a symmetry group.
// It computes also the remote convergence ratio  rc=Rorem/ro.
// bcoil=true: coil calculation;  bcoil=false: symmetry group calculation.
// ig: coil index for bcoil=true, and group index for bcoil=false.
// bcoil=true: Brem[ig][n], n=2,...,nmax  remote source constants for coils are used.
// bcoil=false: BremG[ig][n], n=2,...,nmax  remote source constantsfor groups are used.
// z: for a coil, it is relative to the coil center (z=0);
//       for a group: z is defined in the local group Z-coordinate system.
//  nmax: maximal index of the source constants (maximum of n).
//  SI units are used !
// If the convergence of the Legendre-series expansion is too slow:
//     rc>rclimit,  and the return value is false
//  (in this case one should use another computation method !!!)
//
// z0, Rorem, ro, u, sr, rc, rcn:
{
    double z0, Rorem;  // source point and conv. radius
    if (bcoil == true) {
        z0 = 0.;
        Rorem = frorem[ig];
    }
    else {
        z0 = fz0remG[ig];
        Rorem = froremG[ig];
    }
    double delz = z - z0;
    double ro = sqrt(r * r + delz * delz);
    if (ro < 1.e-9)  // field point is very close to the source point --> rc>1 !
        ro = 1.e-9;
    double u = delz / ro;
    double sr = r / ro;
    rc = Rorem / ro;  // convergence ratio
    double rcn = rc * rc;
    // If rc>rclimit: the Legendre polynomial series is too slow, or not convergent (for rc>1) !!!
    if (rc > frclimit) {
        return false;
    }
    // First 2 terms of Legendre polynomial P and its derivative Pp (P-primed)
    fP[0] = 1.;
    fP[1] = u;
    fPp[0] = 0.;
    fPp[1] = 1.;
    //
    bool even = false;   // true:  only the even n terms are used
    if (bcoil == false)  // group calculation
    {
        double sum = 0., sumodd = 0.;
        for (int n = 2; n <= 12; n++)
            sum += fabs(fBremG[ig][n]);
        for (int n = 3; n <= 12; n += 2)
            sumodd += fabs(fBremG[ig][n]);
        if (sumodd < sum * 1.e-8)
            even = true;
    }
    // We start here the series expansion:
    Bz = Br = 0.;
    int nlast = 0;
    double brem;
    if (bcoil == true || even == true)  // even calculation
    {
        // Only the even terms are needed in the series
        //   (because for the middle source point in the coil centre
        //     Brem[ig][n]=0 for odd n)
        for (int n = 2; n <= 4; n++) {
            if (bcoil == true)
                brem = fBrem[ig][n];
            else
                brem = fBremG[ig][n];
            //        P[n]=c1[n]*u*P[n-1]-c2[n]*P[n-2];
            //        Pp[n]=c3[n]*u*Pp[n-1]-c4[n]*Pp[n-2];
            double uPp = u * fPp[n - 1];
            double uP = u * fP[n - 1];
            fP[n] = 2. * uP - fP[n - 2] - (uP - fP[n - 2]) * fC0[n];
            fPp[n] = 2. * uPp - fPp[n - 2] + (uPp - fPp[n - 2]) * fC1[n];
            rcn = rcn * rc;
            fBzplus[n] = brem * rcn * fP[n];
            fBrplus[n] = sr * brem * fc5[n] * rcn * fPp[n];
            Bz += fBzplus[n];
            Br += fBrplus[n];
        }
        double u2 = u * u;
        double rc2 = rc * rc;
        for (int n = 6; n <= fnmax - 1; n += 2) {
            if (bcoil == true)
                brem = fBrem[ig][n];
            else
                brem = fBremG[ig][n];
            nlast = n;
            fP[n] = (fc7[n] * u2 - fc8[n]) * fP[n - 2] - fc9[n] * fP[n - 4];
            fPp[n] = (fc10[n] * u2 - fc11[n]) * fPp[n - 2] - fc12[n] * fPp[n - 4];
            rcn = rcn * rc2;
            fBzplus[n] = brem * rcn * fP[n];
            fBrplus[n] = sr * brem * fc5[n] * rcn * fPp[n];
            Bz += fBzplus[n];
            Br += fBrplus[n];
            double Beps = 1.e-15 * (fabs(Bz) + fabs(Br));
            double Bdelta = fabs(fBzplus[n]) + fabs(fBrplus[n]) + fabs(fBzplus[n - 2]) + fabs(fBrplus[n - 2]);
            if (Bdelta < Beps || Bdelta < 1.e-20)
                break;
        }
    }
    else  // even + odd group calculation: both even and odd terms are needed in the series
    {
        for (int n = 2; n <= fnmax - 1; n++) {
            brem = fBremG[ig][n];
            nlast = n;
            fP[n] = fc1[n] * u * fP[n - 1] - fc2[n] * fP[n - 2];
            fPp[n] = fc3[n] * u * fPp[n - 1] - fc4[n] * fPp[n - 2];
            rcn = rcn * rc;
            fBzplus[n] = brem * rcn * fP[n];
            fBrplus[n] = sr * brem * fc5[n] * rcn * fPp[n];
            Bz += fBzplus[n];
            Br += fBrplus[n];
            if (n > 5) {
                double Beps = 1.e-15 * (fabs(Bz) + fabs(Br));
                double Bdelta = fabs(fBzplus[n]) + fabs(fBrplus[n]) + fabs(fBzplus[n - 1]) + fabs(fBrplus[n - 1]) +
                                fabs(fBzplus[n - 2]) + fabs(fBrplus[n - 2]) + fabs(fBzplus[n - 3]) + fabs(fBrplus[n - 3]);
                if (Bdelta < Beps || Bdelta < 1.e-20)
                    break;
            }
        }
    }
    if (nlast >= fnmax - 2)
        return false;
    else
        return true;
}


/////////////////////////////////////////////////


//  #include "MagfieldMagcharge.cc"

/////////////////////////////////////////////////////

// Magnetic field calculation with remote magnetic charge method
//  (local axisymmetric case)

bool MagfieldCoils::Magfield2Magcharge(int i, double z, double r, double& Bz, double& Br, double& rc)
{
    // This function computes the magnetic field components Bz and Br at fieldpoint z and r by remote
    // magnetic charge expansion, for coil i.
    // It computes also the remote convergence ratio rc.
    // The Vrem[i][n], n=0,...,nmax  magnetic charge remote source constants for coil i are used.
    // z: it is relative to the coil center (z=0).
    //  nmax: maximal index of the source constants (maximum of n).
    //  SI units are used !
    // If the convergence of the Legendre-series expansion is too slow:
    //     rc>rclimit,   and the return value is false
    //  (in this case one should use another computation method !!!)
    //
    const double mu0 = 4. * M_PI * 1.e-7;
    double Hzmax, Hrmax, Hzmin, Hrmin, Hz, Hr, Mz, rcmin, rcmax;
    double L, Zmin, Zmax, Rmin, Rmax, sigma;
    L = fcoil[i][7];
    Rmin = fcoil[i][8];
    Rmax = fcoil[i][9];    // coil length, inner and outer radius
    sigma = fcoil[i][10];  // current density
    Zmin = -L / 2.;
    Zmax = L / 2.;  // coil endpoints relative to coil center
    bool hfieldmax = Hfield(i, z - Zmax, r, Hzmax, Hrmax, rcmax);
    bool hfieldmin = Hfield(i, z - Zmin, r, Hzmin, Hrmin, rcmin);
    rc = FMAX(rcmin, rcmax);
    // If rc>rclimit: the Legendre polynomial series is too slow, or not convergent (for rc>1) !!!
    if (hfieldmax == false || hfieldmin == false) {
        return false;
    }
    Hz = Hzmax - Hzmin;
    Hr = Hrmax - Hrmin;
    if (z >= Zmin && z <= Zmax && r <= Rmin)
        Mz = sigma * (Rmax - Rmin);
    else if (z >= Zmin && z <= Zmax && r > Rmin && r <= Rmax)
        Mz = sigma * (Rmax - r);
    else
        Mz = 0.;
    Bz = mu0 * (Mz + Hz);
    Br = mu0 * Hr;
    //    cout.precision(15);
    //    cout << scientific  <<z << "     " << r <<"     " << mu0*Mz << endl;
    //    cout<< scientific <<  "Hzmax,Hzmin=" << '\t' << Hzmax  <<'\t' << Hzmin<<  endl;
    return true;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

bool MagfieldCoils::Hfield(int i, double z, double r, double& Hz, double& Hr, double& rc)
{
    // Computes the H field components Hz and Hr
    // at field point (z,r) of a disc end of coil i, by special
    // remote zonal harmonic expansion.
    // z is relative to the disc position (disc is at z=0).
    // It computes also the remote convergence ratio  rc=rorem/ro.
    // Source point is at disk position (z=0)-> only even n mag. charge source coefficients are used.
    // If the convergence of the Legendre-series expansion is too slow:
    //     rc>rclimit,   and the return value is false
    //  (in this case one should use another computation method !!!)
    double z0 = 0.;
    double rorem = fcoil[i][9];  // Rmax
    double delz = z - z0;
    double ro = sqrt(r * r + delz * delz);
    if (ro < 1.e-9)  // field point is very close to the source point --> rc>1 !
        ro = 1.e-9;
    double u = delz / ro;
    double s = r / ro;
    rc = rorem / ro;  // convergence ratio
    double rcn1 = rc * rc;
    // If rc>rclimit: the Legendre polynomial series is too slow, or not convergent (for rc>1) !!!
    if (rc > frclimit) {
        return false;
    }
    // First 2 terms of Legendre polynomial P and its derivative Pp (P-primed)
    fP[0] = 1.;
    fP[1] = u;
    fPp[0] = 0.;
    fPp[1] = 1.;
    //
    for (int n = 2; n <= 3; n++) {
        fP[n] = fc1[n] * u * fP[n - 1] - fc2[n] * fP[n - 2];
        fPp[n] = fc3[n] * u * fPp[n - 1] - fc4[n] * fPp[n - 2];
    }
    double u2 = u * u;
    double rc2 = rc * rc;
    double Cz = 1. / rorem;
    double Cr = s / rorem;
    Hz = Hr = 0.;
    // We start here the series expansion.
    // Only the odd-n terms are needed in the series.
    double Hzplus, Hrplus, Hzplusold, Hrplusold, Heps, Hdelta;
    int nlast = 0;
    for (int n = 1; n <= fnmax - 1; n += 2) {
        nlast = n;
        if (n >= 4) {
            fP[n] = (fc7[n] * u2 - fc8[n]) * fP[n - 2] - fc9[n] * fP[n - 4];
            fPp[n] = (fc10[n] * u2 - fc11[n]) * fPp[n - 2] - fc12[n] * fPp[n - 4];
        }
        Hzplus = Cz * fVrem[i][n - 1] * n * rcn1 * fP[n];
        Hrplus = Cr * fVrem[i][n - 1] * rcn1 * fPp[n];
        Hz += Hzplus;
        Hr += Hrplus;
        Heps = 1.e-14 * (fabs(Hz) + fabs(Hr));
        if (n >= 3) {
            Hdelta = fabs(Hzplus) + fabs(Hrplus) + fabs(Hzplusold) + fabs(Hrplusold);
            if (Hdelta < Heps || Hdelta < 1.e-20)
                break;
        }
        rcn1 *= rc2;
        Hzplusold = Hzplus;
        Hrplusold = Hrplus;
    }
    if (nlast >= fnmax - 2) {
        rc = 1.;
        return false;
    }
    return true;
}

///////////////////////////////////////////////


//  #include "MagfieldCentral.cc"

/////////////////////////////////////////////////////

// Magnetic field calculation with central zonal harmonic expansion
//  (local axisymmetric case)

bool MagfieldCoils::Magfield2Central(bool bcoil, int ig, int j, double z, double r, double& Bz, double& Br, double& rc)
// This function computes the magnetic field components Bz and Br at fieldpoint z and r by central expansion,
//  either for a coil or for a symmetry group.
// It computes also the central convergence ratio  rc=ro/Rocen.
// bcoil=true: coil calculation;  bcoil=false: symmetry group calculation.
// ig: coil index for bcoil=true, and group index for bcoil=false.
// j: source point index.
// bcoil=true: Bcen[ig][j][n], n=0,...,nmax  central source constants for coils are used.
// bcoil=false: BcenG[ig][j][n], n=0,...,nmax  central source constants for groups are used.
// z: for a coil, it is relative to the coil center (z=0);
//       for a group: z is defined in the local group Z-coordinate system.
//  nmax: maximal index of the source constants (maximum of n).
//  SI units are used !
// If the convergence of the Legendre-series expansion is too slow:
//     rc>=rclimit,  and the return value is false
//  (in this case one should use another computation method !!!)
//
// z0, Rocen, ro, u, sr, rc, rcn:
{
    const double mu0 = 4. * M_PI * 1.e-7;
    double z0, Rocen;  // source point and conv. radius
    if (bcoil == true) {
        z0 = fz0cen[ig][j];
        Rocen = frocen[ig][j];
    }
    else {
        z0 = fz0cenG[ig][j];
        Rocen = frocenG[ig][j];
    }
    double delz = z - z0;
    double ro = sqrt(r * r + delz * delz);
    // If the field point is very close to the source point:
    if (ro < 1.e-14) {
        if (bcoil == true)
            Bz = fBcen[ig][j][0];
        else
            Bz = fBcenG[ig][j][0];
        Br = 0.;
        rc = 0.;
        return true;
    }
    double u = delz / ro;
    double sr = r / ro;
    rc = ro / Rocen;  // convergence ratio
    double rcn = rc;
    // If rc>=rclimit: the Legendre polynomial series is too slow, or not convergent (for rc>1) !!!
    if (rc >= frclimit) {
        return false;
    }
    // First 2 terms of Legendre polynomial P and its derivative Pp (P-primed)
    fP[0] = 1.;
    fP[1] = u;
    fPp[0] = 0.;
    fPp[1] = 1.;
    // First 2 terms of the series:
    if (bcoil == true) {
        Bz = fBcen[ig][j][0] + fBcen[ig][j][1] * rc * u;
        Br = -sr * fBcen[ig][j][1] / 2. * rc;
    }
    else {
        Bz = fBcenG[ig][j][0] + fBcenG[ig][j][1] * rc * u;
        Br = -sr * fBcenG[ig][j][1] / 2. * rc;
    }
    int nlast = 0;
    if (rc < 1.e-10)
        goto label;
    //
    // We start here the central series expansion:
    double bcen;
    for (int n = 2; n <= fnmax - 1; n++) {
        if (bcoil == true)
            bcen = fBcen[ig][j][n];
        else
            bcen = fBcenG[ig][j][n];
        nlast = n;
        rcn *= rc;
        double uPp = u * fPp[n - 1];
        double uP = u * fP[n - 1];
        fP[n] = 2. * uP - fP[n - 2] - (uP - fP[n - 2]) * fC0[n];
        fPp[n] = 2. * uPp - fPp[n - 2] + (uPp - fPp[n - 2]) * fC1[n];
        //     P[n]=c1[n]*u*P[n-1]-c2[n]*P[n-2];
        //     Pp[n]=c3[n]*u*Pp[n-1]-c4[n]*Pp[n-2];
        fBzplus[n] = bcen * rcn * fP[n];
        fBrplus[n] = -sr * bcen * fc6[n] * rcn * fPp[n];
        Bz += fBzplus[n];
        Br += fBrplus[n];
        if (n > 5) {
            double Beps = 1.e-15 * (fabs(Bz) + fabs(Br));
            double Bdelta = fabs(fBzplus[n]) + fabs(fBrplus[n]) + fabs(fBzplus[n - 1]) + fabs(fBrplus[n - 1]) +
                            fabs(fBzplus[n - 2]) + fabs(fBrplus[n - 2]) + fabs(fBzplus[n - 3]) + fabs(fBrplus[n - 3]);
            if (Bdelta < Beps || Bdelta < 1.e-20)
                break;
        }
    }
    if (nlast >= fnmax - 1) {
        rc = 1.;
        return false;
    }
label:;
    // Effective central correction for Bz:
    if (bcoil == true) {
        double Lhalf = fcoil[ig][7] * 0.5;
        if (z > -Lhalf && z < Lhalf) {
            double Rmin = fcoil[ig][8], Rmax = fcoil[ig][9];
            double sigma = fcoil[ig][10];  // current density
            if (r > Rmax)
                Bz += -mu0 * sigma * (Rmax - Rmin);
            else if (r > Rmin && r < Rmax)
                Bz += -mu0 * sigma * (r - Rmin);
        }
    }
    return true;
}


/////////////////////////////////////////////////


//  #include "MagfieldCoil.cc"

bool MagfieldCoils::MagfieldCoil(int i, const double* P, double* B)
{
    // This function computes the magnetic field components B[0],B[1],B[2]
    // at a field point P[0],P[1],P[2], due to the 3-dimensional coil
    // with index i, using central, remote or magnetic charge zonal harmonic expansions,
    // or elliptic integrals.
    // SI units are used (P[k] in m, B[k] in T, k=0,1,2 --> components x, y, z)!
    // The coil is defined by the coil parameters coil[i][j], j=0,...,13.
    // The return value is true, if only zonal harmonic expansions are used for the magnetic field calc.;
    //   if elliptic integral calc. is used: the return value is false.
    //
    // We start now the magnetic field calculation.
    // First we define the local coordinate system of the coil, and the
    //   z and r local coordinates of the field point P.
    // Local coordinate system:
    //    origo at coil center,
    //    local z axis parallel to coil axis, in u vector direction.
    bool magfieldcoil = true;
    double C[3], u[3];
    for (int k = 0; k < 3; k++)
        C[k] = fcoil[i][1 + k];  // coil center
    u[0] = fcoil[i][11];         // coil direction unit vector component ux
    u[1] = fcoil[i][12];         // coil direction unit vector component uy
    u[2] = fcoil[i][13];         // coil direction unit vector component uz
    // Local cylindrical z and r coordinates of the field point P (relative to coil center and direction):
    double Ploc[3], Pr[3], w[3];
    for (int k = 0; k < 3; k++)
        Ploc[k] = P[k] - C[k];
    double z = Ploc[1] * u[1] + Ploc[2] * u[2] + Ploc[0] * u[0];
    for (int k = 0; k < 3; k++)
        Pr[k] = Ploc[k] - z * u[k];
    double r = sqrt(Pr[1] * Pr[1] + Pr[2] * Pr[2] + Pr[0] * Pr[0]);
    double r2 = r * r;
    //
    // Starting now Bz and Br calculation.
    bool sps;
    int jbest;
    double rcbest;
    //  ---------------------------
    // Step 1. Probably most of the coils are far from the field point, therefore
    // we try first the remote series expansion
    // (with remote source point at the coil center)
    double ro2 = r2 + z * z;
    double rorem2 = frorem[i] * frorem[i];
    double Bz, Br, rc;
    if (ro2 > rorem2 * 2.) {
        bool magfield = Magfield2Remote(true, i, z, r, Bz, Br, rc);
        if (magfield == true)
            goto labelend;
    }
    // Step 2. We try the central zonal method in the beginning of tracking,
    // by searching all central source points.
    if (fjlast[i] == -2) {
        fjlast[i] = -1;
        sps = SourcePointSearching(true, i, z, r, 0, jbest, rcbest);
        if (sps == true) {
            bool magfield = Magfield2Central(true, i, jbest, z, r, Bz, Br, rc);
            if (magfield == true) {
                magfieldcoil = magfield;
                goto labelend;
            }
        }
    }
    //  ---------------------------
    // Step 3. The field point is close to the coil. We try next the central
    // Legendre polynomial expansion. If some source point for this coil
    // has been already used (jlast[i]>-1), we search the central source
    // point j close to the old source point jlast[i].
    sps = false;
    if (fjlast[i] > -1)
        sps = SourcePointSearching(true, i, z, r, 1, jbest, rcbest);
    if (fjlast[i] > -1 && sps == false)
        sps = SourcePointSearching(true, i, z, r, 2, jbest, rcbest);
    if (sps == true) {
        bool magfield = Magfield2Central(true, i, jbest, z, r, Bz, Br, rc);
        if (magfield == true)
            goto labelend;
    }
    //  ---------------------------
    // Step 4. We try again the remote series expansion
    // (with remote source point at the coil center)
    if (ro2 > rorem2 * 1.0204) {
        bool magfield = Magfield2Remote(true, i, z, r, Bz, Br, rc);
        if (magfield == true)
            goto labelend;
    }
    //  ---------------------------
    //  ---------------------------
    // Step 5. We try now the magnetic charge method
    {
        double L = fcoil[i][7];
        double Rmax = fcoil[i][9];  // coil length and outer radius
        double Zmin = -L / 2.;
        double Zmax = L / 2.;  // coil endpoints relative to coil center
        double romax2 = r2 + (z - Zmax) * (z - Zmax);
        double romin2 = r2 + (z - Zmin) * (z - Zmin);
        rorem2 = Rmax * Rmax;
        if (romax2 > rorem2 * 1.0204 && romin2 > rorem2 * 1.0204) {
            bool magfield = Magfield2Magcharge(i, z, r, Bz, Br, rc);
            if (magfield == true)
                goto labelend;
        }
    }
    //  ---------------------------
    // Step 6. We try again the central zonal method, by searching now all central source points.
    sps = SourcePointSearching(true, i, z, r, 0, jbest, rcbest);
    if (sps == true) {
        bool magfield = Magfield2Central(true, i, jbest, z, r, Bz, Br, rc);
        if (magfield == true) {
            magfieldcoil = magfield;
            goto labelend;
        }
    }
    // -----------------------------
    // Step 7. Unfortunately, no appropriate central, remote or magnetic charge expansion
    // was found. We have to use elliptic integrals:
    //     cout << "step 6 " << endl;
    Magfield2EllipticCoil(i, z, r, Bz, Br);
    magfieldcoil = false;
//
// B[k],k=0,1,2 calculation from Bz, Br:
labelend:;
    if (r < 1.e-15 || fabs(Br) < fabs(Bz) * 1.e-15)
        for (int k = 0; k < 3; k++)
            B[k] = Bz * u[k];
    else {
        for (int k = 0; k < 3; k++)
            w[k] = Pr[k] / r;
        for (int k = 0; k < 3; k++)
            B[k] = Bz * u[k] + Br * w[k];
    }
    return magfieldcoil;
}


/////////////////////////////////////////////////

// #include "MagfieldGroup.cc"

bool MagfieldCoils::MagfieldGroup(int g, const double* P, double* B)
{
    // This function computes the magnetic field components B[0],B[1],B[2]
    // at a field point P[0],P[1],P[2], due to the symmetry group
    // with index g, using central or remote  zonal harmonic expansions,
    // or single coil calculations.
    // SI units are used (P[k] in m, B[k] in T, k=0,1,2 --> components x, y, z)!
    // The return value is true, if only zonal harmonic expansions are used for the magnetic field calc.;
    //   if elliptic integral calc. is also used: the return value is false.
    // If the group has only 1 coil --> coil calculation:
    if (fNc[g] == 1)
        return MagfieldCoil(fCin[g][0], P, B);
    //
    // We start now the magnetic field calculation.
    // First we define the local coordinate system of the group, and the
    //   z and r local coordinates of the field point P.
    // Local coordinate system:
    //    origo at Z[g][0], local z axis parallel to group axis, in u vector direction.
    bool magfieldgroup = true;
    double C[3], u[3], w[3];
    for (int k = 0; k < 3; k++)
        C[k] = fLine[g][k];  // group origo
    for (int k = 0; k < 3; k++)
        u[k] = fLine[g][3 + k];  // group direction unit vector
    // Local cylindrical z and r coordinates of the field point P (relative to group origo and direction):
    double Ploc[3], Pr[3];
    for (int k = 0; k < 3; k++)
        Ploc[k] = P[k] - C[k];
    double z = Ploc[1] * u[1] + Ploc[2] * u[2] + Ploc[0] * u[0];
    for (int k = 0; k < 3; k++)
        Pr[k] = Ploc[k] - z * u[k];
    double r = sqrt(Pr[1] * Pr[1] + Pr[2] * Pr[2] + Pr[0] * Pr[0]);
    double r2 = r * r;
    //
    // Starting now Bz and Br calculation.
    bool sps;
    int jbest;
    double rcbest;
    //  ---------------------------
    // Step 1. We try first the remote series expansion
    // (with remote source point at the group center)
    double ro2 = r2 + (z - fz0remG[g]) * (z - fz0remG[g]);
    double rorem2 = froremG[g] * froremG[g];
    double Bz, Br, rc;
    if (ro2 > rorem2 * 2.) {
        bool magfield = Magfield2Remote(false, g, z, r, Bz, Br, rc);
        if (magfield == true)
            goto labelend;
    }
    //  ---------------------------
    // Step 2. We try the central zonal method in the beginning of tracking,
    // by searching all central source points.
    if (fjlastG[g] == -2) {
        fjlastG[g] = -1;
        sps = SourcePointSearching(false, g, z, r, 0, jbest, rcbest);
        if (sps == true) {
            bool magfield = Magfield2Central(false, g, jbest, z, r, Bz, Br, rc);
            if (magfield == true)
                goto labelend;
        }
    }
    //  ---------------------------
    // Step 3. The field point is close to the group. We try next the central
    // Legendre polynomial expansion. If some source point for this group
    // has been already used (jlastG[g]>-1), we search the central source
    // point j close to the old source point jlastG[g].
    sps = false;
    if (fjlastG[g] > -1)
        sps = SourcePointSearching(false, g, z, r, 1, jbest, rcbest);
    if (fjlastG[g] > -1 && sps == false)
        sps = SourcePointSearching(false, g, z, r, 2, jbest, rcbest);
    if (sps == true) {
        bool magfield = Magfield2Central(false, g, jbest, z, r, Bz, Br, rc);
        if (magfield == true)
            goto labelend;
    }
    //  ---------------------------
    // Step 4. We try again the remote series expansion
    // (with remote source point at the group center)
    if (ro2 > rorem2 * 1.0204) {
        bool magfield = Magfield2Remote(false, g, z, r, Bz, Br, rc);
        if (magfield == true)
            goto labelend;
    }
    //  ---------------------------
    // Step 5. We try again the central zonal method, by searching now all central source points.
    sps = SourcePointSearching(false, g, z, r, 0, jbest, rcbest);
    if (sps == true) {
        bool magfield = Magfield2Central(false, g, jbest, z, r, Bz, Br, rc);
        if (magfield == true)
            goto labelend;
    }
    // -----------------------------
    // Step 6. Unfortunately, no appropriate central or remote expansion
    // was found. We have to use coil calculations:
    {
        B[0] = B[1] = B[2] = 0.;
        double Bcoil[3];
        bool magfieldcoil;
        for (int c = 0; c < fNc[g]; c++) {
            int i = fCin[g][c];
            magfieldcoil = MagfieldCoil(i, P, Bcoil);
            for (int k = 0; k < 3; k++)
                B[k] += Bcoil[k];
            if (magfieldcoil == false)
                magfieldgroup = false;
        }
        return magfieldgroup;
    }
//
// B[k],k=0,1,2 calculation from Bz, Br from remote or central zonal harmonic expansion:
labelend:;
    if (r < 1.e-15 || fabs(Br) < fabs(Bz) * 1.e-15)
        for (int k = 0; k < 3; k++)
            B[k] = Bz * u[k];
    else {
        for (int k = 0; k < 3; k++)
            w[k] = Pr[k] / r;
        for (int k = 0; k < 3; k++)
            B[k] = Bz * u[k] + Br * w[k];
    }
    return magfieldgroup;
}

/////////////////////////////////////////////////

//  #include "MagfieldSPS.cc"

bool MagfieldCoils::SourcePointSearching(bool coil, int ig, double z, double r, int type, int& jbest, double& rcbest)
{
    // This function searches the optimal central source point either
    // for a coil (if coil==true) or for a group (if coil==false).
    // Coil or group index: ig.
    // Field point is defined by z and r (relative to the coil center or in the
    // group Z-system, and to the coil or group direction).
    // type=0: all central source points of the coil or group are searched;
    // type=1:  only the 3 source points near the last source point jlast are searched;
    // type=2:  only the 11 source points near the last source point jlast are searched.
    // If a good central source point is found: the function return value is true;
    //     if no good central source point is found: the function return value is false.
    // The best source point index found by this function: jbest (should be used only with true return value);
    //  the corresponding best convergence ratio (for source point jbest): rcbest (only for true return value).
    double rcmin2, delz, ro2, rocen2;
    double r2 = r * r;
    if (coil == true)  // coil case
    {
        int i = ig;
        if (type == 0)  // all central source points of the coil are searched
        {
            rcmin2 = frclimit * frclimit;
            jbest = -1;
            for (int j = 0; j < fNsp[i]; j++) {
                delz = z - fz0cen[i][j];
                ro2 = r2 + delz * delz;
                rocen2 = frocen[i][j] * frocen[i][j];
                if (ro2 < rocen2 * rcmin2) {
                    rcmin2 = ro2 / rocen2;
                    jbest = j;
                }
            }
            if (jbest > -1) {
                rcbest = sqrt(rcmin2);
                fjlast[i] = jbest;
                return true;
            }
            else
                return false;
        }     // end of coil=true,  type=0 case
        else  // only the 3 or the 11 source points near the last source point jlast are searched
        {
            if (fjlast[i] < 0)
                return false;
            int delj;
            if (type == 1)
                delj = 1;
            else
                delj = 5;
            int jmin = fjlast[i] - delj;
            if (jmin < 0)
                jmin = 0;
            int jmax = fjlast[i] + delj;
            if (jmax > fNsp[i] - 1)
                jmax = fNsp[i] - 1;
            rcmin2 = frclimit * frclimit;
            jbest = -1;
            for (int j = jmin; j <= jmax; j++) {
                delz = z - fz0cen[i][j];
                ro2 = r2 + delz * delz;
                rocen2 = frocen[i][j] * frocen[i][j];
                if (ro2 < rocen2 * rcmin2) {
                    rcmin2 = ro2 / rocen2;
                    jbest = j;
                }
            }
            if (jbest > -1) {
                rcbest = sqrt(rcmin2);
                fjlast[i] = jbest;
                return true;
            }
            else
                return false;
        }  // end of coil=true,  type=1 or 2 case
    }      // end of coil case
    else   // group case
    {
        int g = ig;
        if (type == 0)  // all central source points of the group are searched
        {
            rcmin2 = frclimit * frclimit;
            jbest = -1;
            for (int j = 0; j < fNspG[g]; j++) {
                delz = z - fz0cenG[g][j];
                ro2 = r2 + delz * delz;
                rocen2 = frocenG[g][j] * frocenG[g][j];
                if (ro2 < rocen2 * rcmin2) {
                    rcmin2 = ro2 / rocen2;
                    jbest = j;
                }
            }
            if (jbest > -1) {
                rcbest = sqrt(rcmin2);
                fjlastG[g] = jbest;
                return true;
            }
            else
                return false;
        }     // end of coil=false,  type=0 case
        else  // only the 3 or the 11 source points near the last source point jlast are searched
        {
            if (fjlastG[g] < 0)
                return false;
            int delj;
            if (type == 1)
                delj = 1;
            else
                delj = 5;
            int jmin = fjlastG[g] - delj;
            if (jmin < 0)
                jmin = 0;
            int jmax = fjlastG[g] + delj;
            if (jmax > fNspG[g] - 1)
                jmax = fNspG[g] - 1;
            rcmin2 = frclimit * frclimit;
            jbest = -1;
            for (int j = jmin; j <= jmax; j++) {
                delz = z - fz0cenG[g][j];
                ro2 = r2 + delz * delz;
                rocen2 = frocenG[g][j] * frocenG[g][j];
                if (ro2 < rocen2 * rcmin2) {
                    rcmin2 = ro2 / rocen2;
                    jbest = j;
                }
            }
            if (jbest > -1) {
                rcbest = sqrt(rcmin2);
                fjlastG[g] = jbest;
                return true;
            }
            else
                return false;
        }  // end of coil=false,  type=1 or 2 case
    }      // end of group case
}


//  #include "CarlsonEllipticIntegrals.cc"
/////////////////////////////////////////////////////

// Complete elliptic integral calculations of Carlson
//           (according to Numerical Recipes)

/////////////////////////////////////////////////////


double MagfieldCoils::RF_Carlson(double x, double y, double z)
{
    // This function computes Carlson's elliptic integral of the first kind:
    // R_F(x,y,z). x, y, z must be nonnegative, and at most one can be zero
    //  (see: Press et al., Numerical Recipes, Sec. 6.11).
    const double ERRTOL = 0.002, TINY = 1.e-38, BIG = 1.e38, C1 = 1. / 24., C2 = 0.1, C3 = 3. / 44., C4 = 1. / 14.,
                 THIRD = 1. / 3.;
    double alamb, ave, delx, dely, delz, e2, e3, sqrtx, sqrty, sqrtz, xt, yt, zt;
    if (FMIN3(x, y, z) < 0. || FMIN3(x + y, x + z, y + z) < TINY || FMAX3(x, y, z) > BIG) {
        puts("Message from function MagfieldCoils::RF_Carlson: invalid arguments !!!");
        puts("Program running is stopped !!! ");
        exit(1);
    }
    xt = x;
    yt = y;
    zt = z;
    do {
        sqrtx = sqrt(xt);
        sqrty = sqrt(yt);
        sqrtz = sqrt(zt);
        alamb = sqrtx * (sqrty + sqrtz) + sqrty * sqrtz;
        xt = 0.25 * (xt + alamb);
        yt = 0.25 * (yt + alamb);
        zt = 0.25 * (zt + alamb);
        ave = THIRD * (xt + yt + zt);
        delx = (ave - xt) / ave;
        dely = (ave - yt) / ave;
        delz = (ave - zt) / ave;
    } while (FMAX3(fabs(delx), fabs(dely), fabs(delz)) > ERRTOL);
    e2 = delx * dely - delz * delz;
    e3 = delx * dely * delz;
    return (1. + (C1 * e2 - C2 - C3 * e3) * e2 + C4 * e3) / sqrt(ave);
}

///////////////////////////////////////////////////////////////

double MagfieldCoils::RD_Carlson(double x, double y, double z)
{
    // This function computes Carlson's elliptic integral of the second kind:
    // R_D(x,y,z). x and y must be nonnegative, and at most one can be zero.
    // z must be positive
    //  (see: Press et al., Numerical Recipes, Sec. 6.11).
    const double ERRTOL = 0.0015, TINY = 1.e-25, BIG = 1.e22, C1 = 3. / 14., C2 = 1. / 6., C3 = 9. / 22., C4 = 3. / 26.,
                 C5 = 0.25 * C3, C6 = 1.5 * C4;
    double alamb, ave, delx, dely, delz, ea, eb, ec, ed, ee, fac, sum, sqrtx, sqrty, sqrtz, xt, yt, zt;
    if (FMIN(x, y) < 0. || FMIN(x + y, z) < TINY || FMAX3(x, y, z) > BIG) {
        puts("Message from function MagfieldCoils::RD_Carlson: invalid arguments !!!");
        puts("Program running is stopped !!! ");
        exit(1);
    }
    xt = x;
    yt = y;
    zt = z;
    sum = 0.;
    fac = 1.;
    do {
        sqrtx = sqrt(xt);
        sqrty = sqrt(yt);
        sqrtz = sqrt(zt);
        alamb = sqrtx * (sqrty + sqrtz) + sqrty * sqrtz;
        sum += fac / (sqrtz * (zt + alamb));
        fac = 0.25 * fac;
        xt = 0.25 * (xt + alamb);
        yt = 0.25 * (yt + alamb);
        zt = 0.25 * (zt + alamb);
        ave = 0.2 * (xt + yt + 3. * zt);
        delx = (ave - xt) / ave;
        dely = (ave - yt) / ave;
        delz = (ave - zt) / ave;
    } while (FMAX3(fabs(delx), fabs(dely), fabs(delz)) > ERRTOL);
    ea = delx * dely;
    eb = delz * delz;
    ec = ea - eb;
    ed = ea - 6. * eb;
    ee = ed + ec + ec;
    return 3. * sum +
           fac * (1. + ed * (-C1 + C5 * ed - C6 * delz * ee) + delz * (C2 * ee + delz * (-C3 * ec + delz * C4 * ea))) /
               (ave * sqrt(ave));
}

///////////////////////////////////////////////////////////////

double MagfieldCoils::RJ_Carlson(double x, double y, double z, double p)
{
    // This function computes Carlson's elliptic integral of the third kind:
    // R_J(x,y,z,p). x, y and z must be nonnegative, and at most one can be zero.
    // p must be nonzero. If p<0, the Cauchy principal value is returned.
    //  (see: Press et al., Numerical Recipes, Sec. 6.11).
    const double ERRTOL = 0.0015, TINY = 1.e-20, BIG = 1.e12, C1 = 3. / 14., C2 = 1. / 3., C3 = 3. / 22., C4 = 3. / 26.,
                 C5 = 0.75 * C3, C6 = 1.5 * C4, C7 = 0.5 * C2, C8 = 2. * C3;
    double alamb, alpha, ans, ave, beta, delp, delx, dely, delz, ea, eb, ec, ed, ee, fac, pt, rho, sum, sqrtx, sqrty,
        sqrtz, tau, xt, yt, zt;
    double rcx = 0.;
    double b = 0.;
    double a = 0.;
    if (FMIN3(x, y, z) < 0. || FMIN(FMIN(x + y, x + z), FMIN(y + z, fabs(p))) < TINY ||
        FMAX(FMAX(x, y), FMAX(z, fabs(p))) > BIG) {
        puts("Message from function MagfieldCoils::RJ_Carlson: invalid arguments !!!");
        puts("Program running is stopped !!! ");
        exit(1);
    }
    sum = 0.;
    fac = 1.;
    if (p > 0.) {
        xt = x;
        yt = y;
        zt = z;
        pt = p;
    }
    else {
        xt = FMIN3(x, y, z);
        zt = FMAX3(x, y, z);
        yt = x + y + z - xt - zt;
        a = 1. / (yt - p);
        b = a * (zt - yt) * (yt - xt);
        pt = yt + b;
        rho = xt * zt / yt;
        tau = p * pt / yt;
        rcx = RC_Carlson(rho, tau);
    }
    do {
        sqrtx = sqrt(xt);
        sqrty = sqrt(yt);
        sqrtz = sqrt(zt);
        alamb = sqrtx * (sqrty + sqrtz) + sqrty * sqrtz;
        alpha = pow2(pt * (sqrtx + sqrty + sqrtz) + sqrtx * sqrty * sqrtz);
        beta = pt * pow2(pt + alamb);
        sum += fac * RC_Carlson(alpha, beta);
        ;
        fac = 0.25 * fac;
        xt = 0.25 * (xt + alamb);
        yt = 0.25 * (yt + alamb);
        zt = 0.25 * (zt + alamb);
        pt = 0.25 * (pt + alamb);
        ave = 0.2 * (xt + yt + zt + 2. * pt);
        delx = (ave - xt) / ave;
        dely = (ave - yt) / ave;
        delz = (ave - zt) / ave;
        delp = (ave - pt) / ave;
    } while (FMAX(FMAX(fabs(delx), fabs(dely)), FMAX(fabs(delz), fabs(delp))) > ERRTOL);
    ea = delx * (dely + delz) + dely * delz;
    eb = delx * dely * delz;
    ec = delp * delp;
    ed = ea - 3. * ec;
    ee = eb + 2. * delp * (ea - ec);
    ans = 3. * sum + fac *
                         (1. + ed * (-C1 + C5 * ed - C6 * ee) + eb * (C7 + delp * (-C8 + delp * C4)) +
                          delp * ea * (C2 - delp * C3) - C2 * delp * ec) /
                         (ave * sqrt(ave));
    if (p < 0.)
        ans = a * (b * ans + 3. * (rcx - RF_Carlson(xt, yt, zt)));
    return ans;
}

///////////////////////////////////////////////////////////////

double MagfieldCoils::RC_Carlson(double x, double y)
{
    // This function computes Carlson's degenerate elliptic integral:
    // R_C(x,y). x must be nonnegative, and y must be nonzero.
    // If y<0, the Cauchy principal value is returned.
    //  (see: Press et al., Numerical Recipes, Sec. 6.11).
    const double ERRTOL = 0.001, TINY = 1.e-38, BIG = 1.e38, SQRTNY = 1.e-19, TNBG = TINY * BIG, COMP1 = 2.236 / SQRTNY,
                 COMP2 = TNBG * TNBG / 25., THIRD = 1. / 3., C1 = 0.3, C2 = 1. / 7., C3 = 0.375, C4 = 9. / 22.;
    double alamb, ave, s, w, xt, yt;
    if (x < 0. || y == 0. || (x + fabs(y)) < TINY || x + fabs(y) > BIG || (y < -COMP1 && x > 0. && x < COMP2)) {
        puts("Message from function MagfieldCoils::RC_Carlson: invalid arguments !!!");
        puts("Program running is stopped !!! ");
        exit(1);
    }
    if (y > 0.) {
        xt = x;
        yt = y;
        w = 1.;
    }
    else {
        xt = x - y;
        yt = -y;
        w = sqrt(x) / sqrt(xt);
    }
    do {
        alamb = 2. * sqrt(xt) * sqrt(yt) + yt;
        xt = 0.25 * (xt + alamb);
        yt = 0.25 * (yt + alamb);
        ave = THIRD * (xt + yt + yt);
        s = (yt - ave) / ave;
    } while (fabs(s) > ERRTOL);
    return w * (1. + s * s * (C1 + s * (C2 + s * (C3 + s * C4)))) / sqrt(ave);
}

// #include "GaussLegendreIntegration.cc"

////////////////////////////////////////////////////////////////////////////////////////////

// Calculation of integration nodes and weights for Gauss-Legendre integration

void MagfieldCoils::GaussLegendreIntegration(int& N, double a, double b, double* x, double* w)
// This function computes the nodes x[i] and weights w[i], i=0,...,N-1, for numerical Gauss-Legendre integration
// from a to b.
// This code can change the input integer N!
// Dimension of arrays x and w has to be minimum N!
{
    const double x2[1] = {0.5773502691896257};
    const double w2[1] = {1.};
    const double x3[2] = {0., 0.7745966692414834};
    const double w3[2] = {0.8888888888888888, 0.5555555555555556};
    const double x4[2] = {0.3399810435848563, 0.8611363115940526};
    const double w4[2] = {0.6521451548625461, 0.3478548451374538};
    const double x6[3] = {0.2386191860831969, 0.6612093864662645, 0.9324695142031521};
    const double w6[3] = {0.4679139345726910, 0.3607615730481386, 0.1713244923791704};
    const double x8[4] = {0.1834346424956498, 0.5255324099163290, 0.7966664774136267, 0.9602898564975363};
    const double w8[4] = {0.3626837833783620, 0.3137066458778873, 0.2223810344533745, 0.1012285362903763};
    const double x16[8] = {0.09501250983763744,
                           0.28160355077925891,
                           0.45801677765722739,
                           0.61787624440264375,
                           0.75540440835500303,
                           0.86563120238783174,
                           0.94457502307323258,
                           0.98940093499164993};
    const double w16[8] = {0.189450610455068496,
                           0.182603415044923589,
                           0.169156519395002532,
                           0.149595988816576731,
                           0.124628971255533872,
                           0.095158511682492785,
                           0.062253523938647892,
                           0.027152459411754095};
    const double x32[16] = {0.048307665687738316,
                            0.144471961582796493,
                            0.239287362252137075,
                            0.331868602282127650,
                            0.421351276130635345,
                            0.506899908932229390,
                            0.587715757240762329,
                            0.663044266930215201,
                            0.732182118740289680,
                            0.794483795967942407,
                            0.849367613732569970,
                            0.896321155766052124,
                            0.934906075937739689,
                            0.964762255587506431,
                            0.985611511545268335,
                            0.997263861849481564};
    const double w32[16] = {0.09654008851472780056,
                            0.09563872007927485942,
                            0.09384439908080456564,
                            0.09117387869576388471,
                            0.08765209300440381114,
                            0.08331192422694675522,
                            0.07819389578707030647,
                            0.07234579410884850625,
                            0.06582222277636184684,
                            0.05868409347853554714,
                            0.05099805926237617619,
                            0.04283589802222680057,
                            0.03427386291302143313,
                            0.02539206530926205956,
                            0.01627439473090567065,
                            0.00701861000947009660};
    static double X[33][32];
    static double W[33][32];
    static bool start = true;
    int m, n;
    if (start == true) {
        m = 2;
        X[m][0] = -x2[0];
        X[m][1] = x2[0];
        W[m][0] = w2[0];
        W[m][1] = w2[0];
        m = 3;
        X[m][0] = -x3[1];
        X[m][1] = x3[0];
        X[m][2] = x3[1];
        W[m][0] = w3[1];
        W[m][1] = w3[0];
        W[m][2] = w3[1];
        m = 4;
        n = m / 2;
        for (int i = 0; i < n; i++) {
            X[m][i] = -x4[n - 1 - i];
            W[m][i] = w4[n - 1 - i];
            X[m][n + i] = x4[i];
            W[m][n + i] = w4[i];
        }
        m = 6;
        n = m / 2;
        for (int i = 0; i < n; i++) {
            X[m][i] = -x6[n - 1 - i];
            W[m][i] = w6[n - 1 - i];
            X[m][n + i] = x6[i];
            W[m][n + i] = w6[i];
        }
        m = 8;
        n = m / 2;
        for (int i = 0; i < n; i++) {
            X[m][i] = -x8[n - 1 - i];
            W[m][i] = w8[n - 1 - i];
            X[m][n + i] = x8[i];
            W[m][n + i] = w8[i];
        }
        m = 16;
        n = m / 2;
        for (int i = 0; i < n; i++) {
            X[m][i] = -x16[n - 1 - i];
            W[m][i] = w16[n - 1 - i];
            X[m][n + i] = x16[i];
            W[m][n + i] = w16[i];
        }
        m = 32;
        n = m / 2;
        for (int i = 0; i < n; i++) {
            X[m][i] = -x32[n - 1 - i];
            W[m][i] = w32[n - 1 - i];
            X[m][n + i] = x32[i];
            W[m][n + i] = w32[i];
        }
        start = false;
    }
    //
    if (N < 2)
        N = 2;
    else if (N >= 5 && N <= 6)
        N = 6;
    else if (N >= 7 && N <= 11)
        N = 8;
    else if (N >= 12 && N <= 20)
        N = 16;
    else if (N >= 21 && N <= 45)
        N = 32;
    else if (N >= 46) {
        int M = N / 32 + 1;
        N = 32 * M;
    }
    //
    double Cp = (b + a) / 2.;
    double Cm = (b - a) / 2.;
    if (N <= 32) {
        for (int i = 0; i < N; i++) {
            x[i] = Cp + Cm * X[N][i];
            w[i] = Cm * W[N][i];
        }
    }
    else {
        int M = N / 32;              // number of subintervals
        double d = (b - a) / M;      // length of subinterval
        for (int m = 0; m < M; m++)  // subinterval loop
        {
            double A = a + d * m;
            double B = A + d;  // subinterval limits
            Cp = (B + A) / 2.;
            Cm = (B - A) / 2.;
            int j = 32 * m;
            for (int i = 0; i < 32; i++) {
                x[j + i] = Cp + Cm * X[32][i];
                w[j + i] = Cm * W[32][i];
            }
        }
    }
}

//////////////////////////////////////////////////////////////////////////////

/*
   The class MagfieldCoils computes magnetic field of coils. Each coil is axisymmetric in its own symmetry
   system, but these systems can have arbitrary directions.
   The code is similar to the C code magfield3,
   but it also contains many changes. The main reference to the code is the paper:
     Progress In Electromagnetics Research B 32, 352 (2011).
The C++ code has 2 constructors: 
  A, The first constructor has 6 input parameters; the user has to employ this one in the beginning of the calculation.
  This reads the coil parameters (by function CoilRead) from the file defined by the string coilfilename; these parameters are stored
  in the array coil[fNcoil][14], where fNcoil denotes the number of coils. Function CoilRead
  contains explanations about the coil input file structure.
  Then the coil parameters are tested,
  and symmetry groups are computed; the coil and symmetry group parameters are written into the
  coil-group data file (by function CoilGroupWrite).
Afterwards, various types (central, remote and magnetic charge) of source constants are computed, and 
they are written into 2 source data files.
See the above paper concerning the symmetry groups and the details of the source constant computations.
  B, The second constructor has only 2 string input parameters: these define the directory and file name of the
coil-group and the 2 source constant data files. These 3 files are then read to the memory, and distributed into
various arrays. The user can employ this second constructor if the coil-group and source files have been already
earlier computed by using the first constructor (the second constructor is much faster than the first one).

After having defined a MagfieldCoils object by using either the first or the second constructor, the user can
make magnetic field calculations. The main function for this purpose is the Magfield(P, B), where
P[0], P[1] and P[2] define the x,y,z components of the field point, and B[0], B[1] ans B[2] are the Bx, By and Bz
magnetic field components. The function Magfield attempts to use the zonal harmonics expansion methods
(central, remote or magnetic charge) for the magnetic field calculation; if these are not convergent at the
given field point, then the much slower elliptic integral method is used.
The bool return value of this function shows whether the elliptic integral method has been used (true if only
the zonal harmonic method has been used; false if for some coils also the elliptic integral method was used).
With the function MagfieldElliptic one can directly use the elliptic integral method for all coils
(the magnetic field value by MagfieldElliptic should be very precisely equal to the magnetic field value
of the Magfield function).

*/
