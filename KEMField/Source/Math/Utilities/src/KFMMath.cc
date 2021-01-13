#include "KFMMath.hh"

#include <cstdlib>

namespace KEMField
{

const double KFMMath::sqrt_of_three = 1.7320508075688772935274463;

////////////////////////////////////////////////////////////////////////////////
//table of factorials for integers up to 170

const double KFMMath::factorial_table[FACTORIAL_TABLE_SIZE] = {
    1.0,                                      //0
    1.0,                                      //1
    2.0,                                      //2
    6.0,                                      //3
    24.0,                                     //4
    120.0,                                    //5
    720.0,                                    //6
    5040.0,                                   //7
    40320.0,                                  //8
    362880.0,                                 //9
    3628800.0,                                //10
    39916800.0,                               //11
    479001600.0,                              //12
    6227020800.0,                             //13
    87178291200.0,                            //14
    1307674368000.0,                          //15
    20922789888000.0,                         //16
    355687428096000.0,                        //17
    6402373705728000.0,                       //18
    121645100408832000.0,                     //19
    2432902008176640000.0,                    //20
    51090942171709440000.0,                   //21
    1124000727777607680000.0,                 //22
    25852016738884976640000.0,                //23
    620448401733239439360000.0,               //24
    15511210043330985984000000.0,             //25
    403291461126605635584000000.0,            //26
    10888869450418352160768000000.0,          //27
    304888344611713860501504000000.0,         //28
    8841761993739701954543616000000.0,        //29
    265252859812191058636308480000000.0,      //30
    8222838654177922817725562880000000.0,     //31
    263130836933693530167218012160000000.0,   //32
    8683317618811886495518194401280000000.0,  //33
    2.95232799039604140847618609644e38,       //34
    1.03331479663861449296666513375e40,       //35
    3.71993326789901217467999448151e41,       //36
    1.37637530912263450463159795816e43,       //37
    5.23022617466601111760007224100e44,       //38
    2.03978820811974433586402817399e46,       //39
    8.15915283247897734345611269600e47,       //40
    3.34525266131638071081700620534e49,       //41
    1.40500611775287989854314260624e51,       //42
    6.04152630633738356373551320685e52,       //43
    2.65827157478844876804362581101e54,       //44
    1.19622220865480194561963161496e56,       //45
    5.50262215981208894985030542880e57,       //46
    2.58623241511168180642964355154e59,       //47
    1.24139155925360726708622890474e61,       //48
    6.08281864034267560872252163321e62,       //49
    3.04140932017133780436126081661e64,       //50
    1.55111875328738228022424301647e66,       //51
    8.06581751709438785716606368564e67,       //52
    4.27488328406002556429801375339e69,       //53
    2.30843697339241380472092742683e71,       //54
    1.26964033536582759259651008476e73,       //55
    7.10998587804863451854045647464e74,       //56
    4.05269195048772167556806019054e76,       //57
    2.35056133128287857182947491052e78,       //58
    1.38683118545689835737939019720e80,       //59
    8.32098711274139014427634118320e81,       //60
    5.07580213877224798800856812177e83,       //61
    3.14699732603879375256531223550e85,       //62
    1.982608315404440064116146708360e87,      //63
    1.268869321858841641034333893350e89,      //64
    8.247650592082470666723170306800e90,      //65
    5.443449390774430640037292402480e92,      //66
    3.647111091818868528824985909660e94,      //67
    2.480035542436830599600990418570e96,      //68
    1.711224524281413113724683388810e98,      //69
    1.197857166996989179607278372170e100,     //70
    8.504785885678623175211676442400e101,     //71
    6.123445837688608686152407038530e103,     //72
    4.470115461512684340891257138130e105,     //73
    3.307885441519386412259530282210e107,     //74
    2.480914081139539809194647711660e109,     //75
    1.885494701666050254987932260860e111,     //76
    1.451830920282858696340707840860e113,     //77
    1.132428117820629783145752115870e115,     //78
    8.946182130782975286851441715400e116,     //79
    7.156945704626380229481153372320e118,     //80
    5.797126020747367985879734231580e120,     //81
    4.753643337012841748421382069890e122,     //82
    3.945523969720658651189747118010e124,     //83
    3.314240134565353266999387579130e126,     //84
    2.817104114380550276949479442260e128,     //85
    2.422709538367273238176552320340e130,     //86
    2.107757298379527717213600518700e132,     //87
    1.854826422573984391147968456460e134,     //88
    1.650795516090846108121691926250e136,     //89
    1.485715964481761497309522733620e138,     //90
    1.352001527678402962551665687590e140,     //91
    1.243841405464130725547532432590e142,     //92
    1.156772507081641574759205162310e144,     //93
    1.087366156656743080273652852570e146,     //94
    1.032997848823905926259970209940e148,     //95
    9.916779348709496892095714015400e149,     //96
    9.619275968248211985332842594960e151,     //97
    9.426890448883247745626185743100e153,     //98
    9.332621544394415268169923885600e155,     //99
    9.33262154439441526816992388563e157,      //100
    9.42594775983835942085162312450e159,      //101
    9.61446671503512660926865558700e161,      //102
    9.90290071648618040754671525458e163,      //103
    1.02990167451456276238485838648e166,      //104
    1.08139675824029090050410130580e168,      //105
    1.146280563734708354534347384148e170,     //106
    1.226520203196137939351751701040e172,     //107
    1.324641819451828974499891837120e174,     //108
    1.443859583202493582204882102460e176,     //109
    1.588245541522742940425370312710e178,     //110
    1.762952551090244663872161047110e180,     //111
    1.974506857221074023536820372760e182,     //112
    2.231192748659813646596607021220e184,     //113
    2.543559733472187557120132004190e186,     //114
    2.925093693493015690688151804820e188,     //115
    3.393108684451898201198256093590e190,     //116
    3.96993716080872089540195962950e192,      //117
    4.68452584975429065657431236281e194,      //118
    5.57458576120760588132343171174e196,      //119
    6.68950291344912705758811805409e198,      //120
    8.09429852527344373968162284545e200,      //121
    9.87504420083360136241157987140e202,      //122
    1.21463043670253296757662432419e205,      //123
    1.50614174151114087979501416199e207,      //124
    1.88267717688892609974376770249e209,      //125
    2.37217324288004688567714730514e211,      //126
    3.01266001845765954480997707753e213,      //127
    3.85620482362580421735677065923e215,      //128
    4.97450422247728744039023415041e217,      //129
    6.46685548922047367250730439554e219,      //130
    8.47158069087882051098456875820e221,      //131
    1.11824865119600430744996307608e224,      //132
    1.48727070609068572890845089118e226,      //133
    1.99294274616151887673732419418e228,      //134
    2.69047270731805048359538766215e230,      //135
    3.65904288195254865768972722052e232,      //136
    5.01288874827499166103492629211e234,      //137
    6.91778647261948849222819828311e236,      //138
    9.61572319694108900419719561353e238,      //139
    1.34620124757175246058760738589e241,      //140
    1.89814375907617096942852641411e243,      //141
    2.69536413788816277658850750804e245,      //142
    3.85437071718007277052156573649e247,      //143
    5.55029383273930478955105466055e249,      //144
    8.04792605747199194484902925780e251,      //145
    1.17499720439091082394795827164e254,      //146
    1.72724589045463891120349865931e256,      //147
    2.55632391787286558858117801578e258,      //148
    3.80892263763056972698595524351e260,      //149
    5.71338395644585459047893286526e262,      //150
    8.62720977423324043162318862650e264,      //151
    1.31133588568345254560672467123e267,      //152
    2.00634390509568239477828874699e269,      //153
    3.08976961384735088795856467036e271,      //154
    4.78914290146339387633577523906e273,      //155
    7.47106292628289444708380937294e275,      //156
    1.17295687942641442819215807155e278,      //157
    1.85327186949373479654360975305e280,      //158
    2.94670227249503832650433950735e282,      //159
    4.71472363599206132240694321176e284,      //160
    7.59070505394721872907517857094e286,      //161
    1.22969421873944943411017892849e289,      //162
    2.00440157654530257759959165344e291,      //163
    3.28721858553429622726333031164e293,      //164
    5.42391066613158877498449501421e295,      //165
    9.00369170577843736647426172359e297,      //166
    1.50361651486499904020120170784e300,      //167
    2.52607574497319838753801886917e302,      //168
    4.26906800900470527493925188890e304,      //169
    7.25741561530799896739672821113e306,      //170
};

////////////////////////////////////////////////////////////////////////////////

double KFMMath::Factorial(int arg)
{
    if (arg < FACTORIAL_TABLE_SIZE && arg >= 0) {
        return factorial_table[arg];
    }
    else {
        return std::numeric_limits<double>::quiet_NaN();
    }
}

double KFMMath::SqrtFactorial(int arg)
{
    double val = 1.0;
    for (int i = 1; i <= arg; i++) {
        val *= KFMSquareRootUtilities::SqrtInteger(i);
    }
    return val;
}

double KFMMath::A_Coefficient(int upper, int lower)
{
    //the following implementation, which doesn't use the factorial function directly
    //allows us to avoid overflow for integers greater than 170
    return (std::pow(-1.0, lower)) /
           (KFMMath::SqrtFactorial(std::abs(lower - upper)) * KFMMath::SqrtFactorial(std::abs(lower + upper)));
}


//returns the Schmidt semi-normalized associated legendre polynomials
double KFMMath::ALP_nm(int n, int m, const double& x)
{
    if ((n < 0) || (n < m) || (std::fabs(x) > 1.0)) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    double norm = 0;

    if (m == 0) {
        norm = KFMSquareRootUtilities::InverseSqrtInteger(2 * n + 1);
    }
    else {
        norm = KFMSquareRootUtilities::InverseSqrtInteger(2 * (2 * n + 1));
        if (m % 2 != 0) {
            norm *= -1.0;
        }
    }

    if (n == m) {
        return norm * ALP_mm(m, x);
    }
    else if (n == m + 1) {
        return norm * x * KFMSquareRootUtilities::SqrtInteger(2 * m + 3) * ALP_mm(m, x);
    }
    else {
        double p_b = ALP_mm(m, x);
        double p_a = x * KFMSquareRootUtilities::SqrtInteger(2 * m + 3) * p_b;
        double plm, alm, blm;
        plm = 0;

        for (int l = m + 2; l <= n; l++) {
            alm = KFMSquareRootUtilities::SqrtIntegerRatio((2 * l - 1) * (2 * l + 1), (l - m) * (l + m));
            blm = KFMSquareRootUtilities::SqrtIntegerRatio((2 * l + 1) * (l + m - 1) * (l - m - 1),
                                                           (l - m) * (l + m) * (2 * l - 3));
            plm = alm * x * p_a - blm * p_b;
            p_b = p_a;
            p_a = plm;
        }

        return norm * plm;
    }
}

//______________________________________________________________________________


void KFMMath::ALP_nm_array(int n_max, const double& x, double* val)
{

    val[0] = 1.0;
    if (n_max == 0) {
        return;
    };

    double u = std::sqrt((1.0 - x) * (1.0 + x));
    val[2] = u * sqrt_of_three;
    val[1] = x * sqrt_of_three;
    if (n_max == 1) {
        //normalize, then return
        val[2] *= -1.0 * KFMSquareRootUtilities::InverseSqrtInteger(6);
        val[1] *= KFMSquareRootUtilities::InverseSqrtInteger(3);
        return;
    };


    //evaluate the base cases for each p(m,m) and p(m, m-1) up to m = n_max
    int l, m, si, si_a, si_b;
    double p_mm = val[2];
    double alm, blm;
    for (m = 2; m <= n_max; m++) {
        //first base case value P(m,m)
        si_a = (m * (m + 1)) / 2 + m;
        p_mm *= u * KFMSquareRootUtilities::SqrtIntegerRatio(2 * m + 1, 2 * m);
        val[si_a] = p_mm;

        //second base case value P(m, m-1)
        si_b = ((m) * (m + 1)) / 2 + m - 1;
        si_a = (m * (m - 1)) / 2 + m - 1;
        val[si_b] = x * KFMSquareRootUtilities::SqrtInteger(2 * (m - 1) + 3) * val[si_a];
    }

    //do reccurance over the rest of the whole array for (l,m) > (1,1)
    for (m = 0; m <= n_max; m++) {
        for (l = m + 2; l <= n_max; l++) {
            si = (l * (l + 1)) / 2 + m;
            si_a = (l * (l - 1)) / 2 + m;
            si_b = ((l - 2) * (l - 1)) / 2 + m;

            alm = KFMSquareRootUtilities::SqrtIntegerRatio((2 * l - 1) * (2 * l + 1), (l - m) * (l + m));
            blm = KFMSquareRootUtilities::SqrtIntegerRatio((2 * l + 1) * (l + m - 1) * (l - m - 1),
                                                           (l - m) * (l + m) * (2 * l - 3));

            val[si] = alm * x * val[si_a] - blm * val[si_b];
        }
    }

    //normalize
    double norm;
    for (l = 0; l <= n_max; l++) {
        for (m = 0; m <= l; m++) {
            if (m == 0) {
                norm = KFMSquareRootUtilities::InverseSqrtInteger(2 * l + 1);
            }
            else {
                norm = KFMSquareRootUtilities::InverseSqrtInteger(2 * (2 * l + 1));
                if (m % 2 != 0) {
                    norm *= -1.0;
                }
            }

            si = (l * (l + 1)) / 2 + m;
            val[si] *= norm;
        }
    }
}


//______________________________________________________________________________

//returns the derivative of the Schmidt semi-normalized associated legendre polynomials
double KFMMath::ALPDerv_nm(int n, int m, const double& x)
{
    if ((n < 0) || (n < m) || (std::fabs(x) > 1.0)) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    if (n == 0) {
        return 0.0;
    }
    else {
        if (n == m) {
            double dm = m;
            double inv_u = 1.0 / std::sqrt((1.0 - x) * (1.0 + x));
            double norm;

            if (m == 0) {
                norm = KFMSquareRootUtilities::InverseSqrtInteger(2 * n + 1);
            }
            else {
                norm = KFMSquareRootUtilities::InverseSqrtInteger(2 * (2 * n + 1));
                if (m % 2 != 0) {
                    norm *= -1.0;  //condon-shortley phase
                }
            }
            return dm * norm * x * inv_u * ALP_mm(m, x);
        }
        else {

            double norm;

            if (m == 0) {
                norm = KFMSquareRootUtilities::InverseSqrtInteger(2 * n + 1);
            }
            else {
                norm = KFMSquareRootUtilities::InverseSqrtInteger(2 * (2 * n + 1));
                if (m % 2 != 0) {
                    norm *= -1.0;  //condon-shortley phase
                }
            }

            double inv_u = 1.0 / std::sqrt((1.0 - x) * (1.0 + x));
            double fnm = KFMSquareRootUtilities::SqrtIntegerRatio((n * n - m * m) * (2 * n + 1), (2 * n - 1));
            double num = n * x * ALP_nm_unormalized(n, m, x) - fnm * ALP_nm_unormalized(n - 1, m, x);
            return norm * num * inv_u;
        }
    }
}

//______________________________________________________________________________

void KFMMath::ALPDerv_nm_array(int n_max, const double& x, double* val)
{
    ALP_nm_unormalized_array(n_max, x, val);

    double u = std::sqrt((1.0 - x) * (1.0 + x));
    double inv_u = 1.0 / u;
    double flm, plm, plm_lm1;
    int si_a, si_b;

    for (int m = 0; m <= n_max; m++) {
        for (int l = n_max; l >= m + 1; l--) {
            si_a = (l * (l + 1)) / 2 + m;
            plm = val[si_a];
            si_b = (l * (l - 1)) / 2 + m;
            plm_lm1 = val[si_b];
            flm = KFMSquareRootUtilities::SqrtIntegerRatio((l * l - m * m) * (2 * l + 1), (2 * l - 1));

            val[si_a] = inv_u * (l * x * plm - flm * plm_lm1);
        }
    }

    //take care of sectoral derivatives last
    for (int m = 0; m <= n_max; m++) {
        int si_a = m * (m + 1) / 2 + m;
        val[si_a] = m * x * inv_u * val[si_a];
    }

    //normalize
    double norm;
    for (int l = 0; l <= n_max; l++) {
        for (int m = 0; m <= l; m++) {
            if (m == 0) {
                norm = KFMSquareRootUtilities::InverseSqrtInteger(2 * l + 1);
            }
            else {
                norm = KFMSquareRootUtilities::InverseSqrtInteger(2 * (2 * l + 1));
                if (m % 2 != 0) {
                    norm *= -1.0;  //condon-shortley phase
                }
            }

            si_a = (l * (l + 1)) / 2 + m;
            val[si_a] *= norm;
        }
    }
}


//______________________________________________________________________________


//base case P_m^m(x), this should not be called directly as it is unnormalized
//use ALP_nm instead
double KFMMath::ALP_mm(int m, const double& x)
{
    if (m == 0) {
        return 1.0;
    }
    else if (m == 1) {
        double u = std::sqrt((1.0 - x) * (1.0 + x));
        return u * KFMSquareRootUtilities::SqrtInteger(3);
    }
    else {
        double p_mm = KFMSquareRootUtilities::SqrtInteger(3);
        double u = std::sqrt((1.0 - x) * (1.0 + x));
        p_mm *= u;
        for (int i = 2; i <= m; i++) {
            p_mm *= u * KFMSquareRootUtilities::SqrtIntegerRatio((2 * i + 1), (2 * i));
        }
        return p_mm;
    }
}

//______________________________________________________________________________

//returns the un-normalized associated legendre polynomials
double KFMMath::ALP_nm_unormalized(int n, int m, const double& x)
{
    if ((n < 0) || (n < m) || (std::fabs(x) > 1.0)) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    if (n == m) {
        return ALP_mm(m, x);
    }
    else if (n == m + 1) {
        return x * KFMSquareRootUtilities::SqrtInteger(2 * m + 3) * ALP_mm(m, x);
    }
    else {
        double p_b = ALP_mm(m, x);
        double p_a = x * KFMSquareRootUtilities::SqrtInteger(2 * m + 3) * p_b;
        double plm, alm, blm;
        plm = 0;

        for (int l = m + 2; l <= n; l++) {
            alm = KFMSquareRootUtilities::SqrtIntegerRatio((2 * l - 1) * (2 * l + 1), (l - m) * (l + m));
            blm = KFMSquareRootUtilities::SqrtIntegerRatio((2 * l + 1) * (l + m - 1) * (l - m - 1),
                                                           (l - m) * (l + m) * (2 * l - 3));
            plm = alm * x * p_a - blm * p_b;
            p_b = p_a;
            p_a = plm;
        }

        return plm;
    }
}

//______________________________________________________________________________

void KFMMath::ALP_nm_unormalized_array(int n_max, const double& x, double* val)
{
    val[0] = 1.0;
    if (n_max == 0) {
        return;
    };

    double u = std::sqrt((1.0 - x) * (1.0 + x));
    val[2] = u * sqrt_of_three;
    val[1] = x * sqrt_of_three;
    if (n_max == 1) {
        return;
    };


    //evaluate the base cases for each p(m,m) and p(m, m-1) up to m = n_max
    int l, m, si, si_a, si_b;
    double p_mm = val[2];
    double alm, blm;
    for (m = 2; m <= n_max; m++) {
        //first base case value P(m,m)
        si_a = (m * (m + 1)) / 2 + m;
        p_mm *= u * KFMSquareRootUtilities::SqrtIntegerRatio((2 * m + 1), (2 * m));
        val[si_a] = p_mm;

        //second base case value P(m, m-1)
        si_b = ((m) * (m + 1)) / 2 + m - 1;
        si_a = (m * (m - 1)) / 2 + m - 1;
        val[si_b] = x * KFMSquareRootUtilities::SqrtInteger(2 * (m - 1) + 3) * val[si_a];
    }

    //do reccurance over the rest of the whole array for (l,m) > (1,1)
    for (m = 0; m <= n_max; m++) {
        for (l = m + 2; l <= n_max; l++) {
            si = (l * (l + 1)) / 2 + m;
            si_a = (l * (l - 1)) / 2 + m;
            si_b = ((l - 2) * (l - 1)) / 2 + m;

            alm = KFMSquareRootUtilities::SqrtIntegerRatio(((2 * l - 1) * (2 * l + 1)), ((l - m) * (l + m)));
            blm = KFMSquareRootUtilities::SqrtIntegerRatio(((2 * l + 1) * (l + m - 1) * (l - m - 1)),
                                                           ((l - m) * (l + m) * (2 * l - 3)));

            val[si] = alm * x * val[si_a] - blm * val[si_b];
        }
    }
}


//______________________________________________________________________________

void KFMMath::ALPAndFirstDerv_array(int n_max, const double& x, double* PlmVal, double* PlmDervVal)
{

    ALP_nm_unormalized_array(n_max, x, PlmVal);

    double u = std::sqrt((1.0 - x) * (1.0 + x));
    double inv_u;

    inv_u = 1.0 / u;

    double flm, plm, plm_lm1;
    int si_a, si_b;

    for (int m = 0; m <= n_max; m++) {
        for (int l = n_max; l >= m + 1; l--) {
            si_a = (l * (l + 1)) / 2 + m;
            plm = PlmVal[si_a];
            si_b = (l * (l - 1)) / 2 + m;
            plm_lm1 = PlmVal[si_b];
            flm = KFMSquareRootUtilities::SqrtIntegerRatio(((l * l - m * m) * (2 * l + 1)), (2 * l - 1));
            PlmDervVal[si_a] = inv_u * (l * x * plm - flm * plm_lm1);
        }
    }

    //take care of sectoral derivatives last
    for (int m = 0; m <= n_max; m++) {
        double dm = m;
        int si_a = m * (m + 1) / 2 + m;
        PlmDervVal[si_a] = dm * x * inv_u * PlmVal[si_a];
    }

    //normalize
    double norm;
    for (int l = 0; l <= n_max; l++) {
        for (int m = 0; m <= l; m++) {
            if (m == 0) {
                norm = KFMSquareRootUtilities::InverseSqrtInteger(2 * l + 1);
            }
            else {
                norm = KFMSquareRootUtilities::InverseSqrtInteger(2 * (2 * l + 1));
                if (m % 2 != 0) {
                    norm *= -1.0;  //condon-shortley phase
                }
            }

            si_a = (l * (l + 1)) / 2 + m;
            PlmVal[si_a] *= norm;
            PlmDervVal[si_a] *= norm;
        }
    }
}


//______________________________________________________________________________

double KFMMath::I_secn(int n, double lower_limit, double upper_limit)
{
    if (n <= 0) {
        return 0;
    }

    if (n == 1) {
        double numer = std::fabs(std::tan(upper_limit) + (1.0) / std::cos(upper_limit));
        double denom = std::fabs(std::tan(lower_limit) + (1.0) / std::cos(lower_limit));
        return std::log(numer / denom);
    }

    if (n == 2) {

        double sec_up = (1.0 / std::cos(upper_limit));
        double sec_low = (1.0 / std::cos(lower_limit));
        double sin_diff = std::sin(upper_limit - lower_limit);
        return sec_up * sec_low * sin_diff;
    }

    double sec_up = (1.0 / std::cos(upper_limit));
    double sec_low = (1.0 / std::cos(lower_limit));

    double tan_up = std::tan(upper_limit);
    double tan_low = std::tan(lower_limit);

    double a, b, up, low, result;

    if (n % 2 == 0)  // only even powers involved
    {
        double s_2 = tan_up - tan_low;
        b = s_2;
        for (int i = 2; i < n;) {
            up = tan_up * (std::pow(sec_up, (double) (i))) * (1.0 / ((double) (i) + 1.0));
            low = tan_low * (std::pow(sec_low, (double) (i))) * (1.0 / ((double) (i) + 1.0));
            a = up - low;
            result = a + (((double) (i)) / ((double) (i) + 1.0)) * b;
            b = result;
            i += 2;
        }
    }
    else  //only odd powers involved
    {
        up = std::log(std::fabs(tan_up + sec_up));
        low = std::log(std::fabs(tan_low + sec_low));
        double s_1 = up - low;
        b = s_1;
        for (int i = 1; i < n;) {
            up = tan_up * (std::pow(sec_up, (double) (i))) * (1.0 / ((double) (i) + 1.0));
            low = tan_low * (std::pow(sec_low, (double) (i))) * (1.0 / ((double) (i) + 1.0));
            a = up - low;
            result = a + (((double) (i)) / ((double) (i) + 1.0)) * b;
            b = result;
            i += 2;
        }
    }

    return result;
}


//______________________________________________________________________________


void KFMMath::I_secn_array(int n_max, double lower_limit, double upper_limit, double* val)
{
    if (n_max < 0) {
        return;  //n_max not valid!
    }

    //n=0 case
    val[0] = 0;
    if (n_max == 0) {
        return;
    };

    double up, low;
    double sec_up = (1.0 / std::cos(upper_limit));
    double sec_low = (1.0 / std::cos(lower_limit));
    double tan_up = std::tan(upper_limit);
    double tan_low = std::tan(lower_limit);

    //n=1 case
    double numer = std::fabs(tan_up + sec_up);
    double denom = std::fabs(tan_low + sec_up);
    val[1] = std::log(numer / denom);
    if (n_max == 1) {
        return;
    };

    //n=2 case
    double sin_diff = std::sin(upper_limit - lower_limit);
    val[2] = sec_up * sec_low * sin_diff;
    if (n_max == 2) {
        return;
    };

    for (int n = 1; n <= n_max - 2; n++) {
        up = tan_up * (std::pow(sec_up, (double) (n))) * (1.0 / ((double) (n) + 1.0));
        low = tan_low * (std::pow(sec_low, (double) (n))) * (1.0 / ((double) (n) + 1.0));
        val[n + 2] = (up - low) + (((double) (n)) / ((double) (n) + 1.0)) * val[n];
    }
}


//______________________________________________________________________________


double KFMMath::I_trig1(int n, double lower_limit, double upper_limit)
{
    if (n <= 0) {
        return 0;
    }

    if (n == 1) {
        double cos_up = std::cos(upper_limit);
        double cos_low = std::cos(lower_limit);
        return std::log(cos_low / cos_up);
    }

    //12/7/13, removed an erroneous factor of -1
    return (1.0 / ((double) n - 1.0)) *
           (std::pow(std::cos(upper_limit), -1 * n + 1) - std::pow(std::cos(lower_limit), -1 * n + 1));
}

void KFMMath::I_trig1_array(int n_max, double lower_limit, double upper_limit, double* val)
{
    if (n_max < 0) {
        return;  //n_max not valid!
    }

    val[0] = 0;
    if (n_max == 0) {
        return;
    };

    double cos_up = std::cos(upper_limit);
    double cos_low = std::cos(lower_limit);

    val[1] = std::log(cos_low / cos_up);
    if (n_max == 1) {
        return;
    };

    double p;
    for (int n = 2; n <= n_max; n++) {
        p = -1 * n + 1;
        val[n] = (1.0 / ((double) n - 1.0)) * (std::pow(cos_up, p) - std::pow(cos_low, p));
    }
}


//______________________________________________________________________________


double KFMMath::I_cheb1(int l, int m, double lower_limit, double upper_limit)
{
    if (m < 0 || l < 0) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    if (m == 0 && l >= 0) {
        return I_secn(l + 2, lower_limit, upper_limit);
    }

    if (m == 1 && l >= 0) {
        return I_secn(l + 1, lower_limit, upper_limit);
    }

    int size = ((l + 1) * (l + 2)) / 2;
    auto* val = new double[size];

    val[0] = I_secn(2, lower_limit, upper_limit);  //(0,0)

    int si, si_m1, si_a, si_b;

    for (int j = 1; j <= l; j++) {
        si = j * (j + 1) / 2;
        si_m1 = j * (j - 1) / 2;
        val[si] = I_secn(j + 2, lower_limit, upper_limit);  //(j,0) base case
        val[si + 1] = val[si_m1];                           //(j,1) base case

        for (int k = 2; k <= j; k++) {
            si = j * (j + 1) / 2 + k;
            si_a = j * (j - 1) / 2 + k - 1;
            si_b = j * (j + 1) / 2 + k - 2;
            val[si] = 2.0 * val[si_a] - val[si_b];

            if (j == l && m == k) {
                break;
            };
        }
    }

    si = l * (l + 1) / 2 + m;
    double temp = val[si];

    delete[] val;

    return temp;
}


//______________________________________________________________________________


void KFMMath::I_cheb1_array(int l_max, double lower_limit, double upper_limit, double* val)
{
    if (l_max < 0) {
        return;
    }

    val[0] = I_secn(2, lower_limit, upper_limit);  //(0,0)
    if (l_max == 0) {
        return;
    };

    int si, si_m1, si_a, si_b;

    for (int j = 1; j <= l_max; j++) {
        si = j * (j + 1) / 2;
        si_m1 = j * (j - 1) / 2;
        val[si] = I_secn(j + 2, lower_limit, upper_limit);  //(j,0) base case
        val[si + 1] = val[si_m1];                           //(j,1) base case

        for (int k = 2; k <= j; k++) {
            si = j * (j + 1) / 2 + k;
            si_a = j * (j - 1) / 2 + k - 1;
            si_b = j * (j + 1) / 2 + k - 2;
            val[si] = 2.0 * val[si_a] - val[si_b];
        }
    }
}

//______________________________________________________________________________

void KFMMath::I_cheb1_array_fast(int l_max, double lower_limit, double upper_limit, double* scratch, double* val)
{
    if (l_max < 0) {
        return;
    }

    I_secn_array(l_max + 2, lower_limit, upper_limit, scratch);

    val[0] = scratch[2];  //I_secn( 2, lower_limit, upper_limit); //(0,0)
    if (l_max == 0) {
        return;
    };

    int si, si_m1, si_a, si_b;

    for (int j = 1; j <= l_max; j++) {
        si = j * (j + 1) / 2;
        si_m1 = j * (j - 1) / 2;
        val[si] = scratch[j + 2];  // I_secn( j+2, lower_limit, upper_limit);   //(j,0) base case
        val[si + 1] = val[si_m1];  //(j,1) base case

        for (int k = 2; k <= j; k++) {
            si = j * (j + 1) / 2 + k;
            si_a = j * (j - 1) / 2 + k - 1;
            si_b = j * (j + 1) / 2 + k - 2;
            val[si] = 2.0 * val[si_a] - val[si_b];
        }
    }
}

//______________________________________________________________________________


double KFMMath::I_cheb2(int l, int m, double lower_limit, double upper_limit)
{
    if (m < 0 || l < 0) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    if (m == 0 && l >= 0) {
        return 0;
    }

    if (m == 1 && l >= 0) {
        return I_trig1(l + 2, lower_limit, upper_limit);
    }

    int size = ((l + 1) * (l + 2)) / 2;
    auto* val = new double[size];

    val[0] = 0;  //(0,0)

    int si, si_a, si_b;

    for (int j = 1; j <= l; j++) {
        si = j * (j + 1) / 2;
        val[si] = 0;                                             //(j,0) base case
        val[si + 1] = I_trig1(j + 2, lower_limit, upper_limit);  //(j,1) base case

        for (int k = 2; k <= j; k++) {
            si = j * (j + 1) / 2 + k;
            si_a = j * (j - 1) / 2 + k - 1;
            si_b = j * (j + 1) / 2 + k - 2;
            val[si] = 2.0 * val[si_a] - val[si_b];
            if (j == l && m == k) {
                break;
            };
        }
    }

    si = l * (l + 1) / 2 + m;
    double temp = val[si];

    delete[] val;

    return temp;
}


//______________________________________________________________________________


void KFMMath::I_cheb2_array(int l_max, double lower_limit, double upper_limit, double* val)
{
    if (l_max < 0) {
        return;
    }

    val[0] = 0;  //(0,0)
    if (l_max == 0) {
        return;
    };

    int si, si_a, si_b;

    for (int j = 1; j <= l_max; j++) {
        si = j * (j + 1) / 2;
        val[si] = 0;                                             //(j,0) base case
        val[si + 1] = I_trig1(j + 2, lower_limit, upper_limit);  //(j,1) base case

        for (int k = 2; k <= j; k++) {
            si = j * (j + 1) / 2 + k;
            si_a = j * (j - 1) / 2 + k - 1;
            si_b = j * (j + 1) / 2 + k - 2;
            val[si] = 2.0 * val[si_a] - val[si_b];
        }
    }
}


//______________________________________________________________________________


double KFMMath::K_norm(int l, int m, double h)
{
    double plm = ALP_nm(l, std::abs(m), 0.);
    return (1.0 / ((double) l + 2.0)) * std::pow(h, l + 2) * plm;
}


void KFMMath::K_norm_array(int l_max, double h, double* val)
{
    double hpow = h;
    double plm;
    double fac;

    for (int l = 0; l <= l_max; l++) {
        hpow *= h;
        fac = (1.0 / ((double) l + 2.0));
        for (int m = 0; m <= l; m++) {
            plm = ALP_nm(l, m, 0.);
            val[((l) * (l + 1)) / 2 + m] = fac * hpow * plm;
        }
    }
}


void KFMMath::K_norm_array(int l_max, double h, const double* plm, double* val)
{
    double hpow = h;
    double fac;
    int si;

    for (int l = 0; l <= l_max; l++) {
        hpow *= h;
        fac = (1.0 / ((double) l + 2.0));
        for (int m = 0; m <= l; m++) {
            si = (l * (l + 1)) / 2 + m;
            val[si] = fac * hpow * plm[si];
        }
    }
}


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////


void KFMMath::SphericalHarmonic_Cart(int l, int m, const double* cartesian_coords, double* result)
{
    double cosTheta = KFMMath::CosTheta(cartesian_coords);
    double phi = KFMMath::Phi(cartesian_coords);
    double plm = KFMMath::ALP_nm(l, std::abs(m), cosTheta);
    result[0] = std::cos(m * phi) * plm;
    result[1] = std::sin(m * phi) * plm;
}

void KFMMath::RegularSolidHarmonic_Cart(int l, int m, const double* cartesian_coords, double* result)
{
    KFMMath::SphericalHarmonic_Cart(l, m, cartesian_coords, result);
    double radiusPower = std::pow(KFMMath::Radius(cartesian_coords), l);
    result[0] *= radiusPower;
    result[1] *= radiusPower;
}

void KFMMath::IrregularSolidHarmonic_Cart(int l, int m, const double* cartesian_coords, double* result)
{
    KFMMath::SphericalHarmonic_Cart(l, m, cartesian_coords, result);
    double radiusPower = std::pow(KFMMath::Radius(cartesian_coords), -1. * (l + 1.));
    result[0] *= radiusPower;
    result[1] *= radiusPower;
}

std::complex<double> KFMMath::SphericalHarmonic_Sph(int l, int m, const double* spherical_coords)
{
    double cosTheta = std::cos(spherical_coords[1]);
    double phi = spherical_coords[2];
    double plm = KFMMath::ALP_nm(l, std::abs(m), cosTheta);
    return std::complex<double>(std::cos(m * phi) * plm, std::sin(m * phi) * plm);
}

std::complex<double> KFMMath::SphericalHarmonic_Cart(int l, int m, const double* cartesian_coords)
{
    double cosTheta = KFMMath::CosTheta(cartesian_coords);
    double phi = KFMMath::Phi(cartesian_coords);
    double plm = KFMMath::ALP_nm(l, std::abs(m), cosTheta);
    return std::complex<double>(std::cos(m * phi) * plm, std::sin(m * phi) * plm);
}

std::complex<double> KFMMath::RegularSolidHarmonic_Cart(int l, int m, const double* cartesian_coords)
{
    std::complex<double> val = KFMMath::SphericalHarmonic_Cart(l, m, cartesian_coords);
    val *= std::pow(KFMMath::Radius(cartesian_coords), l);
    return val;
}

std::complex<double> KFMMath::IrregularSolidHarmonic_Cart(int l, int m, const double* cartesian_coords)
{
    std::complex<double> val = KFMMath::SphericalHarmonic_Cart(l, m, cartesian_coords);
    val *= std::pow(KFMMath::Radius(cartesian_coords), -1. * (l + 1.));
    return val;
}

//______________________________________________________________________________


void KFMMath::RegularSolidHarmonic_Cart_Array(int n_max, const double* cartesian_coords, std::complex<double>* result)
{
    unsigned int max_size = (n_max * (n_max + 1)) / 2 + n_max + 1;

    double cosTheta = KFMMath::CosTheta(cartesian_coords);
    double phi = KFMMath::Phi(cartesian_coords);
    double r = KFMMath::Radius(cartesian_coords);

    //compute the array of powers of r
    double r_pow[n_max + 1];
    r_pow[0] = 1.0;
    for (int i = 1; i <= n_max; i++) {
        r_pow[i] = r * r_pow[i - 1];
    }

    //compute alp array
    double plm[max_size];
    KFMMath::ALP_nm_array(n_max, cosTheta, plm);

    //compute the array of cos(m*x) and sin(m*x)
    double cos_vec[n_max + 1];
    double sin_vec[n_max + 1];

    //intial values needed for recursion
    double sin = std::sin(phi);
    double sin_over_two = std::sin(phi / 2.0);
    double eta_real = -2.0 * sin_over_two * sin_over_two;
    double eta_imag = sin;

    //space to store last value
    double real = 1.0;
    double imag = 0.0;

    cos_vec[0] = 1.0;
    sin_vec[0] = 0.0;

    //scratch space
    double u, v, mag2, delta;

    for (int i = 1; i <= n_max; i++) {
        u = real + eta_real * real - eta_imag * imag;
        v = imag + eta_imag * real + eta_real * imag;

        //re-scale to correct round off errors
        mag2 = u * u + v * v;
        delta = 1.0 / std::sqrt(mag2);

        u *= delta;
        v *= delta;

        real = u;
        imag = v;

        cos_vec[i] = real;
        sin_vec[i] = imag;
    }

    for (int n = 0; n <= n_max; n++) {
        for (int m = 0; m <= n; m++) {
            result[n * (n + 1) + m] = std::complex<double>(r_pow[n] * cos_vec[m], r_pow[n] * sin_vec[m]);
            result[n * (n + 1) + m] *= plm[(n * (n + 1)) / 2 + m];
            result[n * (n + 1) - m] = std::conj(result[n * (n + 1) + m]);
        }
    }
}


//______________________________________________________________________________


void KFMMath::IrregularSolidHarmonic_Cart_Array(int n_max, const double* cartesian_coords, std::complex<double>* result)
{
    unsigned int max_size = (n_max * (n_max + 1)) / 2 + n_max + 1;

    double cosTheta = KFMMath::CosTheta(cartesian_coords);
    double phi = KFMMath::Phi(cartesian_coords);
    double inv_r = 1.0 / KFMMath::Radius(cartesian_coords);

    //compute the array of powers of r
    double r_pow[n_max + 1];
    r_pow[0] = inv_r;
    for (int i = 1; i <= n_max; i++) {
        r_pow[i] = inv_r * r_pow[i - 1];
    }

    //compute alp array
    double plm[max_size];
    KFMMath::ALP_nm_array(n_max, cosTheta, plm);

    //compute the array of cos(m*x) and sin(m*x)
    double cos_vec[n_max + 1];
    double sin_vec[n_max + 1];

    //intial values needed for recursion
    double sin = std::sin(phi);
    double sin_over_two = std::sin(phi / 2.0);
    double eta_real = -2.0 * sin_over_two * sin_over_two;
    double eta_imag = sin;

    //space to store last value
    double real = 1.0;
    double imag = 0.0;

    cos_vec[0] = 1.0;
    sin_vec[0] = 0.0;

    //scratch space
    double u, v, mag2, delta;

    for (int i = 1; i <= n_max; i++) {
        u = real + eta_real * real - eta_imag * imag;
        v = imag + eta_imag * real + eta_real * imag;

        //re-scale to correct round off errors
        mag2 = u * u + v * v;
        delta = 1.0 / std::sqrt(mag2);

        u *= delta;
        v *= delta;

        real = u;
        imag = v;

        cos_vec[i] = real;
        sin_vec[i] = imag;
    }

    for (int n = 0; n <= n_max; n++) {
        for (int m = 0; m <= n; m++) {
            result[n * (n + 1) + m] = std::complex<double>(r_pow[n] * cos_vec[m], r_pow[n] * sin_vec[m]);
            result[n * (n + 1) + m] *= plm[(n * (n + 1)) / 2 + m];
            result[n * (n + 1) - m] = std::conj(result[n * (n + 1) + m]);
        }
    }
}


}  // namespace KEMField
