#include <gsl/gsl_monte_miser.h>
#include <gsl/gsl_monte_vegas.h>
#include <cmath>
#include <iostream>
#include "LHAPDF/LHAPDF.h"
#include <string.h>
#include <fstream>
#include <chrono>
#include <regex>

using namespace LHAPDF;
using namespace std;

// #define PI 3.14159265
#define GeVtofb 0.389379338e+12
#define Mtop 172.76 // top quark mass

// set collider energy and luminosity
#define energy 13000
// #define energy 8000
// #define lum 300

// define acceptance cuts
#define eta_cut 2.5
#define pT_cut 25

// define rapidity cuts
#define y_min 0.0
#define y_max 0.0 // y_max = 0 means no upper cut

// // SM parameters
// #define MW 80.425
// #define GammaW 2.124

// SM parameters for validation against Stefano
#define MW 79.9571
#define GammaW 2.01257

#define MZ 91.1876

#define stheta2W (1.0 - pow(MW/MZ,2))

// #define GF 1.1663787 * pow(10,-5)

// #define alphaEM GF * sqrt(2) * pow(MW,2) * stheta2W / PI


// // SM parameters as in ATLAS and PDG
// #define MW 80.370
// #define GammaW 2.085

#define MWprime1 3000
#define MWprime2 4000

#define alphaEM 1 / 128.0
// #define stheta2W pow(0.4817,2)

const double gsm = sqrt(4*M_PI*alphaEM/stheta2W);

// const double gsm = sqrt(4*PI*alphaEM/stheta2W);
// const double gsm = sqrt(4*PI*alphaEM/stheta2W) * sqrt((1 + pow(stheta2W,2))/(1 - stheta2W));
// const double gsmprime = 3 * sqrt(4*PI*alphaEM/stheta2W) * sqrt((1 + pow(stheta2W,2))/(1 - stheta2W));

const double Vud = 0.975;
const double Vus = 0.222;
const double Vub = 0.0;
const double Vcd = 0.222;
const double Vcs = 0.975;
const double Vcb = 0.0;
const double Vtd = 0.0;
const double Vts = 0.0;
const double Vtb = 1.0;


// partial decay width into lepton is (g^2 MW / 48*PI)
// This is multiplied for the 3 generations of leptons
// The same for the three generations of quarks * their color factor 3 -> 3*3 = 9
// So in total is to be multiplied by 3 + 9 = 12
// const double GammaWprime = 12 * pow(gsmprime,2) * MWprime / (48*PI);
// const double GammaWprime = 85;


// // Transverse mass range and number of points
// const double Mt_min = 2000;
// const double Mt_max = 8000;
// const double step = 50;
// const int points_Mt = (Mt_max - Mt_min) / step;

// Transverse mass range and number of points
const double Mt_min = 50;
const double Mt_max = 11000;
const double step = 50;
const int points_Mt = (Mt_max - Mt_min) / step;


// PDF set and grid
// #define setname "MRST2004qed_proton"
// #define setname "CT10"

#define setname "CT18NNLO68cl"
// #define setname "CT18NNLO68cl_AFB_300"
// #define setname "CT18NNLO68cl_AW_300"
// #define setname "CT18NNLO68cl_AFBonAW_300"
// #define setname "CT18NNLO68cl_AFB_3000"
// #define setname "CT18NNLO68cl_AW_3000"
// #define setname "CT18NNLO68cl_AFBonAW_3000"

// #define setname "CT18NNLO68cl_AFB_300_hm"
// #define setname "CT18NNLO68cl_AW_300_hm"
// #define setname "CT18NNLO68cl_AFBonAW_300_hm"
// #define setname "CT18NNLO68cl_AFB_3000_hm"
// #define setname "CT18NNLO68cl_AW_3000_hm"
// #define setname "CT18NNLO68cl_AFBonAW_3000_hm"

const LHAPDF::PDFSet set(setname);
// const size_t nmem = set.size()-1;
const size_t nmem = 0;

const vector<LHAPDF::PDF*> pdf = set.mkPDFs();

// Setting of the integration
const int dim_integration = 2; // Integration in yreduced, eta

// Integration parameters
const int iterations_warmup = 5;
const double calls_warmupfactor = 0.1;
const int iterations_main = 3;
// Integration precision
const double epsrel = 1e-3;
const double epsabs = 1e-12;
const int max_eval = 35000;

struct Parameters {
    int max_iters = 10;
    int PDF_set = 0;
    double Mt;
};


// This function uses the GSL Monte Vegas integration routine
void Integration(double (*IF)(double *x, size_t dim, void *jj), int ndim,
                 int maxeval, double epsrel, double epsabs, double &res,
                 double &err, void *userdata, double calls_warmupfactor = 0.1,
                 int iterations_warmup = 5, int iterations_main = 3) {

    // gsl_vegas
    // Selects the integrand
    size_t calls = 0;
    gsl_monte_function I;
    I.f = IF;
    I.dim = ndim;
    I.params = userdata;
    calls = maxeval; // set number of calls
    const size_t dnum = I.dim;

    // Integration limits [0:1]
    double xmin[dnum], xmax[dnum];
    for (size_t i0 = 0; i0 < dnum; i0++) {
        xmin[i0] = 0.0;
        xmax[i0] = 1.0;
    }

    // Initializes the integration routine
    const gsl_rng_type *T;
    gsl_rng *r;
    gsl_rng_env_setup();
    T = gsl_rng_default;
    // T = gsl_rng_mt19937;
    r = gsl_rng_alloc(T);

    gsl_monte_vegas_state *s = gsl_monte_vegas_alloc(dnum);

    // Parameter block
    // make adjustments if you want to change some parameters
    gsl_monte_vegas_params params;
    gsl_monte_vegas_params_get(s, &params);
    params.alpha = 1.5;
    params.mode = GSL_VEGAS_MODE_IMPORTANCE_ONLY;
    gsl_monte_vegas_params_set(s, &params);

    // -1 -> no output; 0 -> summary information; 1 and 2 -> prints much more information
    s->verbose = -1;

    // Integration warm-up
    // (stage = 0 which begins with a new uniform grid and empty weighted average)
    s->stage = 0;
    s->iterations = iterations_warmup;
    gsl_monte_vegas_integrate(&I, xmin, xmax, dnum, calls * calls_warmupfactor, r, s, &res, &err);

    // Integrates
    // Calling VEGAS with stage = 1 retains the grid from the previous run but discards the weighted average, so that one can “tune” the grid using a relatively small number of points and then do a large run with stage = 1
    s->stage = 1;
    s->iterations = iterations_main;
    gsl_monte_vegas_integrate(&I, xmin, xmax, dnum, calls, r, s, &res, &err);

    // Convergence step
    // perform more iterations (<= maxiters) to improve convergence tries to achieve an appropriate chisquared value
    double prec = abs(err / res);

    for (int counter = 1; counter <= ((Parameters *)userdata)->max_iters > 0; counter++) {
        // relative precision does not make any sense if the result is 0
        if (abs(res) < 1.0e-15 || (prec < epsrel && fabs(gsl_monte_vegas_chisq(s) - 1.0) < 0.5) || prec < epsrel / 2.5) {
            break;
        }
        s->iterations = 1;
        s->stage = 3;

        gsl_monte_vegas_integrate(&I, xmin, xmax, dnum, calls, r, s, &res, &err);

        prec = abs(err / res);
    }

    gsl_monte_vegas_free(s);
    gsl_rng_free(r);
    
}

double GammaWprime (double coupling, double Mass) {
    
    double partial_q, partial_lep;
    double partial_top = 0;
    
    partial_lep = 3 * pow(coupling,2) * Mass / (48 * M_PI); // factor 3 from 3 lepton flavours
    
    partial_q = 3 * 2 * pow(coupling,2) * Mass / (48 * M_PI); // factor 2 from 2 quark flavours (top excluded), factor 3 from 3 colors
    
    if (Mass > 2 * Mtop) partial_top = 3 * pow(coupling,2) * Mass / (48 * M_PI) * (1 + 2 * pow(Mtop/Mass, 2)) * sqrt(1 - 4 * pow(Mtop, 2) / pow(Mass, 2)); // factor 3 from 3 colors
    
    return partial_lep + partial_q + partial_top;
}

double jac (double &yreduced, double &etareduced, double *x, Parameters *params)
{
    double Mt = params->Mt;
    double djac = 1.0;
    
    double yreducedmin = -1.0;
    double yreducedmax = 1.0;    
    // Linear sampling
    yreduced = (yreducedmax - yreducedmin) * x[0] + yreducedmin;
    djac *= yreducedmax - yreducedmin;  
    
    
    double etareducedmin = -1.0;
    double etareducedmax = 1.0;
    // Linear sampling
    etareduced = (etareducedmax - etareducedmin) * x[1] + etareducedmin;
    djac *= etareducedmax - etareducedmin;
    
    
    double eta = etareduced / (1 - pow(etareduced,2));
    double costheta = cos(2 * atan(exp(-eta)));
    double pT = Mt / 2;
    double Minv = 2 * pT / sqrt(1 - pow(costheta,2));
    
    // Acceptance cuts cuts
    double z = pow(Minv,2)/pow(energy,2);
    double y = -(1.0/2.0)*log(z)*(yreduced);
    
    if ((eta + y < -eta_cut) || (eta + y > eta_cut)) djac = 0.0; // This should be the correct cut
    
    if (pT < pT_cut) djac = 0.0;
    
//     // Here one can implement rapidity cuts (but remember that now the integration in yreduced is betweem [-1,1])
//     if ((y < y_min) || (y > y_max)) djac = 0.0;
       
    return djac;
}

// U-DBAR Matrix element (W+)
double udbar_funct (double *x, size_t dim, void *jj)
{
    
    (void)(dim); /* avoid unused parameter warnings */
    
    // Read paramters from structure
    Parameters *params = (Parameters *)jj;
    int PDF_set = (params->PDF_set);
    double Mt = (params->Mt);
    
    // Integration variables
    double yreduced, etareduced;
    double djac = jac(yreduced, etareduced, x, params);
    
    
    // eta
    double eta = etareduced / (1 - pow(etareduced,2));
    // costheta
    double costheta = cos(2 * atan(exp(-eta)));
    // Tansverse momentum (assuming \deltaphi = PI)
    double pT = Mt / 2;
    // Invariant mass
    double Minv = 2 * pT / sqrt(1 - pow(costheta,2));
    
    
    // Partonic cross section parameters
    double Q = Minv;
    double z = pow(Minv,2)/pow(energy,2);
    double y = -(1.0/2.0)*log(z)*(yreduced);
    double x1 = sqrt(z)*exp(y);
    double x2 = sqrt(z)*exp(-y);
    
    // Check on invariant mass
    if (Minv > energy) return 0;
    // Check on rapidity
//     if ( (y < y_min) || (y > y_max and y_max != 0)) return 0;
    
    double dsigma = GeVtofb / (768 * pow(M_PI,2)) * pow(Minv,2) / 4 * pow(1 + costheta,2); // Born matrix element squared without propagator
    
    double GammaWprime1 = GammaWprime(gsm, MWprime1);
    double GammaWprime2 = GammaWprime(gsm, MWprime2);
        
    double propagator = 0.0;
    propagator += pow(gsm,4) * (1 / (pow(pow(Minv,2) - pow(MW,2),2) + pow(GammaW,2) * pow(MW,2))); // W^2
    propagator += pow(gsm,4) * (1 / (pow(pow(Minv,2) - pow(MWprime1,2),2) + pow(GammaWprime1,2) * pow(MWprime1,2))); // W'^2
    propagator += pow(gsm,4) * (1 / (pow(pow(Minv,2) - pow(MWprime2,2),2) + pow(GammaWprime2,2) * pow(MWprime2,2))); // W''^2
    propagator += pow(gsm,2) * pow(gsm,2) * (2 * ((pow(Minv,2) - pow(MW,2)) * ((pow(Minv,2) - pow(MWprime1,2))) + MW * GammaW * MWprime1 * GammaWprime1) / (((pow(pow(Minv,2) - pow(MW,2),2) + pow(GammaW,2) * pow(MW,2))) * ((pow(pow(Minv,2) - pow(MWprime1,2),2) + pow(GammaWprime1,2) * pow(MWprime1,2))))); // W - W'
    propagator += pow(gsm,2) * pow(gsm,2) * (2 * ((pow(Minv,2) - pow(MW,2)) * ((pow(Minv,2) - pow(MWprime2,2))) + MW * GammaW * MWprime2 * GammaWprime2) / (((pow(pow(Minv,2) - pow(MW,2),2) + pow(GammaW,2) * pow(MW,2))) * ((pow(pow(Minv,2) - pow(MWprime2,2),2) + pow(GammaWprime2,2) * pow(MWprime2,2))))); // W - W''
    propagator += pow(gsm,2) * pow(gsm,2) * (2 * ((pow(Minv,2) - pow(MWprime1,2)) * ((pow(Minv,2) - pow(MWprime2,2))) + MWprime1 * GammaWprime1 * MWprime2 * GammaWprime2) / (((pow(pow(Minv,2) - pow(MWprime1,2),2) + pow(GammaWprime1,2) * pow(MWprime1,2))) * ((pow(pow(Minv,2) - pow(MWprime2,2),2) + pow(GammaWprime2,2) * pow(MWprime2,2))))); // W' - W''
    
    dsigma *= propagator; // Multiply by propagator
    dsigma *= 1 / pow(energy,2); // Jacobian x1,x2 -> Minv^2, Y
    dsigma *= -0.5 * log(z); // Jacobian Y, Yreduced
    dsigma *= 4 / (1 - pow(costheta,2)); // Jacobian Minv^2 -> pT^2
    dsigma *= 2 * pT; // Jacobian pT^2 -> pT
    dsigma *= 0.5; // Jacobian pT -> MT
    dsigma *= 2 * M_PI; // Integration in phi
    dsigma *= (1 - pow(costheta,2)); // Jacobian costheta -> eta
    dsigma *= (1 + pow(etareduced,2)) / pow(1 - pow(etareduced,2),2); // Jacobian eta -> etareduced
    
    // Partons PDF
    double f1u = (pdf[PDF_set]->xfxQ(2, x1, Q))/x1;
    double f1c = (pdf[PDF_set]->xfxQ(4, x1, Q))/x1;
    double f2dbar = (pdf[PDF_set]->xfxQ(-1, x2, Q))/x2;
    double f2sbar = (pdf[PDF_set]->xfxQ(-3, x2, Q))/x2;
    double f2u = (pdf[PDF_set]->xfxQ(2, x2, Q))/x2;
    double f2c = (pdf[PDF_set]->xfxQ(4, x2, Q))/x2;
    double f1dbar = (pdf[PDF_set]->xfxQ(-1, x1, Q))/x1;
    double f1sbar = (pdf[PDF_set]->xfxQ(-3, x1, Q))/x1;
    
    // PDF combination
    double udbar_PDF = (f1u*f2dbar + f2u*f1dbar) * pow(Vud,2) + (f1u*f2sbar + f2u*f1sbar) * pow(Vus,2) + (f1c*f2dbar + f2c*f1dbar) * pow(Vcd,2) + (f1c*f2sbar + f2c*f1sbar) * pow(Vcs,2);
    
    return dsigma * udbar_PDF * djac;
}


// D-UBAR Matrix element (W-)
double dubar_funct (double *x, size_t dim, void *jj)
{
    
    (void)(dim); /* avoid unused parameter warnings */
    
    // Read paramters from structure
    Parameters *params = (Parameters *)jj;
    int PDF_set = (params->PDF_set);
    double Mt = (params->Mt);
    
    // Integration variables
    double yreduced, etareduced;
    double djac = jac(yreduced, etareduced, x, params);
    
    
    // eta
    double eta = etareduced / (1 - pow(etareduced,2));
    // costheta
    double costheta = cos(2 * atan(exp(-eta)));
    // Tansverse momentum (assuming \deltaphi = PI)
    double pT = Mt / 2;
    // Invariant mass
    double Minv = 2 * pT / sqrt(1 - pow(costheta,2));
    
    
    // Partonic cross section parameters
    double Q = Minv;
    double z = pow(Minv,2)/pow(energy,2);
    double y = -(1.0/2.0)*log(z)*(yreduced);
    double x1 = sqrt(z)*exp(y);
    double x2 = sqrt(z)*exp(-y);
    
    // Check on invariant mass
    if (Minv > energy) return 0;
    // Check on rapidity
//     if ( (y < y_min) || (y > y_max and y_max != 0)) return 0;
    
    double dsigma = GeVtofb / (768 * pow(M_PI,2)) * pow(Minv,2) / 4 * pow(1 + costheta,2); // Born matrix element squared without propagator
    
    double GammaWprime1 = GammaWprime(gsm, MWprime1);
    double GammaWprime2 = GammaWprime(gsm, MWprime2);
        
    double propagator = 0.0;
    propagator += pow(gsm,4) * (1 / (pow(pow(Minv,2) - pow(MW,2),2) + pow(GammaW,2) * pow(MW,2))); // W^2
    propagator += pow(gsm,4) * (1 / (pow(pow(Minv,2) - pow(MWprime1,2),2) + pow(GammaWprime1,2) * pow(MWprime1,2))); // W'^2
    propagator += pow(gsm,4) * (1 / (pow(pow(Minv,2) - pow(MWprime2,2),2) + pow(GammaWprime2,2) * pow(MWprime2,2))); // W''^2
    propagator += pow(gsm,2) * pow(gsm,2) * (2 * ((pow(Minv,2) - pow(MW,2)) * ((pow(Minv,2) - pow(MWprime1,2))) + MW * GammaW * MWprime1 * GammaWprime1) / (((pow(pow(Minv,2) - pow(MW,2),2) + pow(GammaW,2) * pow(MW,2))) * ((pow(pow(Minv,2) - pow(MWprime1,2),2) + pow(GammaWprime1,2) * pow(MWprime1,2))))); // W - W'
    propagator += pow(gsm,2) * pow(gsm,2) * (2 * ((pow(Minv,2) - pow(MW,2)) * ((pow(Minv,2) - pow(MWprime2,2))) + MW * GammaW * MWprime2 * GammaWprime2) / (((pow(pow(Minv,2) - pow(MW,2),2) + pow(GammaW,2) * pow(MW,2))) * ((pow(pow(Minv,2) - pow(MWprime2,2),2) + pow(GammaWprime2,2) * pow(MWprime2,2))))); // W - W''
    propagator += pow(gsm,2) * pow(gsm,2) * (2 * ((pow(Minv,2) - pow(MWprime1,2)) * ((pow(Minv,2) - pow(MWprime2,2))) + MWprime1 * GammaWprime1 * MWprime2 * GammaWprime2) / (((pow(pow(Minv,2) - pow(MWprime1,2),2) + pow(GammaWprime1,2) * pow(MWprime1,2))) * ((pow(pow(Minv,2) - pow(MWprime2,2),2) + pow(GammaWprime2,2) * pow(MWprime2,2))))); // W' - W''
    
    dsigma *= propagator; // Multiply by propagator
    dsigma *= 1 / pow(energy,2); // Jacobian x1,x2 -> Minv^2, Y
    dsigma *= -0.5 * log(z); // Jacobian Y, Yreduced
    dsigma *= 4 / (1 - pow(costheta,2)); // Jacobian Minv^2 -> pT^2
    dsigma *= 2 * pT; // Jacobian pT^2 -> pT
    dsigma *= 0.5; // Jacobian pT -> MT
    dsigma *= 2 * M_PI; // Integration in phi
    dsigma *= (1 - pow(costheta,2)); // Jacobian costheta -> eta
    dsigma *= (1 + pow(etareduced,2)) / pow(1 - pow(etareduced,2),2); // Jacobian eta -> etareduced
    
    // Partons PDF
    double f1d = (pdf[PDF_set]->xfxQ(1, x1, Q))/x1;
    double f1s = (pdf[PDF_set]->xfxQ(3, x1, Q))/x1;
    double f2ubar = (pdf[PDF_set]->xfxQ(-2, x2, Q))/x2;
    double f2cbar = (pdf[PDF_set]->xfxQ(-4, x2, Q))/x2;
    double f2d = (pdf[PDF_set]->xfxQ(1, x2, Q))/x2;
    double f2s = (pdf[PDF_set]->xfxQ(3, x2, Q))/x2;
    double f1ubar = (pdf[PDF_set]->xfxQ(-2, x1, Q))/x1;
    double f1cbar = (pdf[PDF_set]->xfxQ(-4, x1, Q))/x1;
    
    // PDF combination
    double dubar_PDF = (f1d*f2ubar + f2d*f1ubar) * pow(Vud,2) + (f1d*f2cbar + f2d*f1cbar) * pow(Vcd,2) + (f1s*f2ubar + f2s*f1ubar) * pow(Vus,2) + (f1s*f2cbar + f2s*f1cbar) * pow(Vcs,2);
    
    return dsigma * dubar_PDF * djac;   
}



double Wplus_function (Parameters *params)
{
    
    double udbar, err_udbar;
    
    Integration(&udbar_funct, dim_integration, max_eval, epsrel, epsabs, udbar, err_udbar, params, calls_warmupfactor, iterations_warmup, iterations_main);

    return udbar;
}


double Wminus_function (Parameters *params)
{
    
    double dubar, err_dubar;
    
    Integration(&dubar_funct, dim_integration, max_eval, epsrel, epsabs, dubar, err_dubar, params, calls_warmupfactor, iterations_warmup, iterations_main);

    return dubar;
}


int main(int argc, char** argv)
{
    struct Parameters params;
    
    // check on the rapidity cut
    if (y_min >= eta_cut) {
        printf("\nThe chosen lower rapidity cut is higher than pseudorapidity cut.\n\n");
        return 0;
    }
    
//     // Set PDF table
//     int PDF_set = 0;
//     params.PDF_set = PDF_set;

//     char c = static_cast<char>(MWprime);
    // Output file name
    char filename[45];
    strcpy(filename, "Wpr_");
//     strcpy(filename, "_");
// //     strcpy(filename, &c);
//     strcpy(filename, "6600");
//     strcpy(filename, "_");
    strcat(filename, setname);
    strcat(filename, ".dat");
    
    // Time counter
    std::chrono::steady_clock::time_point begin, end;
    
    ofstream out(filename);    
    
    if (y_min / log(energy/Mt_max) > 1) {
        printf("\nThe chosen lower rapidity cut is too high in this transverse mass range.\n\n");
        return 0;
    }

    // Transverse mass points
    vector<double> Mt_table;
    // Invariant mass table
    Mt_table.push_back(Mt_min);
    for (int i = 1; i <= points_Mt; i++) {
        Mt_table.push_back(Mt_table[i-1]+step);
    }
    
    string text;
    double Wplus, Wminus;
    
    // W+ spectrum
    cout << "Computing W+ Transverse mass spectrum" << endl;
    begin = std::chrono::steady_clock::now();
    text = "Wprpluseigenpoints[" + std::to_string(MWprime1) + ", " + std::to_string(MWprime2) + "] = {{";
    
    for (size_t PDF_set = 0; PDF_set <= nmem; PDF_set++) {
        
        cout << "Computing eigenvector #" << PDF_set << endl;
        
        for (int i = 0; i <= points_Mt; i++) {
            
            params.Mt = Mt_table[i];
            params.PDF_set = PDF_set;
            
            Wplus = Wplus_function(&params);
                        
            stringstream Mt_string;
            Mt_string << fixed << setprecision(3) << Mt_table[i];
            string Mt_s = Mt_string.str();
            
            stringstream data_string;
            data_string << fixed << scientific << setprecision(16) << Wplus;
            string data_s = data_string.str();
            data_s = regex_replace(data_s, regex("e"), "*^");
            
            text = text + "{" + Mt_s + ", " + data_s + "}, ";
        }
        
        text = text.substr(0, text.size()-2);
        text = text + "}, {";
    }
    
    text = text.substr(0, text.size()-3);
    text = text + "}";
    text = text + "\n\n";
    out << text;
    
    end = std::chrono::steady_clock::now();
    cout << "W+ eigenvectors computed in " << std::chrono::duration_cast<std::chrono::seconds>(end - begin).count() << " sec." << endl;
    
    
    // W- spectrum
    cout << "Computing W- Transverse mass spectrum" << endl;
    begin = std::chrono::steady_clock::now();
    text = "Wprminuseigenpoints[" + std::to_string(MWprime1) + ", " + std::to_string(MWprime2) + "] = {{";
    
    for (size_t PDF_set = 0; PDF_set <= nmem; PDF_set++) {
        
        cout << "Computing eigenvector #" << PDF_set << endl;
        
        for (int i = 0; i <= points_Mt; i++) {
            
            params.Mt = Mt_table[i];
            params.PDF_set = PDF_set;
            
            Wminus = Wminus_function(&params);
                        
            stringstream Mt_string;
            Mt_string << fixed << setprecision(3) << Mt_table[i];
            string Mt_s = Mt_string.str();
            
            stringstream data_string;
            data_string << fixed << scientific << setprecision(16) << Wminus;
            string data_s = data_string.str();
            data_s = regex_replace(data_s, regex("e"), "*^");
            
            text = text + "{" + Mt_s + ", " + data_s + "}, ";
        }
        
        text = text.substr(0, text.size()-2);
        text = text + "}, {";
    }
    
    text = text.substr(0, text.size()-3);
    text = text + "}";
    text = text + "\n\n";
    out << text;
    
    end = std::chrono::steady_clock::now();
    cout << "W- eigenvectors computed in " << std::chrono::duration_cast<std::chrono::seconds>(end - begin).count() << " sec." << endl;

    out.close();
    
//     cout << "GammaWprime1/MWprime1 = " << GammaWprime(gsm, MWprime1)/MWprime1 << endl;
//     cout << "GammaWprime1/MWprime2 = " << GammaWprime(gsm, MWprime2)/MWprime2 << endl;
//     cout << "gsm = " << gsm << endl;
    
    return 0;
}
