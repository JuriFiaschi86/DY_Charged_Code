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

// Constants
#define PI 3.14159265
#define GeVtofb 0.389379338e+12

// SM parameters
#define alphaEM 7.29735e-3
#define stheta2W 0.23127

#define MW 80.425
#define GammaW 2.124

// Gauge coupling
const double gsm = sqrt(4*PI*alphaEM/stheta2W);
// CKM Matrix
const double Vud = 0.975;
const double Vus = 0.222;
const double Vub = 0.0;
const double Vcd = 0.222;
const double Vcs = 0.975;
const double Vcb = 0.0;
const double Vtd = 0.0;
const double Vts = 0.0;
const double Vtb = 1.0;

// Collider energy
const double energy = 8000;

// Acceptance cuts
const double eta_cut = 2.5;
const double pT_cut = 25;

// Invariant mass cuts
const double Minv_min = 50;
const double Minv_max = energy;

// x1,x2 integration bounds
const double x1_min = pow(Minv_min / energy, 2);
const double x1_max = 1.0;
const double x2_min = x1_min;
const double x2_max = 1.0;

// Eta scan points
const double eta_l_min = -2.5;
const double eta_l_max = 2.5;
const double step = 0.05;
const int points_eta_l = (eta_l_max - eta_l_min) / step;


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

// Initialize PDF set
const LHAPDF::PDFSet set(setname);
// const size_t nmem = set.size()-1;
const size_t nmem = 0;
const vector<LHAPDF::PDF*> pdf = set.mkPDFs();

// Settings of the integration
const int dim_integration = 2; // Integration in yreduced, Minv
// Integration parameters
const int iterations_warmup = 5;
const double calls_warmupfactor = 0.1;
const int iterations_main = 3;
// Integration precision
const double epsrel = 1e-3;
const double epsabs = 1e-12;
const int max_eval = 35000;


// Parameters structure
struct Parameters {
    int max_iters = 10;
    int PDF_set = 0;
    double eta_l;
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


double jac (double &x1, double &x2, double *x, Parameters *params)
{
    double eta_l = params->eta_l;
    double djac = 1.0;
        
    // Log sampling
    x1 = x1_min * pow(x1_max / x1_min, x[0]);
    djac *= x1 * log(x1_max / x1_min);
    x2 = x2_min * pow(x2_max / x2_min, x[1]);
    djac *= x2 * log(x2_max / x2_min);
    
//     // Linear sampling
//     x1 = (x1_max - x1_min) * x[0] + x1_min;
//     djac *= x1_max - x1_min;
//     x2 = (x2_max - x2_min) * x[1] + x2_min;
//     djac *= x2_max - x2_min;

    
    double Minv = sqrt(x1*x2)*energy;
    
    if ((Minv < Minv_min) || (Minv > Minv_max)) djac = 0.0;
    
    double y = 0.5 * log(x1/x2);
    double eta = eta_l - y;
    
    if ((eta_l < -eta_cut) || (eta_l > eta_cut)) djac = 0.0; // This should be the correct cut
    
    double costheta = cos(2 * atan(exp(-eta)));
    double pT = (Minv / 2) * sqrt(1 - pow(costheta,2)) ;
    
    if (pT < pT_cut) djac = 0.0;
    
    return djac;
}



// U-DBAR Matrix element (W+)
double udbar_funct (double *x, size_t dim, void *jj)
{
    
    (void)(dim); /* avoid unused parameter warnings */
    
    // Read paramters from structure
    Parameters *params = (Parameters *)jj;
    int PDF_set = (params->PDF_set);
    double eta_l = (params->eta_l);
    
    // Integration variables
    double x1, x2;
    double djac = jac(x1, x2, x, params);
    
    double Minv = sqrt(x1*x2)*energy;
    
    if (Minv > energy) djac = 0.0;
    
    double y = 0.5 * log(x1/x2);
    double eta = eta_l - y;
    double costheta = cos(2 * atan(exp(-eta)));
    
    // Factorization scale
    double Q = Minv;
    
    double dsigma = GeVtofb * pow(gsm,4) / (768 * pow(PI,2)) * pow(Minv,2) / 4 * pow(1 + costheta,2); // Born matrix element squared without propagator
    double propagator = 1 / (pow(pow(Minv,2) - pow(MW,2),2) + pow(GammaW,2) * pow(MW,2)); // Propagator single W    
//     double propagator = 1 / (pow(pow(Minv,2) - pow(MW,2),2) + pow(GammaW,2) * pow(MW,2)) + 1 / (pow(pow(Minv,2) - pow(MWprime,2),2) + pow(GammaWprime,2) * pow(MWprime,2)) + 2 * ((pow(Minv,2) - pow(MW,2)) * ((pow(Minv,2) - pow(MWprime,2))) + MW * GammaW * MWprime * GammaWprime) / (((pow(pow(Minv,2) - pow(MW,2),2) + pow(GammaW,2) * pow(MW,2))) * ((pow(pow(Minv,2) - pow(MWprime,2),2) + pow(GammaWprime,2) * pow(MWprime,2)))); // Propagator W + W' +Interf    
    
    dsigma *= propagator; // Multiply by propagator
    dsigma *= 2 * PI; // Integration in phi
    dsigma *= (1 - pow(costheta,2)); // Jacobian costheta -> eta
    
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
    double eta_l = (params->eta_l);
    
    // Integration variables
    double x1, x2;
    double djac = jac(x1, x2, x, params);
    
    double Minv = sqrt(x1*x2)*energy;
    
    if (Minv > energy) djac = 0.0;
    
    double y = 0.5 * log(x1/x2);
    double eta = eta_l - y;
    double costheta = cos(2 * atan(exp(-eta)));
    
    // Factorization scale
    double Q = Minv;
    
    double dsigma = GeVtofb * pow(gsm,4) / (768 * pow(PI,2)) * pow(Minv,2) / 4 * pow(1 + costheta,2); // Born matrix element squared without propagator
    double propagator = 1 / (pow(pow(Minv,2) - pow(MW,2),2) + pow(GammaW,2) * pow(MW,2)); // Propagator single W    
//     double propagator = 1 / (pow(pow(Minv,2) - pow(MW,2),2) + pow(GammaW,2) * pow(MW,2)) + 1 / (pow(pow(Minv,2) - pow(MWprime,2),2) + pow(GammaWprime,2) * pow(MWprime,2)) + 2 * ((pow(Minv,2) - pow(MW,2)) * ((pow(Minv,2) - pow(MWprime,2))) + MW * GammaW * MWprime * GammaWprime) / (((pow(pow(Minv,2) - pow(MW,2),2) + pow(GammaW,2) * pow(MW,2))) * ((pow(pow(Minv,2) - pow(MWprime,2),2) + pow(GammaWprime,2) * pow(MWprime,2)))); // Propagator W + W' +Interf    
    
    dsigma *= propagator; // Multiply by propagator
    dsigma *= 2 * PI; // Integration in phi
    dsigma *= (1 - pow(costheta,2)); // Jacobian costheta -> eta
    
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
    
    

    // Output file name
    char filename[45];
    strcpy(filename, "AW_");
    strcat(filename, setname);
    strcat(filename, ".dat");
    // Time counter
    std::chrono::steady_clock::time_point begin, end;
    
    ofstream out(filename);    
    
//     if (y_min / log(energy/Minv_max) > 1) {
//         printf("\nThe chosen lower rapidity cut is too high in this transverse mass range.\n\n");
//         return 0;
//     }

    // Eta points
    vector<double> eta_l_table;
    // Invariant mass table
    eta_l_table.push_back(eta_l_min);
    for (int i = 1; i < points_eta_l; i++) {
        eta_l_table.push_back(eta_l_table[i-1]+step);
    }
    
    string text;
    double Wplus, Wminus;
    
    // W+ spectrum
    cout << "Computing W+ eta spectrum" << endl;
    begin = std::chrono::steady_clock::now();
    text = "Wpluseigenpoints = {{";
    
    for (size_t PDF_set = 0; PDF_set <= nmem; PDF_set++) {
        
        cout << "Computing eigenvector #" << PDF_set << endl;
        
        for (int i = 0; i < points_eta_l; i++) {
            
            params.eta_l = eta_l_table[i];
            params.PDF_set = PDF_set;
            
            Wplus = Wplus_function(&params);
                        
            stringstream eta_l_string;
            eta_l_string << fixed << setprecision(3) << eta_l_table[i];
            string eta_l_s = eta_l_string.str();
            
            stringstream data_string;
            data_string << fixed << scientific << setprecision(16) << Wplus;
            string data_s = data_string.str();
            data_s = regex_replace(data_s, regex("e"), "*^");
            
            text = text + "{" + eta_l_s + ", " + data_s + "}, ";
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
    cout << "Computing W- eta spectrum" << endl;
    begin = std::chrono::steady_clock::now();
    text = "Wminuseigenpoints = {{";
    
    for (size_t PDF_set = 0; PDF_set <= nmem; PDF_set++) {
        
        cout << "Computing eigenvector #" << PDF_set << endl;
        
        for (int i = 0; i < points_eta_l; i++) {
            
            params.eta_l = eta_l_table[i];
            params.PDF_set = PDF_set;
            
            Wminus = Wminus_function(&params);
                        
            stringstream eta_l_string;
            eta_l_string << fixed << setprecision(3) << eta_l_table[i];
            string eta_l_s = eta_l_string.str();
            
            stringstream data_string;
            data_string << fixed << scientific << setprecision(16) << Wminus;
            string data_s = data_string.str();
            data_s = regex_replace(data_s, regex("e"), "*^");
            
            text = text + "{" + eta_l_s + ", " + data_s + "}, ";
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
    
    
    // AW spectrum
    cout << "Computing AW eta spectrum" << endl;
    begin = std::chrono::steady_clock::now();
    text = "AWeigenpoints = {{";
    
    for (size_t PDF_set = 0; PDF_set <= nmem; PDF_set++) {
        
        cout << "Computing eigenvector #" << PDF_set << endl;
        
        for (int i = 0; i < points_eta_l; i++) {
            
            params.eta_l = eta_l_table[i];
            params.PDF_set = PDF_set;
            
            Wplus = Wplus_function(&params);
            Wminus = Wminus_function(&params);
                        
            stringstream eta_l_string;
            eta_l_string << fixed << setprecision(3) << eta_l_table[i];
            string eta_l_s = eta_l_string.str();
            
            stringstream data_string;
            data_string << fixed << scientific << setprecision(16) << (Wplus - Wminus)/(Wplus + Wminus);
            string data_s = data_string.str();
            data_s = regex_replace(data_s, regex("e"), "*^");
            
            text = text + "{" + eta_l_s + ", " + data_s + "}, ";
        }
        
        text = text.substr(0, text.size()-2);
        text = text + "}, {";
    }
    
    text = text.substr(0, text.size()-3);
    text = text + "}";
    text = text + "\n\n";
    out << text;
    
    end = std::chrono::steady_clock::now();
    cout << "AW eigenvectors computed in " << std::chrono::duration_cast<std::chrono::seconds>(end - begin).count() << " sec." << endl;

    out.close();
    
    return 0;
}
