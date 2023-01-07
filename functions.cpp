#define DB_MIN_IEEE -DBL_MAX 
#define LOG_MAX std::log(DBL_MAX)

#include <cmath>
#include <RcppArmadillo.h>

using namespace Rcpp;

// Enable C++11 via this plugin (Rcpp 0.10.3 or later)
// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::depends(RcppArmadillo)]]

/*
Assign the depth data to state of the dive it is belong to
Input:
- dstart: the onset time of each dive
- depth: the depth data
- data: the dataframe includes the onset, the end, and the state of each dive
Output: the state coressponds to each depth datapoint
*/
// [[Rcpp::export]]
std::vector<double> Depth2State(arma::vec& dstart,arma::vec& depth,DataFrame data)
{
	NumericVector start  = data["Start"];
	NumericVector end    = data["End"];
	NumericVector State  = data["State"];
	int nbObs = depth.n_rows, i = 0, j = 0; 
	arma::vec state(nbObs);	state.zeros();
	while(i<nbObs)
		if ( (dstart[i] >= start[j]) 
			  && (dstart[i] < start[j+1]) ) {
			if (dstart[i] <= end[j])
				state[i] = State[j]; 
			i++; 
		} else j++;

	// http://stackoverflow.com/a/32582650
	return as<std::vector<double> >(wrap(state)); 
}

