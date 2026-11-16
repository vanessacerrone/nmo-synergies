// Probabilities from the original NuFast-Earth engine along one path of layers, for tests/test_nufast_earth.py.
// stdin: s12sq s13sq s23sq delta dm21 dm31 neutrino(1|0); then n_layers and the pairs length [km], rho*Ye [g/cm^3];
// then n_energies and the energies [GeV]. stdout: for each energy, P(alpha -> beta) for alpha, beta = e, mu, tau.
#include <cstdio>
#include <utility>
#include <vector>

#include "NuFastEarth.h"

int main()
{
	double s12sq, s13sq, s23sq, delta, dm21, dm31;
	int neutrino, n_layers, n_energies;
	if (scanf("%lf %lf %lf %lf %lf %lf %d", &s12sq, &s13sq, &s23sq, &delta, &dm21, &dm31, &neutrino) != 7) return 1;
	if (scanf("%d", &n_layers) != 1) return 1;
	std::vector<std::pair<double, double>> trajectory(n_layers);
	for (auto &layer : trajectory)
		if (scanf("%lf %lf", &layer.first, &layer.second) != 2) return 1;
	if (scanf("%d", &n_energies) != 1) return 1;
	std::vector<double> energies(n_energies);
	for (auto &e : energies)
		if (scanf("%lf", &e) != 1) return 1;

	NuFast::Probability_Engine engine;
	engine.Set_Oscillation_Parameters(s12sq, s13sq, s23sq, delta, dm21, dm31, neutrino == 1);
	engine.Set_Trajectory(trajectory);
	engine.Set_E_Spectra(energies);
	std::vector<std::vector<NuFast::Matrix3r>> prob = engine.Get_Probabilities();
	for (int i = 0; i < n_energies; i++)
		for (int a = 0; a < 3; a++)
			for (int b = 0; b < 3; b++)
				printf("%.17g%c", prob[i][0].arr[a][b], (a == 2 && b == 2) ? '\n' : ' ');
	return 0;
}
