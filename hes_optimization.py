import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, differential_evolution
from scipy.stats.qmc import LatinHypercube
from multiprocessing import Pool
from time import time
import pickle

from dymola_wrapper import generate_simulator


def trapint(t, x):
    """ Numerical integration by the trapezoidal rule """
    return np.sum(np.diff(t) * 0.5 * (x[:-1] + x[1:]))


def generate_lhc_points(N, bounds):
    lhc = LatinHypercube(d=len(bounds))
    samples = lhc.random(N)

    lower_bounds = np.array([b[0] for b in bounds])
    upper_bounds = np.array([b[1] for b in bounds])
    
    samples = samples * (upper_bounds - lower_bounds) + lower_bounds

    return samples


def main():
    # Generate an objective function using a Dymola simulation wrapper
    project_path = 'C:\\Users\\A01631741\\Box Sync\\Dymola Install & Libraries\\HESProjectModel\\HESProject.mo'  # TODO: Change this
    model = 'HESProject.HESModel5'
    design_params = ['N_reactors', 'N_windturbines', 'N_solarpanels']
    output_path = 'G:\\My Drive\\Hybrid Energy Systems (MAE 5450)\\Project\\misc_python\\simulation_results\\sim_results'  # TODO: Change this
    
    simulator, interface = generate_simulator(project_path,
                                              model,
                                              design_params,
                                              output_path)

    lcoe_Natrium = 70.59  # CF: 90%
    lcoe_Wind = 36.93  # CF: 41%
    lcoe_SolarPV = 30.43  # CF: 29%; Note: single axis tracking...
    lcoe_electrolysis = 50
    lcoe_H2_storage = 20
    lcoe_fuel_cell = 50
    H2_sale_price = 2.50

    # LCOE_subsys_max = max(LCOE_Natrium, LCOE_Wind, LCOE_SolarPV, LCOE_H2)  # We'll use this when calculating penalty

    def f_obj(x):
        """ Optimization objective function. This calls the Dymola wrapper we created above to run a simulation for the design point x.
        
        A few notes:
            - The names of the model components and their variables are hard-coded here. This code will break if any of those names are changed.
            - The simulation doesn't necessarily take equal time steps, so numerical integration is required to convert each generation and load term from units to power to units of energy.
            - The simulation takes ~5 seconds to run, so be careful how you use it!
            - The simulation wrapper can't be run in parallel!
            - The penalty setup here is arbitrary. Do what you want.
        """

        # Run Dymola simulation for the given design point
        data, params = simulator(list(x))

        # Put simulation results into numpy arrays. Units are all Joules or kg.
        t = data['Time'].to_numpy()
        nuclear = data['nuclearSystem.E_tot_nuclear'].to_numpy()  # [W]
        wind = data['windFarm.PowerOut'].to_numpy()  # [W]
        solar = data['solarFarm2.PowerOut'].to_numpy()  # [W]
        electrolysis = data['hydrogenStorage.power_a'].to_numpy()  # [W]
        electrolysis[electrolysis > 0] = 0  # All of the positive numbers are for demand to get filled, so get rid of those
        electrolysis *= -1
        fuel_cell = data['hydrogenStorage.power_out'].to_numpy()  # [W]
        
        stored_h2 = data['hydrogenStorage.M_H2'].to_numpy()  # [kg]
        m_H2_sold = data['hydrogenStorage.H2_sold'].to_numpy()[-1]  # [kg]
        H2_sell_rate = data['hydrogenStorage.H2_sell_rate'].to_numpy()  # [kg/s]

        demand = data['Load.y'].to_numpy()
        unmet_demand = data['hydrogenStorage.unmet_demand'].to_numpy()

        if not np.all(np.isclose(unmet_demand, 0)):
            return 1e6

        # Integrate power output data to get total energy, converting from [J] to [MWh]
        E_Natrium = trapint(t, nuclear) / 3.6e9
        E_Wind = trapint(t, wind) / 3.6e9
        E_Solar = trapint(t, solar) / 3.6e9
        E_Electrolysis = trapint(t, electrolysis) / 3.6e9
        E_FuelCell = trapint(t, fuel_cell) / 3.6e9
        total_demand = trapint(t, demand) / 3.6e9

        # Capacity factors adjust the subsystem LCOEs
        nuclear_cap = 100 * (E_Natrium / x[0]) / (345 * 366 * 24) / 90  # 100%
        wind_cap = 100 * (E_Wind / x[1]) / (3.2 * 366 * 24) / 41  # 47.4%
        solar_cap = 100 * (E_Solar / x[2]) / (150e-6 * 366 * 24) / 29  # 24.1%
        electrolysis_cap = 100 * (E_Electrolysis) / (max(electrolysis) * 1e-6 * 366 * 24) / 25  # 19.7%
        hfc_cap = 100 * E_FuelCell / (max(fuel_cell) * 1e-6 * 366 * 24) / 25  # 3.1%

        LCOE_Natrium = lcoe_Natrium / nuclear_cap
        LCOE_Wind = lcoe_Wind / wind_cap
        LCOE_SolarPV = lcoe_SolarPV / solar_cap
        LCOE_electrolysis = lcoe_electrolysis / electrolysis_cap
        LCOE_H2_storage = lcoe_H2_storage
        LCOE_fuel_cell = lcoe_fuel_cell / hfc_cap
        
        # Calculate what each kg of excess hydrogen we're making is costing/making us
        n1 = nuclear * LCOE_Natrium + wind * LCOE_Wind + solar * LCOE_SolarPV  # total power generation cost (excl. hydrogen stuff)
        n2 = nuclear + wind + solar  # total power generation (excl. hydrogen stuff)
        n3 = (n2 - demand) / 3.6e9  # excess production
        n4 = n3 * (n1 / n2 + LCOE_electrolysis + LCOE_H2_storage)
        n4[np.isclose(H2_sell_rate, 0)] = 0  # Makes sure we're only counting overproduction that is going to sold H2 (i.e. not counting when we're just refilling our storage)
        C_prod = trapint(t, n4)
        h2_revenue = H2_sale_price * m_H2_sold
        num = C_prod - h2_revenue
        den = trapint(t, n4)
        LCOE_overproduction_eff = num / den  # Cost per MWh (not really an LCOE but whatever)... Positive means that it's actually costing us money (and should be raising the total LCOE of the system)
        H2_prod_cost = C_prod / m_H2_sold
        H2_profit_per_kg = H2_sale_price - H2_prod_cost  # This seems to be giving us good numbers so I'm pretty confident in the work here now.

        # Calculate the LCOE of the system as a weighted average of the component LCOEs
        subsystem_costs = np.array([E_Natrium * LCOE_Natrium,
                                    E_Wind * LCOE_Wind,
                                    E_Solar * LCOE_SolarPV,
                                    E_Electrolysis * LCOE_electrolysis,
                                    E_Electrolysis * LCOE_H2_storage,
                                    E_FuelCell * LCOE_fuel_cell,
                                    # -m_H2_sold * H2_sale_price])
                                    -m_H2_sold * H2_prod_cost])
        LCOE_sys = np.sum(subsystem_costs) / total_demand
        # total_cost = np.sum(subsystem_costs) * 1e-9  # billions of dollars per year
        
        # Penalize any solutions which have unmet demand. We want this term to be sized so that it is an order of magnitude less than LCOE_sys at the point where the unmet demand amount becomes "acceptable" for our design.
        # We'll calculate this as wanting less than 0.01% of the total demand to go unmet. As scaled here, the penalty should be equal to LCOE_subsys_max/10 when the D_rem/D_tot = 0.01%.
        # Squaring the D_rem/D_tot term serves to proportionally penalize more heavily designs above the threshold than those below it.
        # penalty = 1e7 * LCOE_subsys_max * (unmet_demand / total_demand) ** 2

        print(x, LCOE_sys, m_H2_sold * 1e-9)

        # print(trapint(t, n2 - demand))
        # inds_not_selling_h2 = np.isclose(H2_sell_rate, 0)
        excess_power = n2 - demand
        prods = np.array([nuclear, wind, solar])
        prods[:, excess_power < 0] = 0
        excess_power[excess_power < 0] = 0
        totals = np.sum(prods, axis=1) / np.sum(prods)

        print('Hydrogen cost per kg:')
        print(H2_prod_cost)
        
        print('Subsystem Costs')
        print(subsystem_costs / np.sum(subsystem_costs))
        
        print('LCOEs')
        print(LCOE_Natrium, LCOE_Wind, LCOE_SolarPV, LCOE_electrolysis, LCOE_fuel_cell)
        
        return LCOE_sys

    bounds = [(1, 80), (5e3, 15e3), (5e7, 1e9)]
    # N = 1000
    # samples = np.round(generate_lhc_points(N=N, bounds=bounds))
    
    # start = time()
    
    # results = np.zeros((len(samples), 2))
    
    # for i, x in enumerate(samples):
    #     istart = time()
    #     results[i] = f_obj(x)
    #     istop = time()
    #     print('{:4d} / {:4d} \t\t Exec Time: {:.2f}'.format(i + 1, N, istop - istart))
    
    # stop = time()
    
    # np.savetxt('misc_python\\sim_data\\samples.csv', samples)
    # np.savetxt('misc_python\\sim_data\\results.csv', results)
    
    # print(results)
    # print('\nTime Elapsed:', stop - start)
    # print('\nAvg Exec Time:', (stop - start) / N)

    # print(f_obj([60, 7410, 3.8425e8]))

    # opt_res = differential_evolution(f_obj, bounds)
    # print(opt_res)

    with open('opt_res.pkl', 'rb') as f:
        opt_res = pickle.load(f)
    print(opt_res)

    print(f_obj(opt_res.x))

    interface.close()


if __name__ == '__main__':
    main()
